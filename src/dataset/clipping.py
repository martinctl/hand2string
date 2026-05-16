"""Cut a time-range out of a source mp4 into a new mp4 using pyav.

We re-encode (libx264, CRF 23) so cut points are frame-accurate. Audio is
dropped: How2Sign videos carry no useful audio on the signer side and the
visual stream is all the model needs.

Implementation notes:
- We seek to a keyframe at-or-before ``start``, then decode-and-skip until we
  reach the requested start time. The output stream is fed frames whose PTS
  has been *rebased* to start at 0, so the encoder/muxer always see a clean
  monotonic stream (this avoids EINVAL during mux that otherwise shows up on
  some seeked re-encodes).
- We cap output frame count at the natural duration to keep encoder DTS
  monotonic and bounded.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av


_ENCODER_CANDIDATES = ("libx264", "libopenh264", "mpeg4")


def _pick_encoder() -> str:
    for name in _ENCODER_CANDIDATES:
        try:
            av.Codec(name, "w")
            return name
        except Exception:
            continue
    raise RuntimeError(
        f"no usable video encoder found; tried {_ENCODER_CANDIDATES}. "
        "Install PyAV with libx264 support (e.g. `conda install -c conda-forge av`)."
    )


def cut_clip(
    src: Path,
    start: float,
    end: float,
    dst: Path,
    crf: int = 23,
    preset: str = "veryfast",
) -> None:
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")

    src = Path(src).resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = dst.resolve()

    encoder = _pick_encoder()

    try:
        return _cut_clip_inner(src, start, end, dst, crf, preset, encoder)
    except Exception:
        if dst.exists() and dst.stat().st_size == 0:
            try:
                dst.unlink()
            except OSError:
                pass
        raise


def _cut_clip_inner(
    src: Path,
    start: float,
    end: float,
    dst: Path,
    crf: int,
    preset: str,
    encoder: str,
) -> None:
    with av.open(str(src)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"

        in_tb = in_stream.time_base or Fraction(1, 1000)
        fps = in_stream.average_rate or Fraction(25, 1)

        seek_target_av_tb = int(start / Fraction(1, av.time_base))
        try:
            in_container.seek(max(seek_target_av_tb, 0), any_frame=False, backward=True)
        except av.AVError:
            in_container.seek(0)

        with av.open(str(dst), mode="w") as out_container:
            out_stream = out_container.add_stream(encoder, rate=fps)
            out_stream.width = in_stream.width
            out_stream.height = in_stream.height
            out_stream.pix_fmt = "yuv420p"
            opts = {"movflags": "+faststart"}
            if encoder == "libx264":
                opts["crf"] = str(crf)
                opts["preset"] = preset
            elif encoder == "libopenh264":
                opts["b"] = "2M"
            elif encoder == "mpeg4":
                opts["qscale"] = "5"
            out_stream.options = opts
            out_stream.codec_context.time_base = Fraction(1, int(round(float(fps))))

            out_idx = 0
            for frame in in_container.decode(in_stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * in_tb)
                if t < start:
                    continue
                if t >= end:
                    break

                frame.pts = out_idx
                frame.dts = None
                frame.time_base = out_stream.codec_context.time_base
                out_idx += 1
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

            for packet in out_stream.encode():
                out_container.mux(packet)

    if out_idx == 0:
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"no frames in window [{start:.3f}, {end:.3f}) for {src.name}"
        )
