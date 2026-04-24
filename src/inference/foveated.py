# Foveated cropping agent: restrict MediaPipe to the previous-frame hand
# bounding box to reduce per-frame compute.


def crop_to_previous_hand(frame, prev_bbox):
    raise NotImplementedError
