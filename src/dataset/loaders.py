# PyTorch Datasets wrapping extracted landmark sequences.


class LandmarkSequenceDataset:
    def __init__(self, root: str, split: str, window_size: int):
        raise NotImplementedError
