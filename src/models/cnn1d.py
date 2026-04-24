import torch.nn as nn


class CNN1D(nn.Module):
    # 1D-CNN alternative over temporal landmark sequences.
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        raise NotImplementedError
