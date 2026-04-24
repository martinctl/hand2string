import torch.nn as nn


class MLP(nn.Module):
    # Baseline for static signs (e.g. ASL alphabet).
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        raise NotImplementedError
