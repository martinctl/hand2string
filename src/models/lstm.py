import torch.nn as nn


class LSTMClassifier(nn.Module):
    # Temporal model for dynamic gestures over landmark sequences.
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super().__init__()
        raise NotImplementedError
