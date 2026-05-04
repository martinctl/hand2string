import torch.nn as nn


class MLP(nn.Module):
    """Baseline for static signs (e.g. ASL alphabet).

    Input: flattened landmark vector (e.g. 21*3 = 63 for one hand).
    Output: class logits.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
