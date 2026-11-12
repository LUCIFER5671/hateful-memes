"""Small classifiers trained on frozen embeddings."""
import torch
from torch import nn


class Classifier(nn.Module):
    """The lab's classifier: Linear -> ReLU -> (Dropout) -> Linear(1). Outputs one logit for "hateful"."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # one output: the logit for "hateful"

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class StandardizedClassifier(nn.Module):
    """Classifier with the training-set standardization stored inside it (as buffers),
    so it can be applied directly to raw embeddings: logit = net((x - mean) / std)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim) if mean is None else mean.detach().clone().float())
        self.register_buffer("std", torch.ones(input_dim) if std is None else std.detach().clone().float())
        self.net = Classifier(input_dim, hidden_dim, dropout)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)
