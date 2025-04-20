import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

    def forward(self, idx):
        B, T, C = idx.shape
        assert C // self.n_head == self.d_model
        idx = idx.view(-1, B, self.n_head, T, C // self.n_head)
        q = self.Q(idx)
        k = self.K(idx)
        v = self.V(idx)

        qk = q @ k / np.sqrt(self.d_model)
        qk = F.soft_max(qk.view(-1)).view(-1, B, self.n_head, T, C // self.n_head)
        qkv = qk @ v 

        return qkv
