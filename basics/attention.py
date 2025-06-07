import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()

        assert d_model % n_head == 0

        self.n_head = n_head
        # d_model here means n_embed
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

        # MultiHead Attention needs a final linear layer
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, idx):
        B, T, C = idx.shape
        # here C should be equal to d_model or n_embed
        assert C == self.d_model
        q = self.Q(idx)
        k = self.K(idx)
        v = self.V(idx)
        # hs: head size
        hs = C // self.n_head

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        qk = q @ k.transpose(2, 3) / math.sqrt(hs)

        # masking in decoder
        mask = torch.tril(torch.ones(T, T))
        qk = qk.masked_fill(mask == 0, float('-inf'))

        # apply softmax on the last dimension, which is the head size
        qkv = F.softmax(qk, dim=-1) @ v
        # contiguous() is used to make sure the tensor is continuous in memory
        qkv = qkv.transpose(1, 2).contiguous().view(B, T, C)
        qkv = self.fc(qkv)
        return qkv


def GroupQueryAttention():
    def __init__(self, n_head, d_model, n_group):
        super().__init__()

        assert d_model % n_head == 0
        assert n_head % n_group == 0

        self.n_group = n_group
        self.n_head = n_head
        # d_model here means n_embed
        self.d_model = d_model
        # hs: head size
        self.hs = self.d_model // self.n_head
        # gs: group size
        self.gs = self.n_head // self.n_group

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, self.hs * n_group)
        self.V = nn.Linear(d_model, self.hs * n_group)

        # MultiHead Attention needs a final linear layer
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, idx):
        B, T, C = idx.shape
        # here C should be equal to d_model or n_embed
        assert C == self.d_model
        q = self.Q(idx)
        k = self.K(idx)
        v = self.V(idx)

        # q = q.view(B, T, self.n_group * self.gs, self.hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        k = k.view(B, T, self.n_group, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_group, self.hs).transpose(1, 2)

        # expand the k and v
        k = self._expand(k)
        v = self._expand(v)

        qk = q @ k.transpose(2, 3) / math.sqrt(self.hs)

        # masking in decoder
        mask = torch.tril(torch.ones(T, T))
        qk = qk.masked_fill(mask == 0, float('-inf'))

        # apply softmax on the last dimension, which is the head size
        qkv = F.softmax(qk, dim=-1) @ v
        # contiguous() is used to make sure the tensor is continuous in memory
        qkv = qkv.transpose(1, 2).contiguous().view(B, T, C)
        qkv = self.fc(qkv)
        return qkv

    def _expand(self, x):
        B, C, T = x.shape
        # return x.view(B, T, self.n_group, self.gs, self.hs).transpose(1, 2).contiguous().view(B, T, self.n_head, self.hs)
        x = x[:, :, :, None, :].expand(B, self.n_group, self.gs, T, self.hs)
        x = x.contiguous().view(B, self.n_head, T, self.hs)
        return x