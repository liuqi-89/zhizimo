# -*- coding: utf-8 -*-
"""终端用的预测骨干（state_dict 版，避免 pickle 依赖）。

两个架构：
  - GRUNetV1 : 与 v1 notebook 完全一致（1 层 GRU + 时间项），仅用于兼容旧 pickle；
  - GRUNetV4 : GRU + 正弦位置编码 + 1 层 TransformerEncoder（多头注意力）+ 残差 LayerNorm，
               即"v4 GRU-Transformer"，参数约 6.4 万；输出头与 v1 逐位相同。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _Head(nn.Module):
    """两输出头（ΔU 幅值 + 尺度），末层小初始化；与 v1 notebook 一致。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, h):
        return self.net(h)


class GRUNetV1(nn.Module):
    """v1 基准骨干。"""

    def __init__(self, n_feat: int = 10, hidden: int = 64, n_layers: int = 1,
                 init_du: float = 0.5, t_scale_h: float = 1000.0):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True)
        self.head = _Head(hidden)
        self.du0 = nn.Parameter(torch.tensor(float(init_du)))
        self.t_scale = float(t_scale_h)

    def forward(self, x, t_last):
        h = self.emb(x)
        out, _ = self.gru(h)
        z = self.head(out[:, -1, :])
        du = torch.tanh(z[:, 0]) * torch.exp(torch.clamp(z[:, 1], -4.0, 2.0))
        return du + self.du0 * torch.tanh(
            torch.log1p(torch.as_tensor(t_last, dtype=du.dtype) / self.t_scale)
        )


class PositionalEncoding(nn.Module):
    """固定正弦位置编码（不学习参数）。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


class GRUNetV4(nn.Module):
    """v4 GRU-Transformer：GRU 保留时序归纳偏置，单层自注意力补全局依赖。"""

    def __init__(self, n_feat: int = 10, hidden: int = 64, n_layers: int = 1,
                 init_du: float = 0.5, t_scale_h: float = 1000.0,
                 nhead: int = 4, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.pos = PositionalEncoding(hidden)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = _Head(hidden)
        self.du0 = nn.Parameter(torch.tensor(float(init_du)))
        self.t_scale = float(t_scale_h)

    def forward(self, x, t_last):
        h = self.pos(self.emb(x))
        h_g, _ = self.gru(h)
        h_t = self.trans(h)
        out = self.norm(h_t + h_g)
        z = self.head(out[:, -1, :])
        du = torch.tanh(z[:, 0]) * torch.exp(torch.clamp(z[:, 1], -4.0, 2.0))
        return du + self.du0 * torch.tanh(
            torch.log1p(torch.as_tensor(t_last, dtype=du.dtype) / self.t_scale)
        )


ARCHS = {"v1_gru": GRUNetV1, "v4_gru_transformer": GRUNetV4}


def build(arch: str, **cfg):
    if arch not in ARCHS:
        raise ValueError(f"未知骨干：{arch}")
    return ARCHS[arch](**cfg)


def load_state_model(path, map_location="cpu"):
    """加载 state_dict 版模型；返回 (model, cfg)。"""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    arch = obj.get("arch", "v4_gru_transformer")
    cfg = obj.get("cfg", {})
    model = build(arch, **cfg)
    model.load_state_dict(obj["state"])
    model.eval()
    return model, dict(obj, arch=arch, cfg=cfg)
