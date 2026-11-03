# -*- coding: utf-8 -*-
"""训练终端默认模型：v4 GRU-Transformer，在 FC1 + FC2 两个数据集上联合训练。

口径与 notebook 完全一致：
  - 输入特征：rul_pipeline.feature_matrix()（每个数据集各自标准化）；
  - 标签：y = ΔU_1step / σ，σ = 该数据集窗口区间的单步增量标准差；
  - 损失：MSE + LAMBDA_TIME·时间单调正则（与 notebook 相同）；
  - 唯一区别：把两个数据集的窗口**拼起来**训练（相当于更多数据、更接近"面对大量数据"的场景）。

用法：
  python train_terminal_model.py            # 正式训练（20 epoch）
  python train_terminal_model.py --smoke    # 1 epoch 快检
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
import warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.simplefilter("ignore")

import numpy as np
import torch
import torch.nn as nn

import models_terminal as mt
import rul_pipeline as rp

HERE = Path(__file__).parent
MODEL_DIR = HERE / "模型库"
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("FC1 测试集", r"C:\Users\YOGA\Desktop\测试集.csv"),
    ("FC2.v1", r"C:\Users\YOGA\Desktop\电池ai分析\FC2\FC2.v1.csv"),
]

SEQ = 50
EPOCHS = 20
BATCH = 4096
HIDDEN = 64
LR = 2e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SIG_MEAN = 1e-5  # 归一化单位 → µV 的粗略换算基准（仅用于日志显示）
LAMBDA_TIME = 0.30
SEED = 42
VAL_FRAC = 0.08
ARCH = "v4_gru_transformer"
CFG = dict(n_feat=10, hidden=HIDDEN, n_layers=1, t_scale_h=1000.0,
           nhead=4, ff_mult=2, dropout=0.1)


def time_rule_loss(du_pred, tau, n_pairs: int = 2048, gen=None):
    """时间单调正则：更晚的时间窗，|ΔU| 不应更大。"""
    n = len(du_pred)
    if n < 4:
        return torch.zeros((), dtype=du_pred.dtype)
    i = torch.randint(0, n, (n_pairs,), generator=gen)
    j = torch.randint(0, n, (n_pairs,), generator=gen)
    dt = tau[j] - tau[i]
    mask = dt > 0.03 * float(tau.max())
    if mask.sum() == 0:
        return torch.zeros((), dtype=du_pred.dtype)
    a = du_pred[i][mask].abs()
    b = du_pred[j][mask].abs()
    return torch.relu(b - a).mean()


def build_dataset(name: str, path: str):
    blob = open(path, "rb").read()
    prep = rp.prepare(rp.load_csv(blob))
    F, _stats = rp.feature_matrix(prep)
    u = np.asarray(prep["u_s"], float)
    t = np.asarray(prep["t"], float)
    M = len(F) - SEQ
    X = np.lib.stride_tricks.sliding_window_view(F, SEQ, axis=0)[:M]
    X = np.ascontiguousarray(X.transpose(0, 2, 1))          # (M, 10, 50)
    du = np.diff(u)
    sig = float(np.std(du[SEQ - 1:M + SEQ - 1])) or 1.0
    y = (du[SEQ - 1:M + SEQ - 1] / sig).astype(np.float32)
    tau = t[SEQ - 1:M + SEQ - 1].astype(np.float32)
    print(f"  {name}: windows={len(X):,}, σ={sig*1e6:.2f} µV, span={prep['t_end']:.1f} h")
    return dict(name=name, X=X, y=y, tau=tau, sig=sig, n=len(X))


def main(smoke: bool = False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    print("[1] 构造数据集")
    parts = [build_dataset(n, p) for n, p in DATASETS]
    X = np.concatenate([p["X"] for p in parts], axis=0)
    y = np.concatenate([p["y"] for p in parts])
    tau = np.concatenate([p["tau"] for p in parts])
    n = len(X)
    n_val = max(SEQ, int(n * VAL_FRAC))
    n_tr = n - n_val
    print(f"[2] 合并：{n:,} 窗口（训练 {n_tr:,} / 验证 {n_val:,}），特征 {X.shape[1:]}, arch={ARCH}")
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    taut = torch.from_numpy(tau)
    X_tr, y_tr, tau_tr = Xt[:n_tr], yt[:n_tr], taut[:n_tr]
    X_va, y_va = Xt[n_tr:], yt[n_tr:]

    epochs = 1 if smoke else EPOCHS
    model = mt.build(ARCH, **CFG, init_du=float(np.median(np.abs(y[:n_tr]))) or 0.1).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[3] 参数量 {n_param:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=LR * 0.02)
    gen = torch.Generator().manual_seed(SEED)
    log = dict(train=[], val=[], sec=[])
    best = dict(val=float("inf"), epoch=-1, state=None)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_tr, generator=gen)
        tot = 0.0
        nb = 0
        for s in range(0, n_tr, BATCH):
            idx = perm[s:s + BATCH]
            xb, yb, tb = X_tr[idx], y_tr[idx], tau_tr[idx]
            pred = model(xb, tb)
            loss = nn.functional.mse_loss(pred, yb) + LAMBDA_TIME * time_rule_loss(pred, tb, gen=gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        # 验证
        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, n_val, 8192):
                preds.append(model(X_va[s:s + 8192], None if False else taut[n_tr:][s:s + 8192]))
            pv = torch.cat(preds)
            val_mse = float(nn.functional.mse_loss(pv, y_va))
        sec = time.time() - t0
        log["train"].append(tot / max(nb, 1))
        log["val"].append(math.sqrt(val_mse))
        log["sec"].append(sec)
        mark = ""
        if val_mse < best["val"]:
            best = dict(val=val_mse, epoch=ep,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
            mark = "  *best*"
        print(f"  epoch {ep:>2}/{epochs}  loss={tot/max(nb,1):.5f}  val_rmse={math.sqrt(val_mse):.4f} σ "
              f"(≈{math.sqrt(val_mse)*SIG_MEAN*1e6:.1f} µV/步)  {sec:.1f}s{mark}")
        if smoke:
            break

    out = MODEL_DIR / "pemfc_forecast_v4.pt"
    payload = dict(
        arch=ARCH, cfg=CFG, state=best["state"],
        val_rmse=float(math.sqrt(best["val"]) if best["state"] is not None else float("nan")),
        best_epoch=best["epoch"], n_params=n_param, n_windows=int(n),
        datasets=[p["name"] for p in parts],
        sig_by_dataset={p["name"]: p["sig"] for p in parts},
        train_log=log, seed=SEED, seq=SEQ,
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    torch.save(payload, out)
    meta = {k: v for k, v in payload.items() if k not in ("state", "train_log")}
    meta["train_sec"] = float(sum(log["sec"]))
    (MODEL_DIR / "pemfc_forecast_v4_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4] 保存 -> {out}  ({out.stat().st_size/1e6:.2f} MB), best_epoch={best['epoch']}, "
          f"val_rmse={math.sqrt(best['val']):.4f} σ (≈{math.sqrt(best['val'])*SIG_MEAN*1e6:.1f} µV/步)")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
