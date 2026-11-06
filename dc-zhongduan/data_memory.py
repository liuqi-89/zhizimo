from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

MEMORY_DIR = Path(__file__).resolve().parent / "数据记忆库"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

_META = "meta.json"
_D = "d.npy"
_TREND = "trend.npy"

def _slug(text: str, maxlen: int = 28) -> str:
    keep = [c for c in text.strip() if c.isalnum() or c in "-_"]
    return ("".join(keep)[:maxlen] or "dataset")

def _fingerprint(n_rows: int, t_end: float) -> str:
    raw = f"{n_rows:.0f}|{t_end:.4f}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

def fingerprint(n_rows: int, t_end: float) -> str:
    return _fingerprint(int(n_rows), float(t_end))

def load_meta(ds_id: str) -> Optional[dict]:
    return _read_meta(MEMORY_DIR / ds_id) if ds_id else None

def _read_meta(p: Path) -> Optional[dict]:
    try:
        return json.loads((p / _META).read_text(encoding="utf-8"))
    except Exception:
        return None

def save_dataset(name: str, prep: dict, g: dict, cal: dict, extra: Optional[dict] = None) -> str:
    fp = _fingerprint(int(prep["n_rows"]), float(prep["t_end"]))

    for p in MEMORY_DIR.iterdir():
        if p.is_dir() and (_read_meta(p) or {}).get("fingerprint") == fp:
            if p.name == "README.txt":
                continue
            shutil.rmtree(p, ignore_errors=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_id = f"{stamp}_{_slug(name)}"
    d = Path(MEMORY_DIR) / ds_id
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / _D, np.asarray(cal["d"], float).astype(np.float32))

    try:
        t_arr = np.asarray(prep["t"], float)
        u_arr = np.asarray(prep["u_trend"], float)
        step = max(1, len(t_arr) // 400)
        np.save(d / _TREND, np.vstack([t_arr[::step], u_arr[::step]]).astype(np.float32))
    except Exception:
        pass
    meta = dict(
        id=ds_id,
        name=name,
        fingerprint=fp,
        created_at=datetime.now().isoformat(timespec="seconds"),
        n_rows=int(prep["n_rows"]),
        t_end=float(prep["t_end"]),
        u_anchor=float(prep["u_anchor"]),
        u_start=float(prep["u_start"]),
        P_init=float(prep["P_init"]),
        I_ref=float(prep["I_ref"]),
        has_I=bool(prep["has_I"]),
        g=dict(a=float(g["a"]), b=float(g["b"]), c=float(g["c"]),
               rmse=float(g["rmse"]), t0=float(g["t0"])),
        g0_mVh=float(g["g0_mVh"]),
        n_win=int(cal["n_win"]),
        d_mean=float(cal["d_mean"]),
        d_std=float(cal["d_std"]),
        d_p5=float(cal["d_p5"]),
        d_p95=float(cal["d_p95"]),
        quality=(extra or {}).get("quality", ""),
    )
    (d / _META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return ds_id

def list_datasets() -> list[dict]:
    out = []
    try:
        for p in MEMORY_DIR.iterdir():
            if not p.is_dir():
                continue
            m = _read_meta(p)
            if m:
                out.append(m)
    except Exception:
        pass
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out

def load_d(ds_id: str) -> Optional[np.ndarray]:
    try:
        arr = np.load(MEMORY_DIR / ds_id / _D)
        return np.asarray(arr, float)
    except Exception:
        return None

def load_trend(ds_id: str) -> Optional[np.ndarray]:
    try:
        arr = np.load(MEMORY_DIR / ds_id / _TREND)
        return np.asarray(arr, float)
    except Exception:
        return None

def delete_dataset(ds_id: str) -> bool:
    if not ds_id or "/" in ds_id or "\\" in ds_id or ".." in ds_id:
        return False
    p = MEMORY_DIR / ds_id
    if not p.is_dir():
        return False
    shutil.rmtree(p, ignore_errors=True)
    return True

def pooled_profile(current_cal: dict, ds_ids: list[str], taus: np.ndarray,
                   include_current: bool = True, exclude_fp: Optional[str] = None) -> tuple[np.ndarray, int]:
    mats = []
    if include_current:
        mats.append(np.asarray(current_cal["d"], float) / max(np.median(current_cal["d"]), 1e-6))
    for ds in ds_ids:
        meta = load_meta(ds) or {}
        if exclude_fp and meta.get("fingerprint") == exclude_fp:
            continue
        d = load_d(ds)
        if d is None or len(d) < 10:
            continue
        mats.append(d / max(float(np.median(d)), 1e-6))
    if not mats:
        return np.asarray(current_cal["prof"], float), int(include_current)

    rng = np.random.default_rng(2024)
    target = 4000
    pooled = []
    for m in mats:
        m = np.asarray(m, float)
        idx = rng.integers(0, len(m), size=target)
        pooled.append(m[idx])
    pool = np.concatenate(pooled)
    prof = np.quantile(pool, taus)
    prof = prof / max(prof[np.argmin(np.abs(taus - 0.5))], 1e-6)
    return prof, len(mats)

def trend_prior(ds_ids: list[str]) -> Optional[dict]:
    rows = []
    for ds in ds_ids:
        for m in list_datasets():
            if m.get("id") == ds:
                rows.append(m)
    if not rows:
        return None
    a = float(np.median([r["g"]["a"] for r in rows]))
    b = float(np.median([r["g"]["b"] for r in rows]))
    c = float(np.median([r["g"]["c"] for r in rows]))
    return dict(a=a, b=b, c=c, n=len(rows))

def quality_check(prep: dict) -> list[str]:
    msgs = []
    if prep["n_rows"] < 2000:
        msgs.append(f"数据点偏少（{prep['n_rows']} 行），概率层可能不稳定")
    t = np.asarray(prep["t"], float)
    if len(t) > 10:
        gaps = np.diff(t)
        if float(gaps.max()) > 1.0:
            msgs.append(f"存在 >1 h 的数据空档（最大 {gaps.max():.1f} h），已线性插值")
    u = np.asarray(prep["u_s"], float)
    if float(u.min()) < 2.0 or float(u.max()) > 4.2:
        msgs.append(f"电压范围异常（{u.min():.3f}–{u.max():.3f} V），请核对单位")
    drop_pct = 100.0 * (1.0 - prep["u_anchor"] / max(prep["u_start"], 1e-9))
    if drop_pct < 0:
        msgs.append("末端电压高于起始电压（数据可能未按时间排序或整体在恢复）")
    elif drop_pct > 12.0:
        msgs.append(f"末端已衰减 {drop_pct:.1f}%，多数固定阈值已失效，建议调高 FT")
    return msgs
