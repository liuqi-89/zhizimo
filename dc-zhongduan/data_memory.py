from __future__ import annotations

import hashlib
import json
import math
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
                   include_current: bool = True, exclude_fp: Optional[str] = None,
                   current_i_ref: Optional[float] = None) -> tuple[np.ndarray, int]:
    i_cur = None
    if current_i_ref is not None:
        try:
            _v = float(current_i_ref)
            if np.isfinite(_v) and _v > 0:
                i_cur = _v
        except (TypeError, ValueError):
            i_cur = None

    mats: list[np.ndarray] = []
    weights: list[float] = []
    if include_current:
        mats.append(np.asarray(current_cal["d"], float) / max(np.median(current_cal["d"]), 1e-6))
        weights.append(1.0)
    for ds in ds_ids:
        meta = load_meta(ds) or {}
        if exclude_fp and meta.get("fingerprint") == exclude_fp:
            continue
        try:
            _p5 = abs(float(meta.get("d_p5") or 0.0))
            _p95 = abs(float(meta.get("d_p95") or 0.0))
            _dmean = abs(float(meta.get("d_mean") or 0.0))
        except (TypeError, ValueError):
            _p5 = _p95 = _dmean = 0.0
        if _p5 > 20.0 or _p95 > 20.0 or _dmean > 8.0:
            continue

        _w = 1.0
        if i_cur is not None:
            try:
                _i_d = float(meta.get("I_ref") or 0.0)
            except (TypeError, ValueError):
                _i_d = 0.0
            if _i_d > 0:
                _lr = math.log(_i_d / i_cur)
                _w = float(math.exp(-(_lr / 0.7) ** 2))
                if _w < 0.15:
                    continue
            else:
                _w = 0.4
        d = load_d(ds)
        if d is None or len(d) < 10:
            continue
        mats.append(d / max(float(np.median(d)), 1e-6))
        weights.append(_w)
    if not mats:
        return np.asarray(current_cal["prof"], float), int(include_current)

    rng = np.random.default_rng(2024)
    total = 4000 * len(mats)
    wsum = float(np.sum(weights)) or 1.0
    pooled = []
    for m, w in zip(mats, weights):
        m = np.asarray(m, float)
        if m.size == 0:
            continue
        n_i = max(50, int(round(total * float(w) / wsum)))
        idx = rng.integers(0, len(m), size=n_i)
        pooled.append(m[idx])
    if not pooled:
        return np.asarray(current_cal["prof"], float), int(include_current)
    pool = np.concatenate(pooled)
    prof = np.quantile(pool, taus)
    prof = prof / max(prof[np.argmin(np.abs(taus - 0.5))], 1e-6)

    prof = np.clip(prof, 0.1, 5.0)
    return prof, len(mats)

def _family_key(name: str) -> str:
    s = str(name or "").strip()
    for sep in ("·", "|", "-", "_"):
        if sep in s:
            s = s.split(sep)[0]
            break
    return s.strip()[:24]

def condition_sensitivity() -> dict:
    fams: dict[str, list[tuple[float, float]]] = {}
    for r in list_datasets():
        try:
            i_ref = float(r.get("I_ref") or 0.0)
            g0 = float(r.get("g0_mVh") or 0.0)
        except (TypeError, ValueError):
            continue
        if not bool(r.get("has_I")) or i_ref <= 1.0 or g0 <= 1e-9 or g0 >= 1e3:
            continue
        fams.setdefault(_family_key(r.get("name")), []).append((i_ref, g0))

    best: tuple[str, list[tuple[float, float]], float] | None = None
    for fam, pts in fams.items():
        if len(pts) < 3:
            continue
        _i = [p[0] for p in pts]
        spread = max(_i) / max(min(_i), 1e-9)
        if best is None or (len(pts), spread) > (len(best[1]), best[2]):
            best = (fam, pts, spread)
    _fam_counts = {k: len(v) for k, v in fams.items()}
    if best is None:
        return dict(ok=False, n=0, family=None, families=_fam_counts)

    fam, pts, _spread = best
    x = np.log(np.asarray([p[0] for p in pts], float))
    y = np.log(np.asarray([p[1] for p in pts], float))
    try:
        alpha, b = np.polyfit(x, y, 1)
        pred = alpha * x + b
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    except Exception:
        return dict(ok=False, n=len(pts), family=fam, pts=pts, families=_fam_counts)
    return dict(ok=True, n=len(pts), alpha=float(alpha), r2=float(r2), family=fam,
                i_min=min(p[0] for p in pts), i_max=max(p[0] for p in pts), pts=pts,
                families=_fam_counts)

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
    u_med = float(np.nanmedian(u)) if np.isfinite(u).any() else float("nan")
    lo, hi = (0.1, 1.5) if (np.isfinite(u_med) and u_med < 2.0) else (2.0, 4.2)
    if float(np.nanmin(u)) < lo or float(np.nanmax(u)) > hi:
        msgs.append(f"电压范围异常（{np.nanmin(u):.3f}-{np.nanmax(u):.3f} V），"
                    f"请核对单位（单片判定阈值 0.1-1.5 V、电堆 2.0-4.2 V）")
    drop_pct = 100.0 * (1.0 - prep["u_anchor"] / max(prep["u_start"], 1e-9))
    if drop_pct < 0:
        msgs.append("末端电压高于起始电压（数据可能未按时间排序或整体在恢复）")
    elif drop_pct > 12.0:
        msgs.append(f"末端已衰减 {drop_pct:.1f}%，多数固定阈值已失效，建议调高 FT")
    return msgs
