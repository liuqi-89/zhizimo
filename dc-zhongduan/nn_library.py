from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

NN_DIR = Path(__file__).resolve().parent / "神经网络库"
NN_DIR.mkdir(parents=True, exist_ok=True)

_INDEX = NN_DIR / "index.json"
_SERIES = "series.npz"
_META = "meta.json"

_FIELDS = ("t", "u_s", "u_trend", "I_s", "T_s", "RH_s")

def _slug(text: str, maxlen: int = 28) -> str:
    keep = [c for c in str(text).strip() if c.isalnum() or c in "-_"]
    return ("".join(keep)[:maxlen] or "dataset")

def _read_index() -> dict:
    try:
        idx = json.loads(_INDEX.read_text(encoding="utf-8"))
        if isinstance(idx, dict) and isinstance(idx.get("packs"), dict):
            return idx
    except Exception:
        pass
    return {"version": 1, "packs": {}}

def _write_index(idx: dict) -> None:
    try:
        _INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _fmt_bytes(n: float) -> str:
    if n >= 1048576:
        return f"{n / 1048576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n:.0f} B"

def content_key(prep: dict) -> str:
    t = np.round(np.asarray(prep["t"], dtype=float), 6)
    u = np.round(np.asarray(prep["u_s"], dtype=float), 6)
    h = hashlib.sha256()
    h.update(str(len(t)).encode("ascii"))
    h.update(t.astype("<f8").tobytes())
    h.update(u.astype("<f8").tobytes())
    return h.hexdigest()[:16]

def find_duplicate(key: str, idx: Optional[dict] = None) -> Optional[dict]:
    idx = idx if idx is not None else _read_index()
    rec = idx.get("packs", {}).get(key)
    if not isinstance(rec, dict):
        return None

    if not (NN_DIR / str(rec.get("id", "")) / _SERIES).exists():
        return None
    return rec

def archive(name: str, prep: dict, extra: Optional[dict] = None) -> dict:
    try:
        key = content_key(prep)
        idx = _read_index()
        dup = find_duplicate(key, idx)
        if dup is not None:
            return dict(status="duplicate", id=dup.get("id"), key=key, meta=dup)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pack_id = f"{stamp}_{_slug(name)}"
        d = NN_DIR / pack_id
        d.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(prep[k], dtype=np.float32) for k in _FIELDS if k in prep}
        np.savez_compressed(d / _SERIES, **arrays)
        meta = dict(
            id=pack_id,
            name=str(name),
            key=key,
            created_at=datetime.now().isoformat(timespec="seconds"),
            n_rows=int(prep.get("n_rows", 0)),
            n_grid=int(len(np.asarray(prep["t"]))),
            t_end=float(prep.get("t_end", np.nan)),
            u_anchor=float(prep.get("u_anchor", np.nan)),
            u_start=float(prep.get("u_start", np.nan)),
            P_init=float(prep.get("P_init", np.nan)),
            I_ref=float(prep.get("I_ref", np.nan)),
            has_I=bool(prep.get("has_I", False)),
            quality=str((extra or {}).get("quality", "")),
            source=str((extra or {}).get("source", "")),
        )
        (d / _META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        idx.setdefault("packs", {})[key] = meta
        _write_index(idx)
        _ensure_readme()
        return dict(status="added", id=pack_id, key=key, meta=meta)
    except Exception as ex:
        return dict(status="error", id=None, key="", error=f"{type(ex).__name__}: {ex}")

def list_packs() -> list[dict]:
    idx = _read_index()
    out = [m for m in idx.get("packs", {}).values() if isinstance(m, dict)]
    out.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
    return out

def load_pack(pack_id: str) -> Optional[dict]:
    if not pack_id or "/" in pack_id or "\\" in pack_id or ".." in pack_id:
        return None
    d = NN_DIR / pack_id
    try:
        with np.load(d / _SERIES) as z:
            out = {k: np.asarray(z[k], dtype=float) for k in z.files}
        out["meta"] = json.loads((d / _META).read_text(encoding="utf-8"))
        return out
    except Exception:
        return None

def iter_packs() -> Iterator[dict]:
    for meta in sorted(list_packs(), key=lambda m: str(m.get("created_at", ""))):
        pack = load_pack(str(meta.get("id")))
        if pack:
            yield pack

def delete_pack(pack_id: str) -> bool:
    if not pack_id or "/" in pack_id or "\\" in pack_id or ".." in pack_id:
        return False
    p = NN_DIR / pack_id
    if not p.is_dir():
        return False
    idx = _read_index()
    packs = idx.setdefault("packs", {})
    for k, m in list(packs.items()):
        if isinstance(m, dict) and m.get("id") == pack_id:
            packs.pop(k, None)
    _write_index(idx)
    shutil.rmtree(p, ignore_errors=True)
    return True

def stats() -> dict:
    packs = list_packs()
    size = 0
    for m in packs:
        d = NN_DIR / str(m.get("id", ""))
        try:
            size += sum(f.stat().st_size for f in d.iterdir() if f.is_file())
        except Exception:
            pass
    return dict(n=len(packs), mb=size / 1048576.0, size_text=_fmt_bytes(size), keys=len(packs),
                dir=str(NN_DIR))

def _ensure_readme() -> None:
    p = NN_DIR / "README.txt"
    if p.exists():
        return
    try:
        p.write_text(
            "神经网络库（供模型继续训练的数据仓库）\n"
            "=====================================\n"
            "每份数据一个文件夹：<时间戳_slug>/series.npz（t/u_s/u_trend/I_s/T_s/RH_s）\n"
            "                            + meta.json（元信息 + 内容指纹 key）\n"
            "index.json 是索引；相同内容指纹的数据只会入库一次（一份数据只看一次）。\n"
            "读取：import nn_library as nl; nl.list_packs() / nl.load_pack(id) / nl.iter_packs()\n",
            encoding="utf-8")
    except Exception:
        pass
