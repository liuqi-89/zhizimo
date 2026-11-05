# -*- coding: utf-8 -*-
"""神经网络库：把每次分析过的**完整规范序列**打包存档，供神经网络继续学习成长。

与 ``数据记忆库/``（data_memory.py）的分工
------------------------------------------
* 数据记忆库：只存「乘子样本 d + 趋势快照 + 元信息」，用于概率层的历史数据增强；
* 神经网络库（本模块）：存**完整数据**（时间网格、平滑电压、趋势电压、电流/温度/湿度、
  以及 RUL 阈值口径需要的 P_init / I_ref 等），供以后的训练脚本直接读取、继续喂模型。
  目录：``神经网络库/<时间戳_slug>/series.npz + meta.json``，根目录还有 ``index.json``
  与 ``README.txt``。

去重原则（用户要求：一份数据只能看一次，不能加深模型偏见）
----------------------------------------------------------
打包前先算**内容指纹** ``content_key``（时间网格 + 平滑电压序列规范化后的 sha256）；
库里已有相同指纹 → 直接跳过、返回 ``status="duplicate"``，绝不重复入库。
因此同一份原始数据无论分析多少次，神经网络库里只会有一份。

用法
----
>>> import nn_library as nl
>>> nl.archive("FC2.v1", prep)          # -> {"status": "added"/"duplicate", "id": ..., "key": ...}
>>> nl.stats()                          # -> {"n": 3, "mb": 4.2, "keys": 3}
>>> nl.list_packs()                     # 已入库数据（按时间倒序）
>>> nl.iter_packs()                     # 训练用：逐个 yield {"meta": ..., "t": ..., "u_s": ...}

所有操作都容错：库损坏/缺文件时返回空，不影响终端的预测主流程。
"""
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
# 存进 npz 的字段（都是 prepare() 里的规范序列）
_FIELDS = ("t", "u_s", "u_trend", "I_s", "T_s", "RH_s")


# ---------------------------------------------------------------- 基础工具
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


# ---------------------------------------------------------------- 内容指纹
def content_key(prep: dict) -> str:
    """数据内容指纹：时间网格 + 平滑电压序列（四舍五入到 1e-6）的 sha256 前 16 位。

    只认同「实实在在的数据本身」：文件名、上传时间、备注都不参与，
    所以同一份原始数据换个文件名再传一次，依然会被判为重复。
    """
    t = np.round(np.asarray(prep["t"], dtype=float), 6)
    u = np.round(np.asarray(prep["u_s"], dtype=float), 6)
    h = hashlib.sha256()
    h.update(str(len(t)).encode("ascii"))
    h.update(t.astype("<f8").tobytes())
    h.update(u.astype("<f8").tobytes())
    return h.hexdigest()[:16]


def find_duplicate(key: str, idx: Optional[dict] = None) -> Optional[dict]:
    """库里是否已有相同指纹的数据；返回该条 meta 或 None。"""
    idx = idx if idx is not None else _read_index()
    rec = idx.get("packs", {}).get(key)
    if not isinstance(rec, dict):
        return None
    # index 与磁盘不一致时（手动删过目录）视为不存在
    if not (NN_DIR / str(rec.get("id", "")) / _SERIES).exists():
        return None
    return rec


# ---------------------------------------------------------------- 写/读
def archive(name: str, prep: dict, extra: Optional[dict] = None) -> dict:
    """把一份完整数据打包进神经网络库（相同指纹只入一次）。

    返回 ``{"status": "added"/"duplicate", "id": ..., "key": ..., "meta": {...}}``。
    失败（磁盘满/权限）时返回 ``status="error"``，不抛异常、不影响预测。
    """
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
    except Exception as ex:  # noqa: BLE001
        return dict(status="error", id=None, key="", error=f"{type(ex).__name__}: {ex}")


def list_packs() -> list[dict]:
    """列出神经网络库里所有数据（按入库时间倒序）。"""
    idx = _read_index()
    out = [m for m in idx.get("packs", {}).values() if isinstance(m, dict)]
    out.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
    return out


def load_pack(pack_id: str) -> Optional[dict]:
    """读取一份数据的完整序列（返回 dict，含 t/u_s/u_trend/I_s/T_s/RH_s/meta）。

    训练脚本直接用它拼样本；索引损坏时退回按目录读取。
    """
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
    """训练用迭代器：按入库时间顺序逐份 yield 完整数据。"""
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
    """库概况：份数、占用体积、去重前重复尝试次数（本次进程内不做持久化）。"""
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
