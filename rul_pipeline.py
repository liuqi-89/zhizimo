# -*- coding: utf-8 -*-
"""
rul_pipeline.py —— Streamlit 终端的数值核心（与 v1 notebook 完全同口径）

链路（v1 的真实设计）：
    CSV → 重采样 0.01h → 10h 居中平滑(u_s) / 150h 居中趋势(u_trend)
        → 50h 分段衰减速率 → 指数衰减律 g(t)=a·exp(-b·t)+c 最小二乘拟合
        → 10h 窗口乘子 d=y/x 的分位数回归(19 个 τ) → 保序 + 锚定 + 中位归一化
        → AR(1) 分位水平抽样 → 4000 条蒙特卡洛电压路径 → 各失效阈值 FT 的 RUL 分布
    GRU 模型（电池v1）：加载后做一步预测自检，用于显示"AI 侧"的噪声地平面。

对外主要接口：
    load_csv(blob)              -> DataFrame
    prepare(df)                 -> dict（重采样/平滑/阈值/数据摘要）
    fit_decay(prep)             -> dict（g 参数、RMSE）
    calibrate(prep, g)          -> dict（分位数乘子 profile）
    run_monte_carlo(prep, g, prof) -> dict（RUL 矩阵、分位数、扇形带、确定性曲线）
    analyze(blob, model)        -> dict（上面全部 + AI 一步自检）
"""
from __future__ import annotations

import io
import math
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 与 notebook 一致的常量
# ----------------------------------------------------------------------------
DT = 0.01                 # 均匀重采样步长（h）
SMOOTH_WIN_STEPS = 1001   # ≈10 h 居中平滑
TREND_WIN_STEPS = 15001   # ≈150 h 居中趋势
SEG_H = 50.0              # 分段长度（h）
SEG_STEPS = 5000          # 50 h / 0.01 h
N_EDGE = 10               # 分段速率首尾各取 10 点
P_INIT_H = 10.0           # P_init 用前 10 h

WIN_H = 10.0              # 分位标定/蒙特卡洛步长（h）
STRIDE_H = 5.0            # 标定窗口步长（h）
TAUS = np.round(np.arange(0.05, 0.9501, 0.05), 2)
RHO_AR1 = 0.35
Q_CLIP = (0.03, 0.97)
N_MC = 4000
HORIZON_H = 3000.0
# 大数据保护：原始表超过这么多行时等间隔抽稀（0.01h 网格下 60 万行≈ 6000h）
MAX_ROWS = 600_000        # 外推视界（h）；覆盖到 FT=6% 也不会删失
SEED_MC = 2024
RUL_FT = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]   # 6.0% 仅用于终端滑条（UI 范围 3.5~6.0）
CROSS_WIN = WIN_H

# 模型相关（必须与 v1 训练时一致）
MODEL_CFG = dict(n_feat=10, hidden=64, n_layers=1, t_scale_h=1000.0,
                 time_gain=3.0, train_t_ref=None)

# 演示数据候选路径：优先用随代码一起分发的 数据/ 目录（云端可用），
# 其次回退到本机桌面的原始数据集（本地开发用）。两者都没有时该演示项自动隐藏。
_HERE = pathlib.Path(__file__).resolve().parent
DEMO_DIR = _HERE / "数据"
_DESKTOP = r"C:\Users\YOGA\Desktop"
DEMO_SETS = [
    ("FC1 · 测试集.csv（0 → 912.85 h，可对比历史结果）",
     [DEMO_DIR / "demo_fc1.csv", _DESKTOP + r"\测试集.csv"]),
    ("FC2 · FC2.v1.csv（0 → 1020.54 h，衰减更强）",
     [DEMO_DIR / "demo_fc2.csv", _DESKTOP + r"\电池ai分析\FC2\FC2.v1.csv"]),
    ("FC1 · 完整记录 FC1.csv（0 → 1154.21 h）",
     [DEMO_DIR / "demo_fc1_full.csv",
      _DESKTOP + r"\电池ai分析\IEEE2014DataChallengeData\IEEE 2014 Data Challenge Data\FC1_Without_Ripples\FC1.csv"]),
]


def _first_existing(paths):
    for p in paths:
        try:
            if pathlib.Path(p).is_file():
                return str(p)
        except OSError:
            continue
    return None


def _resolve_demos():
    avail = [(name, _first_existing(paths)) for name, paths in DEMO_SETS]
    return ([(n, p) for n, p in avail if p], [n for n, p in avail if p])


DEMO_AVAILABLE, DEMO_LABELS = _resolve_demos()
DEMO_CANDIDATES = [p for _, p in DEMO_AVAILABLE]

# 模型文件候选路径
MODEL_CANDIDATES = [
    "电池v1",
    "models/电池v1",
    "电池v1.pt",
    r"C:\Users\YOGA\PycharmProjects\PythonProject\电池v1",
]


# ----------------------------------------------------------------------------
# 1. 读表 + 智能列识别（兼容不同文献 / 仪器导出的 CSV）
# ----------------------------------------------------------------------------
_ENC_CANDIDATES = ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "latin-1")
_SEP_CANDIDATES = (",", ";", "\t", "|")

_TIME_KEYS = ("time", "hour", "hrs", "h)", "(h", "[h", "时间", "时长", "duration", "run time", "elapsed")
_VOLT_KEYS = ("volt", "voltage", "电压", "utot", "u_tot", "u cell", "ucell", "stack", "总压", "(v)", "v)", "mv")
_CURR_KEYS = ("curr", "current", "电流", "amp", "(a)")
_TEMP_KEYS = ("temp", "温度", "°c", "℃", "thermo", "tin", "tout")
_RH_KEYS = ("hum", "humid", "湿度", "rh", "%")


def _norm_key(name) -> str:
    s = str(name).strip().lower()
    for ch in (" ", "_", "-", "\t", "\n", "\r"):
        s = s.replace(ch, "")
    return s


def _decode_text(blob: bytes):
    """按常见编码依次尝试解码；返回 (文本, 采用的编码, 尝试记录)。"""
    tried = []
    for enc in _ENC_CANDIDATES:
        try:
            text = blob.decode(enc)
            if "\n" in text:
                tried.append((enc, "ok"))
                return text, enc, tried
        except UnicodeDecodeError as ex:
            tried.append((enc, f"失败：{ex.reason}"))
    raise ValueError(
        "无法识别文件编码（已尝试 " + "、".join(f"{e}[{s}]" for e, s in tried) +
        "）。请把数据另存为 UTF-8 / GBK 后再试，或直接粘贴为逗号分隔文本。")


def _looks_header(cells) -> bool:
    txt = [c for c in cells if isinstance(c, str) and c.strip() != ""]
    if len(txt) < 2:
        return False
    num = 0
    for c in txt:
        try:
            float(c.replace(",", ""))
            num += 1
        except ValueError:
            pass
    return num <= len(txt) // 2          # 少于一半是纯数字 → 像表头


def _is_excel(blob: bytes) -> bool:
    """xlsx = ZIP 容器（PK\\x03\\x04）；xls = OLE2 复合文档。"""
    return blob[:4] == b"PK\x03\x04" or blob[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _read_excel_any(blob: bytes):
    """Excel 文件 → (DataFrame, info)，自动跳过前导说明行并找表头。"""
    try:
        import openpyxl  # noqa: F401  （xlsx 需要；缺失时给出明确提示）
    except ImportError:
        raise ValueError("这是 Excel 文件，但当前环境缺少 openpyxl："
                         "请在 pytorch_env 里 `pip install openpyxl`，或把表格另存为 CSV 后上传。")
    bio = io.BytesIO(blob)
    try:
        probe = pd.read_excel(bio, sheet_name=0, header=None)
    except Exception as ex:  # noqa: BLE001
        raise ValueError(f"Excel 解析失败：{ex}")
    if probe.shape[0] < 3 or probe.shape[1] < 2:
        raise ValueError("Excel 内容太少（<3 行或 <2 列），无法识别。")
    hdr = 0
    for i in range(min(15, len(probe))):
        cells = [str(v).strip() for v in probe.iloc[i].tolist()]
        cells = [c for c in cells if c and c.lower() != "nan"]
        if cells and _looks_header(cells):
            hdr = i
            break
    bio.seek(0)
    df = pd.read_excel(bio, sheet_name=0, header=hdr)
    df = df.loc[:, [c for c in df.columns if not df[c].isna().all()]]
    info = dict(encoding="excel", encoding_tried=["excel"],
                sep="Excel", header_row=hdr + 1, n_raw_rows=int(len(df)),
                raw_columns=[str(c) for c in df.columns])
    return df.reset_index(drop=True), info


def read_any(blob: bytes):
    """解码 + 探测分隔符/前导说明行 + 返回原始 DataFrame 和解析信息（不猜列名）。"""
    if _is_excel(blob):
        return _read_excel_any(blob)
    text, enc, tried = _decode_text(blob)
    head = [ln for ln in text.split("\n", 40)[:20] if ln.strip()]
    if len(head) < 3:
        raise ValueError("文件内容太少（< 3 行），无法识别。")
    best = None
    for sep in _SEP_CANDIDATES:
        if not any(ln.count(sep) >= 1 for ln in head[:10]):
            continue
        # 找表头行（只扫前 15 行）
        hdr = 0
        for i, ln in enumerate([ln for ln in head[:15] if ln.strip()]):
            if _looks_header([c.strip() for c in ln.split(sep)]):
                hdr = i
                break
        rows = [ln for ln in head if ln.strip()]
        _cells = [c.strip() for c in rows[min(hdr, len(rows) - 1)].split(sep)]
        _numeric = 0
        for c in _cells:
            try:
                float(c.replace(",", ""))
                _numeric += 1
            except ValueError:
                pass
        _has_header = _numeric < max(1, len(_cells) // 2)
        df = None
        for engine in ("c", "python"):
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=hdr,
                                 header=0 if _has_header else None, engine=engine,
                                 on_bad_lines="skip", low_memory=False)
                break
            except Exception:      # noqa: BLE001
                df = None
        if df is None or df.shape[1] < 2 or len(df) < 20:
            continue
        if not _has_header:
            df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
        df = df.rename(columns={c: f"列{i + 1}" for i, c in enumerate(df.columns)
                                if str(c).startswith("Unnamed")})
        df = df.loc[:, [c for c in df.columns if not df[c].isna().all()]]
        n_num = sum(pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.5
                    for c in df.columns)
        if n_num >= 2:
            best = (df, sep, hdr)
            break                       # 找到可用解析即停止，避免大文件重复读
    if best is None:
        raise ValueError("无法解析成表格：请确认文件是逗号/分号/制表符分隔的文本（CSV/TSV）。")
    df, sep, hdr = best
    info = dict(encoding=enc, encoding_tried=tried,
                sep={",": "逗号", ";": "分号", "\t": "制表符", "|": "竖线"}.get(sep, sep),
                header_row=hdr + 1, n_raw_rows=int(len(df)),
                raw_columns=[str(c) for c in df.columns])
    return df.reset_index(drop=True), info


def auto_map_columns(df: pd.DataFrame) -> dict:
    """用「列名关键词 + 数值形状」双重打分自动识别时间/电压/电流/温度/湿度列。"""
    series, keys = {}, {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.6:
            series[str(c)] = s.to_numpy(float)
            keys[str(c)] = _norm_key(c)
    if len(series) < 2:
        raise ValueError("表格中可用的数值列不足 2 列，无法识别时间/电压。")

    # ---- 时间列：单调递增 + 跨度大 + 名字关键词 ----
    t_scores = []
    for c, v in series.items():
        ok = np.isfinite(v)
        if ok.sum() < 20:
            continue
        vv = v[ok]
        mono = float(np.mean(np.diff(vv) >= 0)) if len(vv) > 2 else 0.0
        span = float(np.nanmax(vv) - np.nanmin(vv))
        med = float(np.nanmedian(vv))
        score = 2.0 * mono + (1.5 if span >= 10 else 0.0) + (1.5 if med >= 0 else -2.0)
        score += 2.5 if any(k in keys[c] for k in _TIME_KEYS) else 0.0
        t_scores.append((score, c, mono, span, med))
    t_scores.sort(reverse=True)
    t_col = t_scores[0][1]
    t_span = t_scores[0][3]
    if t_scores[0][2] < 0.9 or t_span <= 0:
        raise ValueError("没找到单调递增的时间列。可在侧栏「手动指定列」里选时间列。")

    # ---- 电压列：名字关键词 + 数值落在 0.2~10 V（或 mV 量级） + 随时间去趋势下降 ----
    t_raw = series[t_col]
    ok_t = np.isfinite(t_raw)
    u_scores = []
    for c, v in series.items():
        if c == t_col:
            continue
        ok = np.isfinite(v)
        if ok.sum() < 20:
            continue
        med = float(np.nanmedian(v[ok]))
        v_use = v * (1e-3 if med > 50 else 1.0)          # mV → V 自适应
        med_v = float(np.nanmedian(v_use[ok]))
        in_range = 0.05 <= med_v <= 20
        # 与时间线性相关的斜率符号：下降为正分
        m = ok & ok_t
        slope_score = 0.0
        if m.sum() > 50:
            slope = np.polyfit(t_raw[m], v_use[m], 1)[0]
            slope_score = 2.0 if slope < 0 else -1.0
        kw = 3.0 if any(k in keys[c] for k in _VOLT_KEYS) else 0.0
        score = kw + (2.0 if in_range else -3.0) + slope_score
        u_scores.append((score, c, med_v, in_range))
    u_scores.sort(reverse=True)
    if not u_scores or u_scores[0][0] <= 0 or not u_scores[0][3]:
        raise ValueError("没找到可用的电压列（数值应在 0.2~10 V 或 200~10000 mV）。"
                         "可在侧栏「手动指定列」里指定电压列。")
    u_col = u_scores[0][1]

    def _pick(keys_tuple):
        cand = [c for c, k in keys.items() if c not in (t_col, u_col)
                and any(kw in k for kw in keys_tuple)]
        return cand[0] if cand else None

    mapping = {
        "Time (h)": t_col,
        "Utot (V)": u_col,
        "I (A)": _pick(_CURR_KEYS),
        "TinAIR (°C)": _pick(_TEMP_KEYS),
        "HrAIRFC (%)": _pick(_RH_KEYS),
    }
    info = dict(
        time=t_col, voltage=u_col, current=mapping["I (A)"],
        temp=mapping["TinAIR (°C)"], rh=mapping["HrAIRFC (%)"],
        auto=True, time_span=float(t_span))
    return mapping, info


def _to_numeric_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


# 时间列单位：s / min / h（内部统一换算成小时）
_TIME_UNIT_DIV = {"s": 3600.0, "min": 60.0, "h": 1.0}
TIME_UNIT_CHOICES = [("自动识别（推荐）", "auto"), ("小时", "h"),
                     ("分钟", "min"), ("秒", "s")]
_TIME_UNIT_NAME = {"s": "秒", "min": "分钟", "h": "小时", "auto": "自动"}


def _guess_time_unit(t: np.ndarray, col_name: str = "") -> str:
    """按列名关键词 + 数值跨度，猜测时间列原始单位（s / min / h）。"""
    key = _norm_key(col_name) if col_name else ""
    if any(k in key for k in ("hour", "hrs", "hr")) or "(h)" in key or key.endswith("h"):
        return "h"
    if any(k in key for k in ("minute", "mins")) or "(min)" in key:
        return "min"
    if any(k in key for k in ("second", "sec", "secs", "timestamp")) or "(s)" in key:
        return "s"
    v = t[np.isfinite(t)]
    if len(v) < 10:
        return "h"
    span = float(v.max() - v.min())
    if span <= 3000.0:                      # 现有老化数据都是小时级别（0~几千 h）
        return "h"
    # 跨度很大：按"换算成小时后是否落在合理老化试验时长"来判定，优先按秒解释
    for unit in ("s", "min"):
        dur_h = span / _TIME_UNIT_DIV[unit]
        if 5.0 <= dur_h <= 15000.0:
            return unit
    return "h"


def _resolve_time_unit(t: np.ndarray, col_name: str, requested: str = "auto") -> str:
    if requested in _TIME_UNIT_DIV:
        return requested
    return _guess_time_unit(t, col_name)


def make_canonical(df_raw: pd.DataFrame, mapping: dict,
                   time_unit: str = "auto") -> pd.DataFrame:
    """按映射抽取 Time/Utot(+I/T/RH) 生成规范列名，做 mV→V 自适应与清洗。

    time_unit: "auto" 自动识别，"h"/"min"/"s" 手动指定时间列原始单位（统一换算成小时）。
    """
    t_col = mapping.get("Time (h)")
    u_col = mapping.get("Utot (V)")
    if not t_col or not u_col:
        raise ValueError("缺少时间列或电压列，无法分析。")
    t = _to_numeric_series(df_raw[t_col])
    unit = _resolve_time_unit(t, str(t_col), time_unit)
    fell_back = False
    if unit != "h":
        t = t / _TIME_UNIT_DIV[unit]
    # 手动单位把时长压得太短 → 退回自动识别重试（真正的短数据后续仍会明确报错）
    if time_unit not in (None, "auto") and unit == time_unit:
        _v = t[np.isfinite(t)]
        if len(_v) > 10 and float(_v.max() - _v.min()) < 50.0:
            _u2 = _guess_time_unit(_to_numeric_series(df_raw[t_col]), str(t_col))
            if _u2 != unit:
                t = _to_numeric_series(df_raw[t_col]) / _TIME_UNIT_DIV[_u2]
                unit, fell_back = _u2, True
    u = _to_numeric_series(df_raw[u_col])
    u_med = float(np.nanmedian(u)) if np.isfinite(u).any() else np.nan
    if np.isfinite(u_med) and u_med > 50:        # mV → V
        u = u / 1000.0
    out = {"Time (h)": t, "Utot (V)": u}
    for canon, scale in (("I (A)", 1.0), ("TinAIR (°C)", 1.0), ("HrAIRFC (%)", 1.0)):
        src = mapping.get(canon)
        if src and src in df_raw.columns:
            v = _to_numeric_series(df_raw[src]) * scale
            if np.isfinite(v).mean() > 0.5:
                out[canon] = v
    df = pd.DataFrame(out)
    df = df.dropna(subset=["Time (h)", "Utot (V)"])
    df = df[np.isfinite(df["Utot (V)"])]
    df = df.sort_values("Time (h)").reset_index(drop=True)
    df = df[df["Time (h)"].diff().fillna(1.0) > 0].reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"有效数据太短（{len(df)} 行，至少 200 行）：检查时间/电压列是否识别正确。")
    # --- 大数据保护：超过 MAX_ROWS 行时按等间隔抽稀（原始 dt 往往远小于 0.01h 网格，
    #     抽稀不损失信息；保留首尾，并在 df.attrs 记录，界面上会提示）---------------
    if len(df) > MAX_ROWS:
        _n0 = len(df)
        keep = np.unique(np.linspace(0, _n0 - 1, MAX_ROWS).astype(np.int64))
        df = df.iloc[keep].reset_index(drop=True)
        df.attrs["downsampled"] = True
        df.attrs["downsampled_from"] = int(_n0)
    df.attrs["mapping"] = mapping
    df.attrs["time_unit"] = unit
    df.attrs["unit_fallback"] = bool(fell_back)
    return df


def merge_canonical(dfs) -> pd.DataFrame:
    """把同一电池的多个分段/多份规范表按时间顺序拼接（时间重新从 0 开始则接续）。"""
    dfs = [d for d in dfs if d is not None and len(d) > 0]
    if not dfs:
        raise ValueError("没有可合并的数据。")
    if len(dfs) == 1:
        return dfs[0]
    dfs = sorted(dfs, key=lambda d: float(d["Time (h)"].iloc[0]))
    pieces, prev_max = [], None
    for d in dfs:
        t = d["Time (h)"].to_numpy(float)
        tt = t.copy()
        if prev_max is not None and tt[0] < prev_max - 1e-9:   # 时间重启 → 接续
            d_med = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.01
            tt = tt + (prev_max + d_med - tt[0])
        dd = d.copy()
        dd["Time (h)"] = tt
        pieces.append(dd)
        prev_max = float(tt.max())
    out = pd.concat(pieces, ignore_index=True).sort_values("Time (h)")
    out = out.drop_duplicates(subset=["Time (h)"], keep="first").reset_index(drop=True)
    out.attrs.update(pieces[0].attrs)
    out.attrs["merged_files"] = len(pieces)
    out.attrs["merged_rows"] = [int(len(p)) for p in pieces]
    return out


def load_csv(blob: bytes, time_unit: str = "auto") -> pd.DataFrame:
    """一步到位：解码 + 解析 + 自动识别列名 → 规范 DataFrame。"""
    df_raw, info = read_any(blob)
    mapping, minfo = auto_map_columns(df_raw)
    df = make_canonical(df_raw, mapping, time_unit=time_unit)
    df.attrs["info"] = info
    df.attrs["map_info"] = minfo
    return df


def load_csv_many(blobs, time_unit: str = "auto") -> pd.DataFrame:
    """多份原始文件 → 自动识别（优先沿用第一份的列映射）→ 按时间接续合并。"""
    blobs = [b for b in blobs if b]
    if not blobs:
        raise ValueError("没有可读取的文件。")
    raws = [read_any(b) for b in blobs]
    map0, minfo0 = auto_map_columns(raws[0][0])
    dfs = []
    for df_raw, _info in raws:
        mapping = map0
        if not all((v is None or v in df_raw.columns) for v in map0.values()):
            mapping = auto_map_columns(df_raw)[0]
        dfs.append(make_canonical(df_raw, mapping, time_unit=time_unit))
    out = merge_canonical(dfs)
    out.attrs["info"] = raws[0][1]
    out.attrs["map_info"] = minfo0
    return out


def guess_demo_blob() -> Optional[bytes]:
    for p in DEMO_CANDIDATES:
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            continue
    return None


# ----------------------------------------------------------------------------
# 2. 重采样 / 平滑 / 阈值
# ----------------------------------------------------------------------------
def _rolling_mean(a: np.ndarray, win: int, min_periods: int) -> np.ndarray:
    return (
        pd.Series(a)
        .rolling(win, center=True, min_periods=min_periods)
        .mean()
        .bfill()
        .ffill()
        .to_numpy(float)
    )


def _col(df: pd.DataFrame, name: str, default: float) -> np.ndarray:
    return df[name].to_numpy(float) if name in df.columns else np.full(len(df), default, float)


def prepare(df: pd.DataFrame) -> dict:
    """重采样到 0.01 h 网格 + 10h/150h 两级平滑 + RUL 阈值。"""
    t_raw = df["Time (h)"].to_numpy(float)
    u_raw = df["Utot (V)"].to_numpy(float)
    t_raw = t_raw - t_raw[0]
    grid = np.arange(0.0, t_raw[-1] + 1e-9, DT)
    if len(grid) < 5000:
        raise ValueError("数据时间跨度不足（< 50 h），无法做衰减律拟合。")
    u_raw_i = np.interp(grid, t_raw, u_raw)
    I_i = np.interp(grid, t_raw, _col(df, "I (A)", np.nan) if "I (A)" in df.columns else np.full(len(df), np.nan))
    has_I = "I (A)" in df.columns and np.isfinite(I_i).all() and np.nanstd(I_i) >= 0 and np.nanmean(I_i) > 1.0
    if not has_I:
        I_i = np.full_like(grid, np.nan)

    u_s = _rolling_mean(u_raw_i, SMOOTH_WIN_STEPS, 1)
    u_trend = _rolling_mean(u_s, TREND_WIN_STEPS, max(1, TREND_WIN_STEPS // 3))
    T_s = _rolling_mean(np.interp(grid, t_raw, _col(df, "TinAIR (°C)", np.nan)) if "TinAIR (°C)" in df.columns else np.full(len(grid), np.nan), SMOOTH_WIN_STEPS, 1)
    RH_s = _rolling_mean(np.interp(grid, t_raw, _col(df, "HrAIRFC (%)", np.nan)) if "HrAIRFC (%)" in df.columns else np.full(len(grid), np.nan), SMOOTH_WIN_STEPS, 1)

    n_init = max(1, int(round(P_INIT_H / DT)))
    n_last = max(1, min(1000, len(grid)))
    u0_block = u_s[:n_init]
    if has_I:
        I_ref = float(np.nanmedian(I_i[-n_last:]))
        P_init = float(np.nanmean(u0_block * I_i[:n_init]))
        uths = [P_init * (1.0 - f / 100.0) / I_ref for f in RUL_FT]
    else:
        u0 = float(np.nanmean(u0_block))
        uths = [u0 * (1.0 - f / 100.0) for f in RUL_FT]
    return dict(
        t=grid, t_end=float(grid[-1]), u_s=u_s, u_trend=u_trend,
        I_s=I_i, T_s=T_s, RH_s=RH_s, has_I=has_I,
        u_anchor=float(u_s[-1]), P_init=P_init if has_I else float(np.nanmean(u0_block)),
        I_ref=I_ref if has_I else float("nan"),
        uths=uths, n_rows=len(df), t_span=float(t_raw[-1]), u_start=float(u_s[0]),
    )


# ----------------------------------------------------------------------------
# 3. 指数衰减律 g(t) 拟合（50h 分段速率）
# ----------------------------------------------------------------------------
def segment_rates(prep: dict) -> pd.DataFrame:
    u, t = prep["u_trend"], prep["t"]
    n_seg = len(u) // SEG_STEPS
    rows = []
    for k in range(n_seg):
        i0, i1 = k * SEG_STEPS, (k + 1) * SEG_STEPS - 1
        drop = float(u[i0:i0 + N_EDGE].mean() - u[i1 - N_EDGE + 1:i1 + 1].mean())
        dur = SEG_STEPS * DT
        rows.append((k, t[i0], t[i1], 0.5 * (t[i0] + t[i1]), drop, drop / dur))
    return pd.DataFrame(rows, columns=["k", "t0", "t1", "t_mid", "drop", "rate"])


def _fit_exp_c0(tt, rr):
    """纯指数 c=0 的稳健最小二乘（多初值）。返回 (a, b) 或 None。"""
    from scipy.optimize import least_squares
    best = None
    for x0 in [(2e-4, 1e-3), (1e-3, 5e-4), (5e-5, 3e-3), (2e-3, 2e-4)]:
        try:
            r = least_squares(lambda p: p[0] * np.exp(-p[1] * tt) - rr, x0,
                              bounds=([1e-9, 1e-8], [1.0, 5e-2]),
                              xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    return float(best.x[0]), float(best.x[1])


DEFAULT_DECAY_B = 1.2e-3


def _prior_decay_b(default: float = DEFAULT_DECAY_B) -> float:
    """历史记忆库里 b 的中位数（无历史时用默认 1.2e-3 /h），用于 b 不可辨识时“保形”。"""
    try:
        import data_memory as dm
        ids = [d["id"] for d in dm.list_datasets()]
        pr = dm.trend_prior(ids)
        if pr and pr.get("b"):
            b = float(pr["b"])
            if 1e-6 < b < 5e-2:
                return b
    except Exception:
        pass
    return float(default)


def _fit_a_given_b(tt, rr, b: float, c: float = 0.0) -> float:
    """固定 b、c 后对 a 做闭式最小二乘。"""
    e = np.exp(-b * np.asarray(tt, float))
    num = float(np.sum((np.asarray(rr, float) - c) * e))
    den = float(np.sum(e * e))
    return max(num / max(den, 1e-18), 1e-9)


def fit_decay(prep: dict) -> dict:
    """单调递减指数律 g(τ)=a·e^{-bτ}+c（a,b>0, c≥0）最小二乘 + 指数形状保卫。

    形状保卫规则（用户要求：外推曲线“不能忘记这是一条指数型函数”）：
      1) 先做自由拟合（c≥0）；
      2) 若末端斜率由常数项 c 主导（指数项占比 < 30%），或 b·t0>4（指数项已死），
         且 c=0 纯指数拟合的 RMSE 不超过自由拟合的 1.25 倍 → 改用 c=0 纯指数；
      3) 若 b 触边界或 b·t0 < 0.2（数据不足以辨识衰减速率，例如只有后半段数据），
         b 改用历史记忆库先验（中位数，缺省 1.2e-3 /h），再闭式重估 a。
    """
    from scipy.optimize import least_squares  # 惰性导入：加快终端启动
    sd = segment_rates(prep)
    if len(sd) < 4:
        raise ValueError("数据太短：至少需要 4 个 50h 分段（约 200 h 历史）才能拟合衰减律。")
    tt = sd["t_mid"].to_numpy(float)
    rr = sd["rate"].to_numpy(float)
    t0 = prep["t_end"]

    def resid(p):
        a, b, c = p
        return a * np.exp(-b * tt) + c - rr

    best = None
    for x0 in [(2e-4, 1e-3, 0.0), (1e-3, 5e-4, 0.0), (5e-5, 3e-3, 0.0),
               (2e-4, 1e-3, 5e-6), (2e-3, 2e-4, 0.0)]:
        try:
            r = least_squares(resid, x0, bounds=([1e-9, 1e-8, 0.0], [1.0, 5e-2, 1e-2]),
                              xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise ValueError("衰减律拟合失败。")
    a, b, c = [float(v) for v in best.x]
    fitted = a * np.exp(-b * tt) + c
    rmse = float(np.sqrt(np.mean((fitted - rr) ** 2)))

    shape = "free(c≥0)"
    prior_used = False
    b_prior = None
    g0_raw = a * math.exp(-b * t0) + c
    share_end = (a * math.exp(-b * t0) / g0_raw) if g0_raw > 0 else 1.0

    # --- 规则 2：c 主导 → 回到 c=0 纯指数 -------------------------------------
    fit2 = _fit_exp_c0(tt, rr)
    if fit2 is not None:
        a2, b2 = fit2
        rmse2 = float(np.sqrt(np.mean((a2 * np.exp(-b2 * tt) - rr) ** 2)))
        dominated = (c > 1e-12) and (share_end < 0.30 or b * t0 > 4.0)
        if dominated and rmse2 <= rmse * 1.25:
            a, b, c, rmse = a2, b2, 0.0, rmse2
            fitted = a * np.exp(-b * tt)
            shape = "c=0（指数保形：常数项 c 原本主导末端斜率）"

    # --- 规则 3：b 不可辨识 → 历史先验 --------------------------------------
    b_bad = (b <= 1.001e-8) or (b >= 4.99e-2) or (b * t0 < 0.20)
    if b_bad:
        b_prior = _prior_decay_b()
        b = float(b_prior)
        a = _fit_a_given_b(tt, rr, b, c)
        fitted = a * np.exp(-b * tt) + c
        rmse = float(np.sqrt(np.mean((fitted - rr) ** 2)))
        prior_used = True
        shape = "先验 b（本段数据无法辨识衰减速率，用历史记忆库中位数保形）"

    g0 = float(a * math.exp(-b * t0) + c)
    g_inf = float(c)
    # 未来 1000h（≈一个数据段长度）斜率绝对值会降到原来的多少
    ratio_1000 = float((a * math.exp(-b * (t0 + 1000.0)) + c) / g0) if g0 > 0 else 1.0
    return dict(a=a, b=b, c=c, rmse=rmse, t_mid=tt, rate=rr, rate_fit=fitted,
                t0=t0, g0_mVh=g0 * 1e3, n_seg=len(sd),
                shape=shape, prior_used=prior_used, b_prior=b_prior,
                share_end=float(share_end), g_inf_mVh=g_inf * 1e3,
                slope_ratio_1000h=ratio_1000,
                near_linear=bool(ratio_1000 > 0.85),
                # G(t) = |g'(t)| = a·b·e^{-b t}（V/h²）
                G=lambda tt_: a * b * np.exp(-b * np.asarray(tt_, float)))


def g_eval(g: dict, tau) -> np.ndarray:
    tau = np.asarray(tau, float)
    return g["a"] * np.exp(-g["b"] * tau) + g["c"]


# ----------------------------------------------------------------------------
# 4. 分位数回归标定：乘子 d = y / x
# ----------------------------------------------------------------------------
def calibrate(prep: dict, g: dict) -> dict:
    from sklearn.linear_model import QuantileRegressor  # 惰性导入：加快终端启动
    t, u_trend, t0 = prep["t"], prep["u_trend"], prep["t_end"]
    win_steps = int(round(WIN_H / DT))
    stride_steps = int(round(STRIDE_H / DT))
    starts = np.arange(0, max(1, len(u_trend) - win_steps - 2), stride_steps)
    x = g_eval(g, t[starts] + WIN_H / 2.0) * WIN_H * 1e3          # mV 物理预期
    y = (u_trend[starts] - u_trend[starts + win_steps]) * 1e3     # mV 实际趋势降幅
    d = y / x
    # 只去掉非有限值（与 notebook 一致，不做截断）
    keep = np.isfinite(d)
    x, y, d = x[keep], y[keep], d[keep]
    if len(d) < 20:
        raise ValueError("可用标定窗口太少，数据可能过短。")

    qx = np.linspace(max(0.2, float(np.min(x)) * 0.8), float(np.max(x)) * 1.1, 121)
    qf = []
    for q in TAUS:
        m = QuantileRegressor(quantile=float(q), alpha=0.0, solver="highs")
        m.fit(x.reshape(-1, 1), d)
        qf.append(m.intercept_ + m.coef_[0] * qx)
    QF = np.sort(np.asarray(qf), axis=0)                         # 保序
    x_ref = float(np.median(x))
    prof = np.array([np.interp(x_ref, qx, QF[i]) for i in range(len(TAUS))])
    q50 = float(np.interp(0.5, TAUS, prof))
    prof = prof / max(q50, 1e-6)                                  # 中位归一化 → 中位乘子=1
    return dict(x=x, y=y, d=d, qx=qx, QF=QF, x_ref=x_ref, prof=prof, q50=q50,
                n_win=len(d), d_mean=float(d.mean()), d_std=float(d.std()),
                d_p5=float(np.percentile(d, 5)), d_p95=float(np.percentile(d, 95)))


# ----------------------------------------------------------------------------
# 5. 蒙特卡洛：AR(1) 分位水平抽样
# ----------------------------------------------------------------------------
def run_monte_carlo(prep: dict, g: dict, cal: dict,
                    n_mc: int = N_MC, horizon_h: float = HORIZON_H,
                    rho: float = RHO_AR1, seed: int = SEED_MC) -> dict:
    from scipy.stats import norm  # 惰性导入：加快终端启动
    rng = np.random.default_rng(seed)
    t0, u0 = prep["t_end"], prep["u_anchor"]
    n_steps = int(round(horizon_h / WIN_H))
    ks = np.arange(1, n_steps + 1)
    xk = g_eval(g, t0 + (ks - 0.5) * WIN_H) * WIN_H * 1e3        # mV/步（物理预期降幅）
    ta, prof = TAUS, cal["prof"]

    U = np.empty((n_mc, n_steps))
    U[:, 0] = u0
    z = rng.standard_normal(n_mc)
    for k in range(1, n_steps):
        z = rho * z + math.sqrt(1.0 - rho ** 2) * rng.standard_normal(n_mc)
        q = np.clip(norm.cdf(z), Q_CLIP[0], Q_CLIP[1])
        mult = np.interp(q, ta, prof)
        U[:, k] = U[:, k - 1] - xk[k - 1] * mult / 1e3        # mV → V

    uths = prep["uths"]
    R = np.full((n_mc, len(uths)), np.nan)
    for m, uth in enumerate(uths):
        below = U <= uth
        hit = below.any(axis=1)
        first = np.argmax(below, axis=1)
        i1 = np.maximum(first - 1, 0)
        up = np.where(first > 0, U[np.arange(n_mc), i1], u0)
        un = U[np.arange(n_mc), first]
        frac = np.clip((up - uth) / np.maximum(up - un, 1e-12), 0.0, 1.0)
        R[:, m] = np.where(hit, (first + frac) * WIN_H, np.nan)

    # 确定性路径（中位乘子=1）与确定性 RUL
    det = u0 - np.cumsum(xk) / 1e3
    det_curve = np.concatenate([[u0], det])
    det_rul = []
    for uth in uths:
        idx = np.where(det_curve <= uth)[0]
        if len(idx) == 0:
            det_rul.append(float("nan"))
            continue
        i = int(idx[0])
        if i == 0:
            det_rul.append(0.0)
            continue
        up, un = det_curve[i - 1], det_curve[i]
        frac = (up - uth) / max(up - un, 1e-12)
        det_rul.append(float((i - 1 + frac) * WIN_H))

    band_t = t0 + np.concatenate([[0.0], ks * WIN_H])
    band = {p: np.percentile(U, p, axis=0) for p in (5, 25, 50, 75, 95)}
    band = {p: np.concatenate([[u0], v]) for p, v in band.items()}

    stats_rows = []
    for m, ft in enumerate(RUL_FT):
        r = R[:, m]
        rr = r[np.isfinite(r)]
        cens = 100.0 * (1.0 - len(rr) / n_mc)
        stats_rows.append(dict(
            FT=ft, U_th=uths[m], n_valid=len(rr), cens=cens,
            P5=float(np.percentile(rr, 5)) if len(rr) else float("nan"),
            P25=float(np.percentile(rr, 25)) if len(rr) else float("nan"),
            P50=float(np.percentile(rr, 50)) if len(rr) else float("nan"),
            P75=float(np.percentile(rr, 75)) if len(rr) else float("nan"),
            P95=float(np.percentile(rr, 95)) if len(rr) else float("nan"),
            mean=float(rr.mean()) if len(rr) else float("nan"),
            std=float(rr.std()) if len(rr) else float("nan"),
            det=det_rul[m],
        ))
    return dict(U=U.astype(np.float32), R=R, stats=stats_rows,
                band_t=band_t, band=band, det_curve=det_curve, n_steps=n_steps)


def demo_blob(label: str) -> Optional[bytes]:
    """按侧边栏的演示数据标签读文件（找不到对应文件时返回 None）。"""
    for name, path in DEMO_AVAILABLE:
        if name == label and path:
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError:
                return None
    return None


def uth_for_ft(prep: dict, ft: float) -> float:
    """任意失效阈值 FT(%) → 失效电压阈值 U_th。"""
    if prep["has_I"]:
        return float(prep["P_init"] * (1.0 - ft / 100.0) / prep["I_ref"])
    return float(prep["u_start"] * (1.0 - ft / 100.0))


def rul_for_threshold(U, uth: float, u0: float, win: float = WIN_H) -> np.ndarray:
    """从已经算好的电压路径矩阵 U（n_mc × n_steps）现算任意 U_th 下的 RUL。

    返回形状 (n_mc,) 的数组，未穿越的路径为 NaN（窗口内线性插值，与 notebook 同口径）。
    """
    U = np.asarray(U, float)
    n_mc, n_steps = U.shape
    below = U <= uth
    hit = below.any(axis=1)
    first = np.argmax(below, axis=1)
    i1 = np.maximum(first - 1, 0)
    ar = np.arange(n_mc)
    up = np.where(first > 0, U[ar, i1], u0)
    un = U[ar, first]
    frac = np.clip((up - uth) / np.maximum(up - un, 1e-12), 0.0, 1.0)
    return np.where(hit, (first + frac) * win, np.nan)


# ----------------------------------------------------------------------------
# 6. GRU 模型：加载 + 一步预测自检（证明"AI 侧"已接入）
# ----------------------------------------------------------------------------
def _make_grunet_class():
    import torch
    import torch.nn as nn

    class GRUNet(nn.Module):
        """与 v1 notebook 完全一致的骨干（用于反 pickle 与转发）。"""

        def __init__(self, n_feat=10, hidden=64, n_layers=1, init_du=0.5,
                     t_scale_h=1000.0):
            super().__init__()
            self.emb = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU())
            self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
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

    return GRUNet


def _load_state_dict_model(path):
    """加载 models_terminal 的 state_dict 模型（v4 GRU-Transformer 等）。"""
    import models_terminal as mt
    model, obj = mt.load_state_model(path)
    return model, obj


def load_model(path_candidates=None):
    """加载终端模型。

    优先级：
      1) `模型库/pemfc_forecast_v4.pt` —— v4 GRU-Transformer（FC1+FC2 联合训练，state_dict）；
      2) 旧版 `电池v1` pickle（v1 GRU 基准，兜底）。
    返回 (model, 信息dict)；失败返回 (None, 错误信息)。
    """
    import sys

    import torch

    HERE = pathlib.Path(__file__).resolve().parent
    new_candidates = path_candidates or [
        HERE / "模型库" / "pemfc_forecast_v4.pt",
        HERE / "模型库" / "pemfc_forecast.pt",
    ]
    tried = []
    for p in new_candidates:
        p = pathlib.Path(p)
        if not p.exists():
            tried.append(f"{p.name}: 不存在")
            continue
        try:
            model, obj = _load_state_dict_model(p)
            n = int(sum(x.numel() for x in model.parameters()))
            name = "v4 GRU-Transformer" if obj.get("arch", "").startswith("v4") else str(obj.get("arch"))
            return model, dict(ok=True, name=name, path=str(p), n_params=n,
                               arch=obj.get("arch"), trained_at=obj.get("trained_at"),
                               val_rmse=obj.get("val_rmse"), datasets=obj.get("datasets"),
                               source="state_dict")
        except Exception as e:  # noqa: BLE001
            tried.append(f"{p.name}: {type(e).__name__}: {e}")

    # ---- 兜底：旧版 pickle（v1 GRU）--------------------------------------
    GRUNet = _make_grunet_class()
    sys.modules["__main__"].GRUNet = GRUNet
    try:
        torch.serialization.add_safe_globals([GRUNet])
    except Exception:
        pass
    pickle_candidates = [HERE / "电池v1", HERE / "电池v1.pt"] if path_candidates is None \
        else [pathlib.Path(q) for q in path_candidates]
    for p in pickle_candidates:
        p = pathlib.Path(p)
        if not p.exists():
            tried.append(f"{p.name}: 不存在")
            continue
        try:
            model = torch.load(str(p), weights_only=False, map_location="cpu")
            if hasattr(model, "eval"):
                model.eval()
                n = int(sum(x.numel() for x in model.parameters()))
                return model, dict(ok=True, name="v1 GRU（旧版，兜底）", path=str(p),
                                   n_params=n, du0=float(getattr(model, "du0", float("nan"))),
                                   source="pickle")
        except Exception as e:  # noqa: BLE001
            tried.append(f"{p.name}: {type(e).__name__}")
    return None, dict(ok=False, paths=tried)


# ---- 特征与一步自检（与 notebook 同口径） -------------------------------------
def _time_cols(ti, t_ref, gain=3.0):
    ti = np.asarray(ti, float)
    return (ti / t_ref, 2.0 * ti / (ti + t_ref), gain * ti / t_ref,
            np.log1p(ti) / math.log1p(t_ref))


def feature_matrix(prep: dict, t_ref: Optional[float] = None):
    u = np.asarray(prep["u_s"], float)
    t = np.asarray(prep["t"], float)
    I = np.asarray(prep["I_s"], float)
    T = np.asarray(prep["T_s"], float)
    RH = np.asarray(prep["RH_s"], float)
    gain = MODEL_CFG["time_gain"]
    if t_ref is None:
        t_ref = float(prep["t_end"])
    stats = {"u_bar": float(np.mean(u)), "u_scale": 0.05}
    for k, v in (("I", I), ("T", T), ("RH", RH)):
        v = np.asarray(v, float)
        stats[k + "_mu"], stats[k + "_sd"] = float(np.mean(v)), max(float(np.std(v)), 1e-9)
    for k, v in (("dI", np.gradient(I, DT)), ("dT", np.gradient(T, DT))):
        stats[k + "_mu"] = float(np.mean(v))
        stats[k + "_sd"] = max(float(np.std(v)), 1e-9)

    def z(key, v):
        return (np.asarray(v, float) - stats[key + "_mu"]) / stats[key + "_sd"]

    tau, tau_rec, tau_gain, logtau = _time_cols(t, t_ref, gain)
    F = np.stack([(u - stats["u_bar"]) / stats["u_scale"], tau, tau_rec, tau_gain, logtau,
                  z("I", I), z("T", T), z("RH", RH),
                  z("dI", np.gradient(I, DT)), z("dT", np.gradient(T, DT))], axis=1)
    return F.astype(np.float32), stats


def one_step_selfcheck(model, prep: dict, seq: int = 50, n_win: int = 1200) -> Optional[dict]:
    """用最后 n_win 个窗口做一步预测，返回 RMSE(µV/步) 与相关系数。"""
    if model is None:
        return None
    try:
        import torch
        F, stats = feature_matrix(prep)
        u = np.asarray(prep["u_s"], float)
        t = np.asarray(prep["t"], float)
        M = len(F) - seq
        start = max(0, M - n_win)
        X = np.lib.stride_tricks.sliding_window_view(F, seq, axis=0)[start:M]
        X = np.ascontiguousarray(X.transpose(0, 2, 1))
        du = np.diff(u)
        # 与预测窗口同跨度的全局单步增量 σ（与 notebook 口径一致，约 10 µV）
        sig = float(np.std(du[seq - 1:M + seq - 1])) or 1.0
        y = du[seq - 1 + start:M + seq - 1] / sig
        tau = t[seq - 1 + start:M + seq - 1].astype(np.float32)
        preds = []
        with torch.no_grad():
            for s in range(0, len(X), 8192):
                xb = torch.from_numpy(X[s:s + 8192])
                tb = torch.from_numpy(tau[s:s + 8192])
                preds.append(model(xb, tb).numpy())
        pred = np.concatenate(preds)
        rmse_uV = float(np.sqrt(np.mean((pred - y) ** 2)) * sig * 1e6)   # V → µV
        corr = float(np.corrcoef(pred, y)[0, 1]) if len(pred) > 2 else float("nan")
        return dict(rmse_uV=rmse_uV, corr=corr, n=len(pred),
                    sig_uV=float(sig * 1e6))
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# 7. 总入口（给 Streamlit 缓存用）
# ----------------------------------------------------------------------------
def analyze(blob: bytes, model=None, n_mc: int = N_MC, horizon_h: float = HORIZON_H) -> dict:
    df = load_csv(blob)
    prep = prepare(df)
    g = fit_decay(prep)
    cal = calibrate(prep, g)
    mc = run_monte_carlo(prep, g, cal, n_mc=n_mc, horizon_h=horizon_h)
    ai = one_step_selfcheck(model, prep)
    return dict(ok=True, n_rows=prep["n_rows"], t_end=prep["t_end"],
                u_anchor=prep["u_anchor"], u_start=prep["u_start"],
                t=prep["t"], u_s=prep["u_s"], u_trend=prep["u_trend"],
                g=dict(a=g["a"], b=g["b"], c=g["c"], rmse=g["rmse"],
                       g0_mVh=g["g0_mVh"], n_seg=g["n_seg"],
                       t_mid=g["t_mid"], rate=g["rate"], rate_fit=g["rate_fit"]),
                cal=dict(n_win=cal["n_win"], x_ref=cal["x_ref"], x=cal["x"], y=cal["y"],
                         d=cal["d"], prof=cal["prof"], ta=TAUS,
                         d_mean=cal["d_mean"], d_std=cal["d_std"],
                         d_p5=cal["d_p5"], d_p95=cal["d_p95"]),
                mc=dict(stats=mc["stats"], band_t=mc["band_t"], band=mc["band"],
                        det_curve=mc["det_curve"], n_steps=mc["n_steps"],
                        R=mc["R"], U=mc["U"], horizon_h=horizon_h, n_mc=n_mc),
                ai=ai,
                has_I=prep["has_I"], P_init=prep["P_init"], I_ref=prep["I_ref"],
                uths=prep["uths"])


def fmt_h(v) -> str:
    """小时数格式化。"""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "未穿越"
    if v <= 0.5:
        return "0 h（已失效）"
    return f"{v:.0f} h"
