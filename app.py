# -*- coding: utf-8 -*-
"""质子交换膜燃料电池（PEMFC）剩余寿命预测终端 · v2（简洁版）

主界面只保留两张图：
  ① 预测曲线（历史 + 约束后外推交付曲线 + 当前 FT 阈值 + 跨越时刻）
  ② 当前 FT 下的 RUL 概率直方图（可随 FT 任意调节）

其余内容（数据预览、衰减律、全阈值表、不确定性扇形带、历史数据记忆库、
文件路径、模型版本对比）全部收进侧边栏的可折叠区。

"学习框架"
----------
每次预测后自动把本次数据打包存进 ``数据记忆库/``（元信息 + 乘子样本），
可随时在侧边栏查看/剔除。勾选"历史数据增强"后，概率层会把历史数据集的
乘子分布与本次合并（数据集等权），使小样本/短记录也能得到更稳的概率区间。
"""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import data_memory as dm
import live_canvas as lc
import rul_pipeline as rp

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "输出"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DIAG_LOG = HERE / "终端诊断.log"


def log_exc(tag: str, ex: BaseException) -> str:
    """把异常写进 终端诊断.log（附时间戳与 traceback），返回简短文本。"""
    import traceback
    from datetime import datetime as _dt
    tb = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
    try:
        with open(DIAG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n===== {_dt.now():%Y-%m-%d %H:%M:%S} [{tag}] =====\n{tb}")
    except OSError:
        pass
    return tb


FT_EFFECTIVE_AUTOBUMP_MARK = True
VERSION_JSON = HERE / "模型库" / "版本对比.json"

st.set_page_config(page_title="PEMFC 剩余寿命预测终端", page_icon="🔋", layout="wide")


# ============================================================ 缓存与模型
@st.cache_resource(show_spinner=False)
def _model():
    return rp.load_model()


MODEL, MODEL_INFO = None, {"ok": False, "lazy": True}


def _get_model():
    """首次需要时才加载模型（torch 导入约 2 s，避免拖慢首屏）。"""
    global MODEL, MODEL_INFO
    if MODEL is None and not MODEL_INFO.get("tried"):
        MODEL_INFO = {"ok": False, "tried": True}
        try:
            MODEL, MODEL_INFO = _model()
        except Exception as ex:  # noqa: BLE001
            MODEL_INFO = {"ok": False, "tried": True, "error": str(ex)}
    return MODEL


def _png(fig: plt.Figure, dpi: int = 110) -> bytes:
    """把 matplotlib 图转成 PNG 字节，并立刻关闭图（避免内存与重复渲染开销）。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# ============================================================ 读表（多编码/多格式容错）
@st.cache_data(show_spinner=False, max_entries=8)
def _parse_blob(blob: bytes):
    """缓存解析结果：同一次上传的 rerun 不再重复解析 20 MB 文件。"""
    return rp.read_any(blob)


@st.cache_data(show_spinner=False, max_entries=8)
def _load_demo(blob: bytes, time_unit: str = "auto"):
    return rp.load_csv(blob, time_unit=time_unit)


@st.cache_data(show_spinner=False, max_entries=6)
def _load_many(blobs: tuple, time_unit: str = "auto"):
    return rp.load_csv_many(list(blobs), time_unit=time_unit)


# ============================================================ 绘图
def _fig_prediction(prep, g, mc, ft: float, uth: float) -> plt.Figure:
    t0 = prep["t_end"]
    u0 = prep["u_anchor"]
    det = np.asarray(mc["det_curve"], float)
    t_ext = t0 + np.arange(len(det)) * rp.WIN_H

    rr = rp.rul_for_threshold(np.asarray(mc["U"], float), uth=uth, u0=u0)

    fig, ax = plt.subplots(figsize=(11.2, 4.3))
    t, u = prep["t"], prep["u_s"]
    lo = max(0.0, t0 - 700.0)
    m = t >= lo
    _step = max(1, int(m.sum() // 1200))          # 降采样到 ~1200 点，渲染更快
    ax.plot(t[m][::_step], u[m][::_step], color="#2c6fbb", lw=1.1,
            label=f"历史电压（10 h 平滑，0–{t0:.0f} h）")
    ax.plot(t_ext, det, color="k", lw=2.4, label="约束后预测曲线（交付）")
    ax.axhline(uth, color="#c0392b", lw=1.4, ls="--",
               label=f"FT={ft:.1f}% 阈值 = {uth:.4f} V")
    ax.axvline(t0, color="#7f8c8d", lw=1.0, ls=":")
    ax.text(t0, u0, " 预测起点", color="#7f8c8d", fontsize=9, va="bottom")

    valid = rr[np.isfinite(rr)]
    if len(valid) > 0:
        p50 = float(np.percentile(valid, 50))
        ax.axvline(t0 + p50, color="#c0392b", lw=1.2, ls="--", alpha=0.85)
        ax.text(t0 + p50, ax.get_ylim()[0] if False else min(det) + 0.3 * (u0 - min(det)),
                f" P50 穿越 ≈ {p50:.0f} h", color="#c0392b", fontsize=9)

    right = max(t0 + (float(np.percentile(valid, 95)) * 1.35 + 100.0) if len(valid) else t0 + 500.0,
                t0 + 300.0)
    ax.set_xlim(lo, min(right, t_ext[-1]))
    seg = (t_ext >= lo) & (t_ext <= min(right, t_ext[-1]))
    ys = np.concatenate([u[m], det[seg], [uth]])
    pad = max(0.01, 0.06 * (ys.max() - ys.min()))
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_xlabel("时间 t（h）")
    ax.set_ylabel("电压 U（V）")
    ax.set_title("预测曲线：历史 + 约束后外推", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    return fig


def _fig_hist(rr: np.ndarray, ft: float, uth: float) -> plt.Figure:
    valid = np.asarray(rr, float)
    valid = valid[np.isfinite(valid)]
    fig, ax = plt.subplots(figsize=(11.2, 3.9))
    if len(valid) == 0:
        ax.text(0.5, 0.52, "该阈值下电池已失效（RUL = 0 h）", ha="center", va="center",
                fontsize=15, color="#c0392b", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"FT={ft:.1f}%（U_th={uth:.4f} V）的 RUL 分布", fontsize=13)
        fig.tight_layout()
        return fig

    p5, p50, p95 = np.percentile(valid, [5, 50, 95])
    ax.hist(valid, bins=45, color="#5b9bd5", edgecolor="white", linewidth=0.6)
    ax.axvline(p50, color="#c0392b", lw=2.0, label=f"P50 = {p50:.0f} h")
    ax.axvline(p5, color="#7b5ea7", lw=1.3, ls="--", label=f"P5 = {p5:.0f} h")
    ax.axvline(p95, color="#7b5ea7", lw=1.3, ls="--", label=f"P95 = {p95:.0f} h")
    ax.set_xlabel("剩余寿命 RUL（h）")
    ax.set_ylabel("路径数 / 柱")
    ax.set_title(f"FT={ft:.1f}%（U_th={uth:.4f} V）下的 RUL 概率分布"
                 f"（{len(valid)} 条有效路径）", fontsize=13)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def _fig_band(prep, mc) -> plt.Figure:
    bt, band = mc["band_t"], mc["band"]
    det = mc["det_curve"]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.fill_between(bt, band[5], band[95], color="#7fb3d5", alpha=0.35, label="P5–P95（90%）")
    ax.fill_between(bt, band[25], band[75], color="#2f6fb3", alpha=0.45, label="P25–P75（50%）")
    ax.plot(bt, band[50], color="#12395e", lw=1.8, label="中位 P50")
    ax.plot(prep["t_end"] + np.arange(len(det)) * rp.WIN_H, det, "r--", lw=1.8, label="确定性曲线")
    ax.set_xlabel("时间（h）")
    ax.set_ylabel("电压（V）")
    ax.set_title("泛化数据：电压扇形带（历史增强后的概率区间）", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return fig


# ============================================================ 分析流程
def run_analysis(df: pd.DataFrame, name: str, use_history: bool, hist_ids: list[str]) -> dict:
    prep = rp.prepare(df)
    g = rp.fit_decay(prep)
    cal = rp.calibrate(prep, g)
    fp = dm.fingerprint(prep["n_rows"], prep["t_end"])
    prof, n_sets = (dm.pooled_profile(cal, hist_ids, rp.TAUS, exclude_fp=fp) if use_history
                    else (np.asarray(cal["prof"], float), 1))
    cal_use = dict(cal, prof=prof)
    mc = rp.run_monte_carlo(prep, g, cal_use)
    _m = _get_model()
    st.session_state["model_loaded"] = _m is not None
    ai = rp.one_step_selfcheck(_m, prep) if _m is not None else None
    quality = dm.quality_check(prep)
    ds_id = dm.save_dataset(name, prep, g, cal, extra=dict(quality="; ".join(quality)))
    return dict(ok=True, name=name, prep=prep, g=g, cal=cal, cal_use=cal_use,
                mc=mc, ai=ai, quality=quality, ds_id=ds_id,
                n_history=n_sets, use_history=use_history, fp=fp)


def _save_outputs(png1: bytes, png2: bytes, res, ft, uth, rr) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = OUTPUT_ROOT / f"预测_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "图1_预测曲线.png").write_bytes(png1)
    (d / "图2_RUL分布.png").write_bytes(png2)
    valid = rr[np.isfinite(rr)]
    pd.DataFrame({
        "FT(%)": [ft],
        "U_th(V)": [uth],
        "n_valid": [len(valid)],
        "P5_h": [np.percentile(valid, 5) if len(valid) else np.nan],
        "P25_h": [np.percentile(valid, 25) if len(valid) else np.nan],
        "P50_h": [np.percentile(valid, 50) if len(valid) else np.nan],
        "P75_h": [np.percentile(valid, 75) if len(valid) else np.nan],
        "P95_h": [np.percentile(valid, 95) if len(valid) else np.nan],
    }).to_csv(d / "RUL分位数.csv", index=False, encoding="utf-8-sig")
    return d


# ============================================================ 侧边栏
blob, data_name = None, ""
df_input, parse_err = None, None
blobs = []
_tu = "auto"

with st.sidebar:
    st.header("⚙️ 设置")
    src = st.radio("数据来源", ["内置演示数据", "上传数据"], horizontal=True)

    _tu_label = st.selectbox(
        "时间列单位", [c[0] for c in rp.TIME_UNIT_CHOICES], index=0,
        help="自动识别失败时手动指定（同济动态工况数据的时间列是“秒”）。"
             "旧数据/演示数据一律按小时。")
    _tu = dict(rp.TIME_UNIT_CHOICES)[_tu_label]

    if src == "内置演示数据":
        label = st.selectbox("选择演示数据", rp.DEMO_LABELS, index=0)
        blob = rp.demo_blob(label)
        data_name = label.split(" ·")[0] + "_" + label.split("(")[0].split("·")[-1].strip()
        if blob is None:
            parse_err = "演示数据文件不存在，请改用「上传数据」。"
        else:
            try:
                df_input = _load_demo(blob, _tu)
            except Exception as ex:  # noqa: BLE001
                parse_err = f"演示数据读取失败：{ex}"
    else:
        ups = st.file_uploader(
            "上传电池老化数据（CSV / TSV / Excel，可多选；分段文件会自动接续合并）",
            type=["csv", "tsv", "txt", "xlsx", "xls"], accept_multiple_files=True)
        blobs = [u.getvalue() for u in ups] if ups else []
        if len(blobs) == 1:
            blob = blobs[0]
            up = ups[0]
            data_name = Path(up.name).stem
            st.caption(f"文件：{up.name}（{len(blob) / 1024:.0f} KB）")
            try:
                raw_df, _info = _parse_blob(blob)
                st.caption(f"解析：{_info['encoding']} 编码 · {_info['sep']} 分隔 · "
                           f"表头第 {_info['header_row']} 行 · {_info['n_raw_rows']} 行数据")
                auto_map, auto_info = rp.auto_map_columns(raw_df)
                _cols = [str(c) for c in raw_df.columns]
                with st.expander("🔧 列名识别不对？手动指定", expanded=False):
                    def _pick(label, canon, auto_col):
                        opts = ["（自动）"] + _cols
                        default = _cols.index(auto_col) + 1 if auto_col in _cols else 0
                        return st.selectbox(label, opts, index=default, key=f"map_{canon}")

                    _t = _pick("时间列", "Time (h)", auto_info["time"])
                    _u = _pick("电压列", "Utot (V)", auto_info["voltage"])
                    _i = _pick("电流列（可选）", "I (A)", auto_info["current"])
                    _tt = _pick("温度列（可选）", "TinAIR (°C)", auto_info["temp"])
                    _rh = _pick("湿度列（可选）", "HrAIRFC (%)", auto_info["rh"])

                    def _chosen(pick, auto):
                        return auto if pick == "（自动）" else pick

                    map_use = {
                        "Time (h)": _chosen(_t, auto_info["time"]),
                        "Utot (V)": _chosen(_u, auto_info["voltage"]),
                        "I (A)": _chosen(_i, auto_info["current"]),
                        "TinAIR (°C)": _chosen(_tt, auto_info["temp"]),
                        "HrAIRFC (%)": _chosen(_rh, auto_info["rh"]),
                    }
                    st.caption(f"当前使用：时间 ← `{map_use['Time (h)']}`，电压 ← `{map_use['Utot (V)']}`")
                st.caption(f"自动识别：时间 ← `{auto_info['time']}`，电压 ← `{auto_info['voltage']}`"
                           + (f"，电流 ← `{auto_info['current']}`" if auto_info["current"] else ""))
                st.session_state["map_sig"] = "|".join(f"{k}={v}" for k, v in sorted(map_use.items()))
                df_input = rp.make_canonical(raw_df, map_use, time_unit=_tu)
            except Exception as ex:  # noqa: BLE001
                parse_err = str(ex)
        elif len(blobs) > 1:
            blob = blobs[0]
            data_name = Path(ups[0].name).stem + f" 等{len(blobs)}个文件"
            st.caption("多文件模式：" + "、".join(u.name for u in ups[:4])
                       + ("…" if len(ups) > 4 else ""))
            try:
                df_input = _load_many(tuple(blobs), _tu)
                st.session_state["map_sig"] = f"multi{len(blobs)}"
                st.caption(f"已合并 {df_input.attrs.get('merged_files', len(blobs))} 个文件 → "
                           f"共 {len(df_input):,} 行，时间 0 → "
                           f"{float(df_input['Time (h)'].iloc[-1]):.1f} h"
                           f"（分段文件已按时间接续）")
            except Exception as ex:  # noqa: BLE001
                parse_err = str(ex)
        else:
            parse_err = None

    _raw_unit = (df_input.attrs.get("time_unit") if df_input is not None else None) or "h"
    if df_input is not None:
        st.caption(f"时间列解读：{rp._TIME_UNIT_NAME.get(_raw_unit, _raw_unit)}"
                   + ("（已换算为小时）" if _raw_unit != "h" else "")
                   + f" · 有效时长 {float(df_input['Time (h)'].iloc[-1]):.1f} h")
        if df_input.attrs.get("unit_fallback"):
            st.caption(f"ℹ️ 手动指定的时间单位不适用（换算后不足 50 h），"
                       f"已自动识别为「{rp._TIME_UNIT_NAME.get(_raw_unit, _raw_unit)}」重新解析。")
    if df_input is not None and df_input.attrs.get("merged_files"):
        st.caption(f"ℹ️ 已按时间接续合并 {df_input.attrs['merged_files']} 段数据。")

    st.divider()
    ft = st.slider("记录阈值 FT（% 初始功率）", 3.0, 8.0, 4.5, 0.1,
                   help="决定上方指标行与「保存」的内容；图上滑条只用于交互预览，二者互不影响")
    map_sig = st.session_state.get("map_sig", "") if src == "上传 CSV" else ""
    use_history = st.toggle("历史数据增强（概率层）", value=True,
                            help="把记忆库中其他数据集的乘子分布与本数据合并，概率区间更稳")

    st.divider()
    with st.expander("🧠 历史数据记忆库", expanded=False):
        records = dm.list_datasets()
        st.caption(f"共 {len(records)} 份数据；每次预测自动存档，可单独剔除坏数据")
        if records:
            ids = [r["id"] for r in records]
            labels = [f'{r["created_at"][5:16].replace("T", " ")} · {r["name"][:18]}' for r in records]
            _prev = [i for i in st.session_state.get("hist_sel", ids) if i in ids]
            st.session_state["hist_sel"] = _prev
            chosen = st.multiselect(
                "参与概率层合并的数据集", ids, default=_prev,
                format_func=lambda i: labels[ids.index(i)] if i in ids else str(i),
                key="hist_sel")
            for r in records[:12]:
                c1, c2 = st.columns([5, 1])
                c1.markdown(
                    f'**{r["name"][:22]}**  \n'
                    f'`{r["created_at"][5:16].replace("T", " ")}` · {r["t_end"]:.0f} h · '
                    f'g₀={r["g0_mVh"]:.4f} mV/h · n_win={r["n_win"]}')
                if c2.button("🗑", key=f"del_{r['id']}"):
                    dm.delete_dataset(r["id"])
                    st.rerun()
        else:
            chosen = []
            st.caption("暂无存档：第一次预测后会自动写入。")

    with st.expander("📊 其余数据", expanded=False):
        st.caption("预测后可查看：数据预览 / 衰减律 / 全阈值表 / AI 自检")
        _slot_data = st.container()
    with st.expander("🌡️ 泛化数据（不确定性扇形带）", expanded=False):
        _slot_band = st.container()
    with st.expander("📈 历史趋势对比（学习记忆）", expanded=False):
        _slot_trend = st.container()
    with st.expander("📁 图表与数据文件路径", expanded=False):
        _slot_paths = st.container()
    with st.expander("ℹ️ 模型版本对比", expanded=False):
        _slot_ver = st.container()

# ============================================================ 兼容旧变量名
hist_ids = chosen


# ============================================================ 主界面
st.title("🔋 质子交换膜燃料电池（PEMFC）剩余寿命预测")
_n_hist = len(dm.list_datasets())
_minfo = MODEL_INFO if MODEL_INFO.get("ok") else None
_mname = (_minfo or {}).get("name") or "v4 GRU-Transformer（首次预测时加载）"
_npar = (_minfo or {}).get("n_params")
_mtxt = f"{_mname}（{_npar:,} 参数）" if _npar else _mname
st.caption("物理约束指数衰减律 · 分位数回归概率层 · 盲测自回归 ｜ "
           f"模型：{_mtxt} ｜ 数据记忆库：{_n_hist} 份历史")

key = None
if blob is not None:
    import hashlib
    _hash_src = b"".join(blobs) if blobs else blob
    key = hashlib.md5(_hash_src).hexdigest()[:16]

top1, top2, top3 = st.columns([1, 2, 2])
with top1:
    go = st.button("🔮 开始预测", type="primary", width="stretch")
with top2:
    if MODEL_INFO.get("ok"):
        _ds = MODEL_INFO.get("datasets") or []
        _dstxt = ("｜训练数据：" + "、".join(_ds)) if _ds else ""
        st.success(f"AI 模型已加载：{MODEL_INFO.get('name')}"
                   f"（{MODEL_INFO.get('n_params', 0):,} 参数，常驻内存）{_dstxt}", icon="✅")
    elif MODEL_INFO.get("tried"):
        st.info("AI 模型文件未找到，仅使用物理层（仍可预测）。", icon="ℹ️")
    else:
        st.caption("AI 模型将在首次预测时加载（≈2 s，之后常驻内存）。")
with top3:
    if key:
        st.caption(f"当前数据指纹 `{key}` ｜ 历史增强：{'开' if use_history else '关'}")

if df_input is not None and df_input.attrs.get("downsampled"):
    st.caption(f"ℹ️ 数据量较大：已等间隔抽稀到 {len(df_input):,} 行"
               f"（原始 {int(df_input.attrs.get('downsampled_from', 0)):,} 行）以加速分析。")

if parse_err:
    st.error(f"数据读取失败：{parse_err}")

cache_key = f"res_{key}_{map_sig}_{_tu}_{bool(use_history)}_{','.join(hist_ids)}"

if go and df_input is not None:
    with st.spinner("正在拟合衰减律、蒙特卡洛推演 4000 条路径……"):
        try:
            st.session_state[cache_key] = run_analysis(df_input, data_name, use_history, hist_ids)
        except Exception as ex:  # noqa: BLE001
            _tb = log_exc("点击预测", ex)
            st.error(f"分析失败：{ex}")
            with st.expander("查看错误详情（已写入 终端诊断.log）", expanded=False):
                st.code(_tb[-4000:])
            st.session_state.pop(cache_key, None)

res = st.session_state.get(cache_key)
# 只要当前"数据 + 列映射 + 历史增强设置"的组合还没算过，就自动算一次
_done = st.session_state.setdefault("done_keys", set())
if res is None and df_input is not None and cache_key not in _done:
    _done.add(cache_key)
    try:
        with st.spinner("正在预测……"):
            st.session_state[cache_key] = run_analysis(df_input, data_name, use_history, hist_ids)
        res = st.session_state.get(cache_key)
    except Exception as ex:  # noqa: BLE001
        _tb = log_exc("自动预测", ex)
        st.error(f"分析失败：{ex}")
        with st.expander("查看错误详情（已写入 终端诊断.log）", expanded=False):
            st.code(_tb[-4000:])
        res = None

if res is None:
    if df_input is None and not parse_err:
        st.info("👈 在左侧选择演示数据或上传 CSV，然后点「开始预测」。")
    st.stop()

prep, g, mc = res["prep"], res["g"], res["mc"]
_u_eq = (prep.get("P_init", 0.0) / prep["I_ref"]) if (prep.get("has_I") and prep.get("I_ref")) \
    else float(prep.get("u_start", prep["u_anchor"]))
_cur_pct = 100.0 * (1.0 - prep["u_anchor"] / _u_eq) if _u_eq > 0 else 0.0
_sug_ft = min(8.0, max(3.0, np.ceil((_cur_pct + 0.6) * 10.0) / 10.0))
PROJECT_FT_INVALID_MARK = True


def _ft_already_failed(prep: dict, uth: float, valid) -> bool:
    """FT 是否已提前失效：阈值已高于当前电压，或所有路径 RUL 都是 0。"""
    u0 = float(prep["u_anchor"])
    if float(uth) >= u0 - 1e-12:
        return True
    if len(valid) == 0:
        return True
    return bool(float(np.max(valid)) <= 1e-9)

uth = rp.uth_for_ft(prep, ft)
rr = rp.rul_for_threshold(np.asarray(mc["U"], float), uth=uth, u0=prep["u_anchor"])
valid = rr[np.isfinite(rr)]

# 记录阈值在这个电池上已失效时，指标行/保存自动改用"建议阈值"（画布滑条仍可任意预览）
ft_eff, uth_eff, rr_eff, valid_eff, _auto = ft, uth, rr, valid, False
_failed = _ft_already_failed(prep, uth, valid)
if _failed and _sug_ft > ft + 1e-9:
    ft_eff = float(_sug_ft)
    uth_eff = rp.uth_for_ft(prep, ft_eff)
    rr_eff = rp.rul_for_threshold(np.asarray(mc["U"], float), uth=uth_eff, u0=prep["u_anchor"])
    valid_eff = rr_eff[np.isfinite(rr_eff)]
    _auto = True

# ---------------- 指标行 + 两张主图 ----------------
if _failed and not _auto:
    st.warning(
        f"FT={ft:.1f}%：该电池在当前已低于此失效阈值，RUL = 0 h（已失效）。"
        f"当前已衰减约 {_cur_pct:.2f}%，建议把 FT 调到 **{_sug_ft:.1f}%** 以上再看预测。",
        icon="⚠️")
    c1, c2, c3 = st.columns(3)
    c1.metric("P50 剩余寿命", "0 h")
    c2.metric("P5", "0 h")
    c3.metric("P95", "0 h")
else:
    if _auto:
        st.info(f"FT={ft:.1f}% 下该电池已失效，指标行已自动改用建议阈值 **FT={ft_eff:.1f}%**"
                f"（保存同样按此阈值）；画布内滑条仍可任意预览。", icon="ℹ️")
    p5, p50, p95 = np.percentile(valid_eff, [5, 50, 95])
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    c1.metric(f"中位剩余寿命 P50（FT={ft_eff:.1f}%）", f"{p50:.0f} h", help="50% 路径在此前穿越阈值")
    c2.metric("P5（乐观侧）", f"{p5:.0f} h")
    c3.metric("P95（悲观侧）", f"{p95:.0f} h")
    with c4:
        st.caption(f"有效路径 {len(valid_eff)}/4000 ｜ 删失 {100*(1-len(valid_eff)/4000):.0f}%"
                   f" ｜ 历史增强 {res['n_history']} 份")
        if res["quality"]:
            st.caption("⚠️ " + "；".join(res["quality"]))

# ---------- 交互画布：结构 A 只画一次，结构 B 由浏览器端 FT 参数本地联动 ----------
# 设计见 live_canvas.py：静态层=离屏 canvas 缓存（贴图 <1ms），动态层=阈值红线/P50/直方图
# 全部在浏览器本地 rAF 合帧重绘，拖动滑条不触发任何服务器计算 → 前端不会卡死。
_u_eq_full = (prep.get("P_init", 0.0) / prep["I_ref"]) if (prep.get("has_I") and prep.get("I_ref")) \
    else float(prep.get("u_start", prep["u_anchor"]))
_live_key = f"{cache_key}|{res.get('fp', 'x')}"
_live_cache = st.session_state.setdefault("live_cache", {})
try:
    _theme = lc.pick_theme(getattr(st.context, "theme", None))
except Exception:  # noqa: BLE001
    _theme = lc.pick_theme(None)

if _live_key not in _live_cache:
    # 先占位显示闪烁的「加载中……」，再做一次性计算（约 0.2–1 s），避免页面假死
    _ph = st.empty()
    _ph.markdown(
        '<div style="padding:34px 0;text-align:center;font-size:14.5px;letter-spacing:1px;">'
        '正在准备交互画布，加载中'
        '<span style="display:inline-block;animation:__blink 1.4s infinite;">···</span>'
        '<style>@keyframes __blink{0%,20%{opacity:1}50%{opacity:.15}80%,100%{opacity:1}}</style>'
        '</div>',
        unsafe_allow_html=True)
    try:
        with st.spinner("正在把 FT 网格的 RUL 分位与直方图打包成可本地联动的画布……"):
            _live_cache.clear()
            _live = lc.build_live_data(prep, g, mc, u_eq=_u_eq_full)
            _ft_start = float(np.clip(float(ft), _live["ft"][0], _live["ft"][-1]))
            _live_cache[_live_key] = dict(
                live=_live,
                html=lc.build_html(_live, ft_default=_ft_start, theme=_theme),
            )
    finally:
        _ph.empty()          # 清掉占位（成功或失败都清，后续再显示错误）
_live_entry = _live_cache[_live_key]
live = _live_entry["live"]
components.html(_live_entry["html"], height=lc.PRED_H + lc.HIST_H + 150, scrolling=False)
st.caption("拖动**画布内 FT 滑条**：阈值红线、P50 穿越竖线与下方直方图全部在浏览器本地即时联动"
           "（结构 A 静态层只画一次并缓存，结构 B 动态层本地重绘），不触发服务器重算。"
           f"上方指标行与「保存」按左侧记录阈值 FT={ft_eff:.1f}% 计算。")

# 衰减律形态：把"这是一条指数型函数"显式展示出来
_g_shape = g.get("shape", "free(c≥0)")
_g_ratio = float(g.get("slope_ratio_1000h", float("nan")))
_g_end = float(g.get("g0_mVh", float("nan")))
_g_c = float(g.get("c", 0.0)) * 1e3
_stxt = (f"衰减律形态：**{_g_shape}** ｜ g(t_end)={_g_end:.4f} mV/h ｜ 常数项 c={_g_c:.4f} mV/h"
         f" ｜ 未来 1000h 的斜率绝对值 ≈ 当前的 {_g_ratio*100:.0f}%")
if g.get("prior_used"):
    _stxt += f" ｜ 先验 b={float(g.get('b_prior') or 0):.3e}"
st.caption(_stxt)
if g.get("near_linear"):
    nearly_linear_hint = True
    st.caption("⚠️ 该数据本身的衰减已接近**线性**（指数项很弱），所以外推也会接近直线——"
               "这是数据特性，不是模型忘了指数形状；若要得到更凸的外推，需要更早的历史段。")



# 保存按需触发（按左侧记录阈值生成两张 PNG + RUL 分位数表，避免每次拖滑条都写盘）
_c1, _c2 = st.columns([1, 5])
with _c1:
    _save_click = st.button("💾 保存本次结果",
                            help=f"按左侧记录阈值 FT={ft:.1f}% 保存 PNG 与 CSV 到 输出\\ 目录")
if _save_click:
    with st.spinner("正在保存……"):
        _png1 = _png(_fig_prediction(prep, g, mc, ft_eff, uth_eff))
        _png2 = _png(_fig_hist(rr_eff, ft_eff, uth_eff))
        st.session_state["last_out_dir"] = str(
            _save_outputs(_png1, _png2, res, ft_eff, uth_eff, rr_eff))
out_dir = Path(st.session_state.get("last_out_dir", OUTPUT_ROOT))

# ---------------- 侧边栏折叠区内容 ----------------
with _slot_data.container():
    t, u_s, u_trend = prep["t"], prep["u_s"], prep["u_trend"]
    step = max(1, len(t) // 800)
    st.markdown(f"**数据**：{prep['n_rows']} 行 → 网格 {len(t)} 点，末点 **{prep['t_end']:.2f} h**，"
                f"U_anchor={prep['u_anchor']:.4f} V，P_init={prep['P_init']:.1f} W，I_ref={prep['I_ref']:.2f} A")
    st.line_chart(pd.DataFrame({"电压 U(V)": u_s[::step]}, index=t[::step]), height=160)
    st.markdown(f"**衰减律**：g(t) = {g['a']*1e3:.5f}·e^(−{g['b']:.6f} t) + {g['c']*1e3:.5f} mV/h"
                f"（RMSE {g['rmse']:.4f} mV/h，{g['n_seg']} 段）\n\n"
                f"g(t₀) = **{g['g0_mVh']:.5f} mV/h**，曲率上限 G(t₀) = {g['G'](g['t0']):.3e} V/h²")
    rows = []
    for row in mc["stats"]:
        rows.append({"FT(%)": row["FT"], "U_th(V)": row["U_th"],
                     "P5": row["P5"], "P50": row["P50"], "P95": row["P95"],
                     "删失%": row["cens"]})
    st.markdown("**全阈值表（预设 3.5–6.0%）**")
    st.dataframe(pd.DataFrame(rows), width="stretch", height=230, hide_index=True)
    if res["ai"]:
        st.caption(f"AI 一步自检：RMSE={res['ai']['rmse_uV']:.1f} µV/步（增量 σ≈{res['ai']['sig_uV']:.1f} µV）")

with _slot_band.container():
    _band_cache = st.session_state.setdefault("band_cache", {})
    _bk = str(res.get("fp", "x"))
    if _bk not in _band_cache:
        _band_cache.clear()
        _band_cache[_bk] = _png(_fig_band(prep, mc))
    st.image(_band_cache[_bk], width="stretch")
    st.caption("浅色带 P5–P95（90% 概率）、深色带 P25–P75（50% 概率）；红虚线=确定性交付曲线")

with _slot_trend.container():
    _records = dm.list_datasets()
    _cur_fp = res.get("fp")
    _cur_g0 = res["g"]["g0_mVh"]
    st.caption("每次预测都会记住一条「衰减趋势」；这里把当前数据与历史数据放在一起对比，"
               "越接近说明历史记忆越可借鉴（概率层会等权合并它们的乘子分布）。")
    _rows = []
    for _r in _records[:8]:
        if _r.get("fingerprint") == _cur_fp:
            continue
        _rows.append({
            "历史数据": _r["name"][:18],
            "记录长度(h)": round(_r["t_end"], 1),
            "g₀(mV/h)": round(_r["g0_mVh"], 5),
            "与当前g₀偏差(%)": round(100.0 * (_r["g0_mVh"] - _cur_g0) / max(abs(_cur_g0), 1e-9), 1),
            "乘子均值": round(_r["d_mean"], 3),
        })
    if _rows:
        st.dataframe(pd.DataFrame(_rows), width="stretch", height=180, hide_index=True)
    else:
        st.caption("（还没有其它历史数据；再投喂一份数据后即可对比）")
    _tr_step = max(1, len(t) // 600)          # 降采样，避免把 9 万点丢给前端
    _idx = t[::_tr_step]
    _trend_df = pd.DataFrame({"当前 u_trend (V)": np.asarray(prep["u_trend"])[::_tr_step]}, index=_idx)
    for _r in _records[:3]:
        if _r.get("fingerprint") == _cur_fp:
            continue
        _tr = dm.load_trend(_r["id"])
        if _tr is None:
            continue
        _ser = pd.Series(_tr[1], index=_tr[0])
        _trend_df[f'历史 {_r["created_at"][5:16].replace("T", " ")}'] = _ser.reindex(
            _trend_df.index, method="nearest")
    st.line_chart(_trend_df, height=200)

with _slot_paths.container():
    st.code(str(out_dir), language=None)
    st.caption("每次预测都会新建一个时间戳文件夹，含：图1_预测曲线.png、图2_RUL分布.png、RUL分位数.csv")
    st.code(str(dm.MEMORY_DIR), language=None)
    st.caption(f"历史数据记忆库存档目录（当前 {len(dm.list_datasets())} 份，可在上一栏剔除）")

with _slot_ver.container():
    if VERSION_JSON.exists():
        try:
            vrows = json.loads(VERSION_JSON.read_text(encoding="utf-8"))
            st.dataframe(pd.DataFrame(vrows), width="stretch", height=260, hide_index=True)
        except Exception:
            st.caption("版本对比表读取失败")
    else:
        st.caption("版本对比数据待生成（v1–v4 在 FC2 上二次测试后写入 模型库/版本对比.json）")
    _mi = MODEL_INFO or {}
    _mname = _mi.get("name") or "v1 GRU（30,083 参数）"
    _msrc = _mi.get("datasets") or "FC1"
    st.caption(f"默认模型：{_mname}（{_msrc} 联合训练，{_mi.get('n_params', 30083):,} 参数）。"
               "交付层 RUL 由物理层 g(t) 决定，四骨干结果一致；本模型用于一步自检与巡检，"
               "v1 GRU 保留为兜底。")
