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
import model_warm as mw
import nn_library as nl
import resource_guard as rg
import rul_pipeline as rp

try:
    import shell_patch as _shell_patch

    _shell_patch.ensure()
except Exception:
    pass

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Noto Sans CJK JP",
    "WenQuanYi Zen Hei", "Droid Sans Fallback",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "输出"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DIAG_LOG = HERE / "终端诊断.log"

def log_exc(tag: str, ex: BaseException) -> str:
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

_PANEL_CSS = """
<style>
/* 「其余数据」浮层面板：固定在视口右侧、盖在所有内容之上，不占侧栏宽度、不推挤主区 */
.st-key-dsh_other_panel{
  position:fixed; right:20px; top:84px; width:min(460px,42vw);
  max-height:78vh; overflow:auto;
  z-index:999900; background:var(--background-color,#0e1117);
  border:1px solid rgba(160,170,185,.28); border-radius:14px;
  box-shadow:0 18px 48px rgba(0,0,0,.45);
  padding:14px 16px 16px 16px;
}
/* 侧栏的收起/展开按钮常显（默认只在鼠标移到侧栏上时才出现） */
[data-testid="stSidebarCollapseButton"]{opacity:1 !important;}
[data-testid="stSidebarCollapsedControl"]{opacity:1 !important;}
/* 上传区的「Browse files」按钮改成红色：这是评委会第一个点的按钮，要醒目 */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]{
  background-color:#e5484d !important; border:1px solid #d93a3f !important;
  color:#ffffff !important; font-weight:700 !important;
  box-shadow:0 2px 10px rgba(229,72,77,.28) !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]:hover{
  background-color:#f0595e !important; border-color:#e5484d !important; color:#fff !important;
}
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span{color:#ffffff !important;}
</style>
"""
st.markdown(_PANEL_CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def _model():
    return rp.load_model()

MODEL, MODEL_INFO = None, {"ok": False, "lazy": True}

def _get_model():
    global MODEL, MODEL_INFO
    if MODEL is not None:
        return MODEL
    stt = mw.get()
    if stt.get("model") is not None:
        MODEL, MODEL_INFO = stt["model"], (stt.get("info") or {"ok": True, "source": "warm"})
        return MODEL
    if stt.get("started") and not stt.get("done"):
        if mw.wait(timeout=60.0) and mw.get().get("model") is not None:
            MODEL, MODEL_INFO = mw.get()["model"], (mw.get().get("info") or {"ok": True, "source": "warm"})
            return MODEL
    if MODEL is None and not MODEL_INFO.get("tried"):
        MODEL_INFO = {"ok": False, "tried": True}
        try:
            MODEL, MODEL_INFO = _model()
        except Exception as ex:
            MODEL_INFO = {"ok": False, "tried": True, "error": str(ex)}
    return MODEL

try:
    mw.warm(rp.load_model)
except Exception as _ex:
    log_exc("模型预热", _ex)

def _overlay_html(text: str = "加载中", detail: str = "", handover: bool = False) -> str:
    _sub = f'<div class="dsh-sub">{detail}</div>' if detail else ""

    _cls = "dsh-hand" if handover else ""
    _hn = ("#dsh-dim.dsh-hand{pointer-events:auto;animation:dshfade .16s ease,"
           "dshfail .8s linear 240s forwards}"
           "@keyframes dshfail{to{opacity:0;visibility:hidden}}") if handover else ""

    _heavy = ("分析" in text) or ("画布" in text)
    _extra = ("" if not _heavy else
              '<div class="dsh-eta">预计耗时两分钟</div>'
              '<div class="dsh-zidan">让子弹飞一会'
              '<small>坐着等就好 · 遮罩消失前请勿连点</small></div>')
    return (
        "<style>"
        "#dsh-dim{position:fixed;inset:0;z-index:999990;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;pointer-events:none;"
        "background:rgba(8,14,24,.58);contain:strict;animation:dshfade .16s ease}"
        "#dsh-dim .dsh-box{background:rgba(22,30,44,.94);color:#eaf2ff;padding:16px 24px;"
        "border-radius:14px;font-size:15px;letter-spacing:.6px;display:flex;align-items:center;"
        "gap:8px;box-shadow:0 10px 34px rgba(0,0,0,.4);font-family:system-ui,sans-serif;"
        "max-width:min(600px,88vw)}"
        "#dsh-dim .dsh-ring{width:16px;height:16px;border-radius:50%;flex:0 0 auto;"
        "border:2px solid rgba(111,183,255,.28);border-top-color:#6fb7ff;"
        "animation:dshspin .9s linear infinite;will-change:transform}"
        "#dsh-dim .dsh-sub{color:#b9c9dd;font-size:12.5px;line-height:1.75;margin-top:12px;"
        "text-align:center;max-width:min(600px,88vw);white-space:pre-line}"
        "#dsh-dim .dsh-dot{display:inline-block;width:7px;height:7px;border-radius:50%;"
        "background:#6fb7ff;animation:dshb 1.2s infinite;will-change:transform,opacity}"
        "#dsh-dim .dsh-dot:nth-of-type(1){animation-delay:0s}"
        "#dsh-dim .dsh-dot:nth-of-type(2){animation-delay:.15s}"
        "#dsh-dim .dsh-dot:nth-of-type(3){animation-delay:.3s}"
        "#dsh-dim .dsh-bar{position:relative;width:min(360px,60vw);height:5px;border-radius:4px;"
        "margin-top:14px;background:rgba(255,255,255,.16);overflow:hidden}"
        "#dsh-dim .dsh-bar i{position:absolute;inset:0;width:38%;border-radius:4px;"
        "background:linear-gradient(90deg,rgba(111,183,255,0),#6fb7ff,rgba(111,183,255,0));"
        "will-change:transform;animation:dshslide 1.25s ease-in-out infinite}"
        "@keyframes dshspin{to{transform:rotate(360deg)}}"
        "@keyframes dshslide{0%{transform:translateX(-120%)}100%{transform:translateX(280%)}}"
        "@keyframes dshb{0%,80%,100%{opacity:.25;transform:translateY(0)}"
        "40%{opacity:1;transform:translateY(-2px)}}"
        "@keyframes dshfade{from{opacity:0}to{opacity:1}}"
        "#dsh-dim .dsh-eta{position:absolute;left:0;right:0;bottom:31vh;text-align:center;"
        "font-size:13.5px;letter-spacing:3px;color:#f0c674;font-weight:600;opacity:.95}"
        "#dsh-dim .dsh-zidan{position:absolute;left:0;right:0;bottom:23vh;text-align:center;"
        "font-size:26px;letter-spacing:7px;font-weight:700;color:#eaf2ff;opacity:.92;"
        "text-shadow:0 2px 12px rgba(0,0,0,.6)}"
        "#dsh-dim .dsh-zidan small{display:block;margin-top:10px;font-size:12px;"
        "letter-spacing:2px;color:#9fb3cc;font-weight:400;text-shadow:none}"
        + _hn +
        "</style>"
        f'<div id="dsh-dim" class="{_cls}"><div class="dsh-box"><span class="dsh-ring"></span>'
        f'<span>{text}</span><span class="dsh-dot"></span><span class="dsh-dot"></span>'
        f'<span class="dsh-dot"></span></div>{_sub}'
        '<div class="dsh-bar"><i></i></div>'
        + _extra + '</div>'
    )

_OFF_HTML = ('<div id="dsh-dim-off" style="display:none;height:0;width:0;'
             'margin:0;padding:0;overflow:hidden"></div>')

def _show_loading(slot, text: str = "加载中", detail: str = "",
                  handover: bool = False) -> None:
    try:
        slot.markdown(_overlay_html(text, detail, handover=handover),
                      unsafe_allow_html=True)
    except Exception:
        pass

def _clear_loading(slot) -> None:
    try:
        slot.markdown(
            _OFF_HTML, unsafe_allow_html=True)
    except Exception:
        pass

def _png(fig: plt.Figure, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()

@st.cache_data(show_spinner=False, max_entries=8)
def _parse_blob(blob: bytes):
    return rp.read_any(blob)

def _friendly_err(ex: BaseException) -> str:
    raw = str(ex).strip()
    low = raw.lower()
    if isinstance(ex, UnicodeDecodeError) or "codec" in low or "decode" in low or "utf-8" in low:
        return ("文件编码无法识别（可能是不支持的编码或二进制文件）。"
                "请用 Excel 打开后「另存为 CSV UTF-8」，或直接上传 .xlsx。")
    if "empty" in low or "no columns" in low or "tokenizing" in low or "0 rows" in low:
        return "文件里没有可读的表格内容（空文件或只有表头）。请确认上传的是完整的数据文件。"
    if "excel" in low and "openpyxl" in low:
        return "当前环境缺少 Excel 解析组件（openpyxl），请改传 CSV/TSV。"
    if "至少 200 行" in raw or "有效数据太短" in raw:
        return raw + "（数据点太少，无法做衰减律与概率分析）。"
    if "时间跨度不足" in raw or "数据太短" in raw or "标定窗口" in raw:
        return raw + "（当前版本要求至少约 50 h 的完整老化曲线；更短的数据只能做原始数据预览）。"
    if "无法解析成表格" in raw or "数值列不足" in raw or "没找到" in raw:
        return raw
    if raw and not any("\u4e00" <= ch <= "\u9fff" for ch in raw):
        return ("文件格式无法识别（缺少时间列或电压列）。请上传含「时间 / 电压」两列的 CSV/TSV/Excel，"
                f"或展开「🔧 列名识别不对？手动指定」自行指定。原始错误：{raw[:160]}")
    return raw or "未知错误。"

def _reset_session() -> None:
    try:
        mw.reset()
    except Exception:
        pass
    try:
        st.cache_data.clear()
    except Exception:
        pass
    for _k in list(st.session_state.keys()):
        try:
            del st.session_state[_k]
        except Exception:
            pass

@st.cache_data(show_spinner=False, max_entries=8)
def _load_demo(blob: bytes, time_unit: str = "auto"):
    return rp.load_csv(blob, time_unit=time_unit)

@st.cache_data(show_spinner=False, max_entries=6)
def _load_many(blobs: tuple, time_unit: str = "auto"):
    return rp.load_csv_many(list(blobs), time_unit=time_unit)

def _fig_prediction(prep, g, mc, ft: float, uth: float) -> plt.Figure:
    t0 = prep["t_end"]
    u0 = prep["u_anchor"]

    _band = mc.get("band") or {}
    _b50 = _band.get(50) if isinstance(_band, dict) else None
    det = np.asarray(_b50 if _b50 is not None else mc["det_curve"], float)
    t_ext = t0 + np.arange(len(det)) * rp.WIN_H

    rr = rp.rul_for_threshold(np.asarray(mc["U"], float), uth=uth, u0=u0)

    fig, ax = plt.subplots(figsize=(11.2, 4.3))
    t, u = prep["t"], prep["u_s"]
    lo = max(0.0, t0 - 700.0)
    m = t >= lo
    _step = max(1, int(m.sum() // 1200))
    ax.plot(t[m][::_step], u[m][::_step], color="#2c6fbb", lw=1.1,
            label=f"历史电压（10 h 平滑，0–{t0:.0f} h）")
    ax.plot(t_ext, det, color="k", lw=2.4, label="中位交付曲线（P50 路径）")
    ax.axhline(uth, color="#c0392b", lw=1.4, ls="--",
               label=f"FT={ft:.1f}% 阈值 = {uth:.4f} V")
    ax.axvline(t0, color="#7f8c8d", lw=1.0, ls=":")
    ax.text(t0, u0, " 预测起点", color="#7f8c8d", fontsize=9, va="bottom")

    valid = rr[np.isfinite(rr)]
    right = max(t0 + (float(np.percentile(valid, 95)) * 1.35 + 100.0) if len(valid) else t0 + 500.0,
                t0 + 300.0)
    right = min(right, t_ext[-1])
    if len(valid) > 0:

        _idx = np.where(det <= uth)[0]
        _xc = None
        if len(_idx):
            _i = int(_idx[0])
            if _i == 0:
                _xc = 0.0
            else:
                _up, _un = float(det[_i - 1]), float(det[_i])
                _xc = float((_i - 1 + (_up - uth) / max(_up - _un, 1e-12)) * rp.WIN_H)
        if _xc is not None and (t0 + _xc) <= right:
            ax.axvline(t0 + _xc, color="#c0392b", lw=1.2, ls="--", alpha=0.85)

            ax.annotate(f"P50 穿越 ≈ {_xc:.0f} h", xy=(t0 + _xc, uth), xytext=(9, -17),
                        textcoords="offset points", color="#c0392b", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))

    ax.set_xlim(lo, right)
    seg = (t_ext >= lo) & (t_ext <= right)
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

def run_analysis(df: pd.DataFrame, name: str, use_history: bool, hist_ids: list[str]) -> dict:
    prep = rp.prepare(df)
    g = rp.fit_decay(prep)
    cal = rp.calibrate(prep, g)
    fp = dm.fingerprint(prep["n_rows"], prep["t_end"])

    _i_cur = prep.get("I_ref") if prep.get("has_I") else None
    prof, n_sets = (dm.pooled_profile(cal, hist_ids, rp.TAUS, exclude_fp=fp,
                                      current_i_ref=_i_cur) if use_history
                    else (np.asarray(cal["prof"], float), 1))
    cal_use = dict(cal, prof=prof)
    cond = rp.condition_profile(prep)
    mc = rp.run_monte_carlo(prep, g, cal_use)
    _m = _get_model()
    st.session_state["model_loaded"] = _m is not None
    ai = rp.one_step_selfcheck(_m, prep) if _m is not None else None
    quality = dm.quality_check(prep)

    _bad_cal = abs(float(cal["d_p95"])) > 20.0 or abs(float(cal["d_p5"])) > 20.0
    if _bad_cal:
        quality = quality + ["标定乘子分布异常（d_p5=%.1f, d_p95=%.1f）：本档未写入历史记忆库，避免污染概率池"
                             % (float(cal["d_p5"]), float(cal["d_p95"]))]
        ds_id = None
    else:
        ds_id = dm.save_dataset(name, prep, g, cal, extra=dict(quality="; ".join(quality)))

    nn = nl.archive(name, prep, extra=dict(quality="; ".join(quality), source="终端预测"))
    return dict(ok=True, name=name, prep=prep, g=g, cal=cal, cal_use=cal_use,
                mc=mc, ai=ai, quality=quality, ds_id=ds_id, nn=nn, cond=cond,
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

def _autosave_previous(key: str) -> None:
    try:
        r = st.session_state.get(key)
        if not isinstance(r, dict) or not r.get("ok") or r.get("saved"):
            return
        _ft = float(r.get("ft_eff", 4.5))
        _uth = float(r.get("uth_eff", np.nan))
        _rr = r.get("rr_eff")
        if _rr is None:
            return
        _png1 = _png(_fig_prediction(r["prep"], r["g"], r["mc"], _ft, _uth))
        _png2 = _png(_fig_hist(_rr, _ft, _uth))
        r["saved"] = str(_save_outputs(_png1, _png2, r, _ft, _uth, _rr))
        rg.log_event("覆盖前自动留档", note=Path(r["saved"]).name)
    except Exception as ex:
        log_exc("覆盖前自动留档", ex)

blob, data_name = None, ""
df_input, parse_err = None, None
blobs = []
_tu = "auto"
_overlay_slot = st.empty()

with st.sidebar:
    st.header("⚙️ 设置")
    if st.button("🔄 刷新（回到初始界面）", key="dsh_refresh", width="stretch",
                 help="一键回到初始界面：清空本次会话的上传文件、预测结果、画布缓存与历史选择"
                      "（磁盘上的数据记忆库/神经网络库/输出目录不受影响）"):
        _reset_session()
        st.rerun()
    src = st.radio("数据来源", ["上传数据", "内置演示数据"], horizontal=True,
                   help="默认不加载任何数据；内置演示数据会占用数十 MB 内存，按需再选。")

    _tu_label = st.selectbox(
        "时间列单位", [c[0] for c in rp.TIME_UNIT_CHOICES], index=0,
        help="自动识别失败时手动指定（同济动态工况数据的时间列是“秒”）。"
             "旧数据/演示数据一律按小时。")
    _tu = dict(rp.TIME_UNIT_CHOICES)[_tu_label]

    if src == "内置演示数据":
        if not rp.DEMO_LABELS:
            parse_err = "当前环境未内置演示数据，请改用「上传数据」。"
        else:
            label = st.selectbox("选择演示数据", rp.DEMO_LABELS, index=0)
            blob = rp.demo_blob(label)
            data_name = label.split(" ·")[0] + "_" + label.split("(")[0].split("·")[-1].strip()
            if blob is None:
                parse_err = "演示数据文件不存在，请改用「上传数据」。"
            else:
                _show_loading(_overlay_slot,
                              f"正在读取演示数据（{len(blob) / 1048576:.1f} MB，请稍候）")
                try:
                    df_input = _load_demo(blob, _tu)
                except Exception as ex:
                    parse_err = f"演示数据读取失败：{_friendly_err(ex)}"
                finally:
                    _clear_loading(_overlay_slot)
    else:
        ups = st.file_uploader(
            "上传电池老化数据（CSV / TSV / Excel，可多选）",
            type=["csv", "tsv", "txt", "xlsx", "xls"], accept_multiple_files=True,
            help="默认「覆盖」模式：每次只分析一份完整曲线，新上传的文件会自动顶替上一份；"
                 "同一块电池的分段文件请切到「拼接」模式。")
        blobs_all = [u.getvalue() for u in ups] if ups else []
        _mode_label = st.radio(
            "多文件怎么处理",
            ["覆盖：每次只分析一份完整曲线（推荐）", "拼接：同一电池的分段文件接成一条"],
            index=0,
            help="覆盖（默认）：不会把新文件接在上一份数据的尾巴上——每次分析都从头开始。\n"
                 "拼接（高级）：仅当这些文件确实来自同一块电池、且时间段首尾相接时才用；"
                 "不同电池拼在一起会让衰减律、RUL 与指标全部失真（这正是以前崩溃的来源）。")
        _merge_mode = str(_mode_label).startswith("拼接")
        if _merge_mode and len(blobs_all) > 1:
            blobs = blobs_all
            blob = blobs[0]
            data_name = Path(ups[0].name).stem + f" 等{len(blobs)}个文件"
            st.caption("拼接模式：" + "、".join(u.name for u in ups[:4])
                       + ("…" if len(ups) > 4 else ""))
            _show_loading(_overlay_slot,
                          f"正在读取并拼接 {len(blobs)} 个文件（按时间接续，请确认是同一电池的分段）")
            try:
                df_input = _load_many(tuple(blobs), _tu)
                st.session_state["map_sig"] = f"multi{len(blobs)}"
                st.caption(f"已拼接 {df_input.attrs.get('merged_files', len(blobs))} 个文件 → "
                           f"共 {len(df_input):,} 行，时间 0 → "
                           f"{float(df_input['Time (h)'].iloc[-1]):.1f} h"
                           f"（仅适用于同一块电池的分段文件）")
            except Exception as ex:
                parse_err = _friendly_err(ex)
            finally:
                _clear_loading(_overlay_slot)
        elif blobs_all:

            _pick_i = 0
            if not _merge_mode:
                _n_ups = len(blobs_all)
                _seen = int(st.session_state.get("__cover_seen_n", 0) or 0)
                _pick_i = st.session_state.get("cover_pick", _n_ups - 1)
                if (not isinstance(_pick_i, int)) or _pick_i < 0 or _pick_i >= _n_ups or _n_ups > _seen:
                    _pick_i = _n_ups - 1
                st.session_state["cover_pick"] = _pick_i
                st.session_state["__cover_seen_n"] = _n_ups
                if _n_ups > 1:

                    _pick_i = st.selectbox(
                        "本次分析的数据文件（覆盖模式）", list(range(_n_ups)), index=_pick_i,
                        format_func=lambda i: f"{ups[i].name}（{len(blobs_all[i]) / 1024:.0f} KB）",
                        key="cover_pick",
                        help="覆盖模式下每次只分析一份完整曲线；新上传的文件会自动切到最新一份，"
                             "上一份不会被接在它后面。")
                blobs = [blobs_all[_pick_i]]
            else:
                blobs = blobs_all
            blob = blobs_all[_pick_i]
            up = ups[_pick_i]
            data_name = Path(up.name).stem
            st.caption(f"文件：{up.name}（{len(blob) / 1024:.0f} KB）"
                       + ("" if _merge_mode else " · 覆盖模式：本次只分析这一份"))
            _show_loading(_overlay_slot,
                          f"正在读取并解析数据（{len(blob) / 1048576:.1f} MB，自动识别编码/列名）")
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
            except Exception as ex:
                parse_err = _friendly_err(ex)
            finally:
                _clear_loading(_overlay_slot)
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
        st.caption(f"ℹ️ 已按时间拼接 {df_input.attrs['merged_files']} 段数据"
                   f"（拼接模式：请确认它们确实来自同一块电池）。")

    import hashlib as _hl
    _sig_src = b"".join(blobs) if blobs else (blob or b"")
    _data_sig = (f"{src}|{_tu}|{len(blobs)}|"
                 f"{_hl.md5(_sig_src).hexdigest()[:12] if _sig_src else 'none'}")
    if df_input is not None and st.session_state.get("__data_sig") != _data_sig:
        st.session_state["__data_sig"] = _data_sig

        for _k in [k for k in list(st.session_state.keys()) if str(k).startswith("res_")]:
            _autosave_previous(_k)
            st.session_state.pop(_k, None)
        rg.note_op("数据解析")
        rg.log_event("数据就绪", note=_data_sig)
        rg.release_hold(_overlay_slot, "数据已解析，正在释放内存")

    st.divider()
    with st.expander("🎚️ 指标与保存阈值（图内联动不经过这里）", expanded=False):

        with st.form("ft_form", clear_on_submit=False):
            ft = st.slider("记录阈值 FT（% 初始功率）", 3.0, 8.0, 4.5, 0.1,
                           help="只决定主界面指标行与「保存」的内容；拖完点下面「应用」才生效。"
                                "日常看数值请直接拖**画布内**滑条（纯浏览器本地联动、不触发服务器计算）")
            st.form_submit_button("应用该阈值（更新指标行 / 保存口径）")
        st.caption("提示：拖动这里的滑条**不会**立刻刷新页面（避免反复重绘导致卡顿），"
                   "点「应用」后才更新指标行与保存。")
    st.caption("⏱️ 高危操作（上传文件 / 开始预测）会启动加载闸门："
               "等资源负载降到红线的 20% 才解除，请勿连续快速点击。")
    map_sig = st.session_state.get("map_sig", "") if src == "上传数据" else ""
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

    with st.expander("🧠 神经网络库（训练数据仓库）", expanded=False):
        _nn = nl.stats()
        st.caption(f"共 {_nn['n']} 份 · {_nn['size_text']}；每次预测成功都会把本次**完整序列**"
                   f"打包入库，供模型继续学习成长。目录：`{Path(nl.NN_DIR).name}\\`")
        if _nn["n"]:
            for _p in nl.list_packs()[:8]:
                st.markdown(
                    f'**{str(_p.get("name", ""))[:20]}**  \n'
                    f'`{str(_p.get("created_at", ""))[5:16].replace("T", " ")}` · '
                    f'{float(_p.get("t_end", 0)):.0f} h · 指纹 `{_p.get("key", "")}`')
        else:
            st.caption("暂无数据：第一次预测成功后会自动入库。")
        st.caption("去重规则：按「时间网格 + 电压序列」的内容指纹查重，**同一份数据只入一次**"
                   "（一份数据只看一次，避免反复学习同一份数据、加深模型偏见）。")

    with st.expander("📊 其余数据", expanded=False):

        st.caption("点一下按钮，右侧浮出对应面板（盖在页面之上，不挤占布局）；再点一次即关闭。")
        _other_sel = st.segmented_control(
            "其余数据面板", ["📋 概览", "📐 衰减律", "📊 阈值表", "🧭 工况"],
            default=None, key="other_panel_sel", label_visibility="collapsed",
            width="stretch")
    with st.expander("🌡️ 泛化数据（不确定性扇形带）", expanded=False):
        _slot_band = st.container()
    with st.expander("📈 历史趋势对比（学习记忆）", expanded=False):
        _slot_trend = st.container()
    with st.expander("📁 图表与数据文件路径", expanded=False):
        _slot_paths = st.container()
    with st.expander("ℹ️ 模型版本对比", expanded=False):
        _slot_ver = st.container()
    with st.expander("🩺 资源监控（红线矩阵 / 加载闸门）", expanded=False):
        _rs = rg.sample()
        st.caption(rg.summary_text(_rs))

        _mrows = rg.matrix_rows(_rs)
        _hdr = list(_mrows[0].keys())
        _lines = ["| " + " | ".join(_hdr) + " |",
                  "|" + "|".join(["---"] * len(_hdr)) + "|"]
        _lines += ["| " + " | ".join(str(r.get(h, "")) for h in _hdr) + " |" for r in _mrows]
        st.markdown("\n".join(_lines))
        st.caption(f"规则：负载分 ≥ 1.0 = 越过红线；高危操作结束后需降到 "
                   f"{rg.RELEASE_RATIO:.0%}（即 {rg.RELEASE_RATIO:.2f}）才解除「加载中」，"
                   f"最长等待约 3.5 s。记录见 `终端资源.log`。")

hist_ids = chosen

_n_hist = len(dm.list_datasets())
_minfo = MODEL_INFO if MODEL_INFO.get("ok") else None
_mname = (_minfo or {}).get("name") or "v4 GRU-Transformer（首次预测时加载）"
_npar = (_minfo or {}).get("n_params")
_mtxt = f"{_mname}（{_npar:,} 参数）" if _npar else _mname

_hero_l, _hero_r = st.columns([5, 1.15])
with _hero_l:
    st.title("🔋 质子交换膜燃料电池（PEMFC）剩余寿命预测")
    st.caption("物理约束指数衰减律 · 分位数回归概率层 · 盲测自回归 ｜ "
               f"模型：{_mtxt} ｜ 数据记忆库：{_n_hist} 份历史")
with _hero_r:
    st.write("")
    with st.popover("👥 制作组简介", width="stretch"):
        st.markdown("### 制作组简介")

        st.markdown("毛炯 黄胤淦 蒋帅 朱其言",
                    unsafe_allow_html=True)
        st.caption("西双banana毒麻蛇")

    _res_label = st.selectbox(
        "画布分辨率",
        ["自动（推荐）", "高清 2.0×", "标准 1.5×", "省内存 1.0×", "极省 0.75×"],
        index=0, key="render_scale_label",
        help="只影响交互画布画多少像素，不影响预测结果与指标。"
             "浏览器吃力、风扇狂转或小蓝点卡住时，选「省内存」或「极省」。")
_res_scale = float({"自动（推荐）": -1.0, "高清 2.0×": 2.0, "标准 1.5×": 1.5,
                    "省内存 1.0×": 1.0, "极省 0.75×": 0.75}[_res_label])

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

cache_key = f"res_{key}_{map_sig}_{_tu}_{bool(use_history)}"

res = st.session_state.get(cache_key)

if go and df_input is not None:
    if not rg.cooldown_ok("predict", seconds=2.5):
        st.warning("操作太快啦：上一次预测刚结束，等加载完全解除后再点。", icon="⏳")
    else:
        rg.note_op("开始预测")
        _show_loading(
            _overlay_slot,
            "正在分析：拟合衰减律 → 分位数标定 → 蒙特卡洛 4000 路径 → 加载 AI 模型",
            "首次运行需要加载 AI 模型（约 2–3 s）；数据越大（如 20 MB 级）整体约 8–15 s。\n"
            "遮罩动画在浏览器本地循环，即使页面短暂无响应也会继续转动——请耐心等它消失，不要连点。")
        try:
            st.session_state[cache_key] = run_analysis(df_input, data_name, use_history, hist_ids)

            for _k in [k for k in list(st.session_state.keys())
                       if str(k).startswith("res_") and k != cache_key]:
                _autosave_previous(_k)
                st.session_state.pop(_k, None)
        except Exception as ex:
            log_exc("点击预测", ex)
            _nerr = int(st.session_state.get("__err_n", 0) or 0) + 1
            st.session_state["__err_n"] = _nerr
            st.error(f"分析失败：{_friendly_err(ex)}")
            with st.expander("查看错误详情（已写入 终端诊断.log）", expanded=False):
                st.code("见 终端诊断.log")

            if _nerr >= 2:
                st.warning(f"本次会话已连续失败 {_nerr} 次。建议一键刷新回到初始界面再试，"
                           f"不要在同一份坏数据上反复点。", icon="🧹")
            if st.button("🔄 一键刷新（清空会话，回到初始界面）", key="dsh_refresh_err"):
                _reset_session()
                st.rerun()
        finally:

            rg.release_hold(_overlay_slot, "分析完成，正在释放资源",
                            keep=st.session_state.get(cache_key) is not None)
        res = st.session_state.get(cache_key)

if res is None:
    if df_input is None and not parse_err:
        st.info("👈 先在左侧**上传数据**（或选择内置演示数据，可选），然后点「🔮 开始预测」。")
        st.caption("提示：默认不加载任何数据、不占用内存；内置演示数据体积较大，按需再选。")
    elif df_input is not None and not parse_err:
        st.info("数据已就绪 👉 点上方「🔮 开始预测」开始分析（约 3–10 秒，期间屏幕会变暗显示加载状态）。")
    st.stop()

prep, g, mc = res["prep"], res["g"], res["mc"]
_u_eq = (prep.get("P_init", 0.0) / prep["I_ref"]) if (prep.get("has_I") and prep.get("I_ref")) \
    else float(prep.get("u_start", prep["u_anchor"]))
_cur_pct = 100.0 * (1.0 - prep["u_anchor"] / _u_eq) if _u_eq > 0 else 0.0
_sug_ft = min(8.0, max(3.0, np.ceil((_cur_pct + 0.6) * 10.0) / 10.0))
PROJECT_FT_INVALID_MARK = True

def _ft_already_failed(prep: dict, uth: float, valid) -> bool:
    u0 = float(prep["u_anchor"])
    if float(uth) >= u0 - 1e-12:
        return True
    if len(valid) == 0:
        return True
    return bool(float(np.max(valid)) <= 1e-9)

uth = rp.uth_for_ft(prep, ft)
rr = rp.rul_for_threshold(np.asarray(mc["U"], float), uth=uth, u0=prep["u_anchor"])
valid = rr[np.isfinite(rr)]

ft_eff, uth_eff, rr_eff, valid_eff, _auto = ft, uth, rr, valid, False
_failed = _ft_already_failed(prep, uth, valid)
if _failed and _sug_ft > ft + 1e-9:
    ft_eff = float(_sug_ft)
    uth_eff = rp.uth_for_ft(prep, ft_eff)
    rr_eff = rp.rul_for_threshold(np.asarray(mc["U"], float), uth=uth_eff, u0=prep["u_anchor"])
    valid_eff = rr_eff[np.isfinite(rr_eff)]
    _auto = True

_no_path = len(valid_eff) == 0

res["ft_eff"], res["uth_eff"], res["rr_eff"] = ft_eff, uth_eff, rr_eff

if _failed and not _auto:
    st.warning(
        f"FT={ft:.1f}%：该电池在当前已低于此失效阈值，RUL = 0 h（已失效）。"
        f"当前已衰减约 {_cur_pct:.2f}%，建议把 FT 调到 **{_sug_ft:.1f}%** 以上再看预测。",
        icon="⚠️")
    c1, c2, c3 = st.columns(3)
    c1.metric("P50 剩余寿命", "0 h")
    c2.metric("P5", "0 h")
    c3.metric("P95", "0 h")
elif _no_path:
    st.warning(
        f"FT={ft:.1f}%（自动建议 {_sug_ft:.1f}%）下**有效路径 0/4000**：所有蒙特卡洛路径都穿不过阈值，"
        f"无法给出 RUL。当前末端电压 {prep['u_anchor']:.4f} V、起始 {prep['u_start']:.4f} V，"
        f"衰减 {_cur_pct:+.2f}%。\n\n"
        "最常见的原因是**把不同电池/不同分段的数据拼成了一条曲线**（末端电压反而升高），"
        "或者是时间列/电压列识别错误。请用「覆盖」模式只分析一份完整曲线，"
        "或在「🔧 列名识别不对？」里检查列映射。",
        icon="🛑")
    c1, c2, c3 = st.columns(3)
    c1.metric("P50 剩余寿命", "— h")
    c2.metric("P5", "— h")
    c3.metric("P95", "— h")
    st.caption("0 条有效路径 —— 不是「算不出来」，而是这份数据里没有任何一条路径能在 "
               f"FT={ft:.1f}% 的阈值下存活。数据修好后再点一次预测即可。")
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

_u_eq_full = (prep.get("P_init", 0.0) / prep["I_ref"]) if (prep.get("has_I") and prep.get("I_ref")) \
    else float(prep.get("u_start", prep["u_anchor"]))
_live_key = f"{cache_key}|{res.get('fp', 'x')}|rs{_res_scale}"
_live_cache = st.session_state.setdefault("live_cache", {})
try:
    _theme = lc.pick_theme(getattr(st.context, "theme", None))
except Exception:
    _theme = lc.pick_theme(None)

if _live_key not in _live_cache:
    if _no_path:

        _live_cache[_live_key] = dict(live=None, html=None)
    else:

        _show_loading(
            _overlay_slot,
            "正在准备交互画布（打包 FT 网格与 RUL 直方图）",
            "打包完成后同一个遮罩会继续盖着，等两张图的每一档 FT 都渲染好、"
            "图片准备率到 90% 以上才揭开。请勿连点。",
            handover=True)
        try:
            _live = lc.build_live_data(prep, g, mc, u_eq=_u_eq_full)
            _ft_start = float(np.clip(float(ft), _live["ft"][0], _live["ft"][-1]))
            _live_cache.clear()
            _live_cache[_live_key] = dict(
                live=_live,
                html=lc.build_html(_live, ft_default=_ft_start, theme=_theme,
                                   render_scale=_res_scale),
            )
        except Exception as ex:
            log_exc("构建交互画布", ex)
            st.error(f"交互画布生成失败：{ex}")
            _live_cache[_live_key] = dict(live=None, html=None)
        finally:

            if _live_cache.get(_live_key, {}).get("html") is None:
                _clear_loading(_overlay_slot)
else:

    _clear_loading(_overlay_slot)
_live_entry = _live_cache.get(_live_key, dict(live=None, html=None))
if _live_entry.get("html"):
    components.html(_live_entry["html"], height=lc.component_height(), scrolling=False)
    st.caption("整页现在只有**一层**加载遮罩（`#dsh-dim`）：点预测时盖上，分析完不揭，接着盖住交互画布"
               "的预加载 —— 浏览器先在最上层把每一档 FT 的两张图真实渲染一遍并压成 1/4 缩略图缓存，"
               "图片准备率到 90% 以上才一次揭开，所以看到的不是「半成品画布」。拖动**画布内 FT 滑条**时，"
               "图 1 的红线/P50 即时联动，图 2 柱状图先贴缩略图（跟手），松手 110 ms 后自动补全分辨率；"
               "长时间连续拖动仍跟不上会短暂灰屏降级 —— 全部在浏览器本地完成，不触发服务器重算。"
               f"上方指标行与「保存」按左侧记录阈值 FT={ft_eff:.1f}% 计算。"
               f"当前画布分辨率：**{_res_label}**（只影响像素密度，不影响计算；浏览器吃力时调低）。")
elif _no_path:
    st.info("本次没有一条有效路径（RUL 全 0），已跳过交互画布。先按上面的提示修正数据"
            "（推荐用「覆盖」模式只分析一份完整曲线）或调整阈值，再点一次预测。")
else:
    st.warning("交互画布未能生成，请重新点「🔮 开始预测」重试（错误已写入 终端诊断.log）。")

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

_c1, _c2 = st.columns([1, 5])
with _c1:
    _save_click = st.button("💾 保存本次结果",
                            help=f"按左侧记录阈值 FT={ft:.1f}% 保存 PNG 与 CSV 到 输出\\ 目录")
if _save_click:
    with st.spinner("正在保存……"):
        _png1 = _png(_fig_prediction(prep, g, mc, ft_eff, uth_eff))
        _png2 = _png(_fig_hist(rr_eff, ft_eff, uth_eff))
        _saved_dir = str(_save_outputs(_png1, _png2, res, ft_eff, uth_eff, rr_eff))
        st.session_state["last_out_dir"] = _saved_dir

        res["saved"] = _saved_dir
out_dir = Path(st.session_state.get("last_out_dir", OUTPUT_ROOT))

if _other_sel:
    with st.container(key="dsh_other_panel"):
        st.caption("浮层面板 · 再点一次侧栏里同一个按钮即可关闭")
        if _other_sel == "📋 概览":
            t, u_s, u_trend = prep["t"], prep["u_s"], prep["u_trend"]
            step = max(1, len(t) // 800)
            st.markdown("**原始数据概览**")
            st.markdown(
                f"**行数**：{prep['n_rows']:,} → 网格 {len(t):,} 点，末点 **{prep['t_end']:.2f} h**  \n"
                f"**末端电压** U_anchor：**{prep['u_anchor']:.4f} V**（起始 {prep['u_start']:.4f} V）  \n"
                f"**电流** I_ref：{prep['I_ref']:.2f} A · **功率** P_init：{prep['P_init']:.1f} W\n\n"
                f"电压曲线（10 h 平滑）：")
            st.line_chart(pd.DataFrame({"电压 U(V)": u_s[::step]}, index=t[::step]), height=170)

        elif _other_sel == "📐 衰减律":
            st.markdown("**衰减律模型**　g(t) = a·e^(−b·t) + c　（mV/h）")
            st.markdown(
                r"$g(t) = %.5f\,e^{-%.6f\,t} + %.5f$ mV/h"
                % (g["a"] * 1e3, g["b"], g["c"] * 1e3))
            st.markdown(
                f"**RMSE** = **{g['rmse']:.4f} mV/h**（{g['n_seg']} 段参与拟合）  \n"
                f"g(t₀) = **{g['g0_mVh']:.5f} mV/h**  \n"
                f"曲率上限 G(t₀) = {g['G'](g['t0']):.3e} V/h²")
            if res["ai"]:
                st.caption(f"AI 一步自检：RMSE={res['ai']['rmse_uV']:.1f} µV/步"
                           f"（增量 σ≈{res['ai']['sig_uV']:.1f} µV）")

        elif _other_sel == "🧭 工况":
            _cond = res.get("cond") or {}
            _sens = dm.condition_sensitivity()
            st.markdown("**运行工况档案**")
            if _cond.get("has_I"):
                st.markdown(
                    f"**{_cond['label']}**  \n"
                    f"电流中位数 **{_cond['i_med']:.2f} A**"
                    f"（范围 {_cond['i_min']:.2f} – {_cond['i_max']:.2f} A）  \n"
                    f"四分位变异系数 IQR/中位数 = **{_cond['i_cv']:.3f}**  \n"
                    f"阈值判据：电流可用 → 按 I_ref 折算寿命阈值")
                if float(_cond["i_cv"]) >= 0.25:
                    st.warning("检测到**变工况/负载波动**：RUL 以「中位电流等效恒流」折算，"
                               "负载波动越大，区间可信度越低。若能把数据按恒流段拆开、"
                               "每段单独分析，结论会稳得多（拼接模式只适合同一块电池的连续分段）。",
                               icon="🧭")
            else:
                st.caption(f"{_cond.get('label', '无电流列')}：阈值直接按电压算。")

            st.markdown("**电流 → 衰减速度 敏感度**")
            if _sens.get("ok"):
                st.markdown(
                    f"幂律拟合 **g₀ ∝ I^α**：α = **{_sens['alpha']:.2f}**"
                    f"（R² = {_sens['r2']:.2f}）  \n"
                    f"来源：「{_sens['family']}」家族的 {_sens['n']} 档，"
                    f"电流 {_sens['i_min']:.1f} – {_sens['i_max']:.1f} A  \n"
                    f"含义：电流翻倍，初始衰减速度约变 **{2 ** _sens['alpha']:.2f} 倍**")
                _i_cur = _cond.get("i_med")
                if _i_cur and (_i_cur < _sens["i_min"] * 0.8 or _i_cur > _sens["i_max"] * 1.25):
                    st.info(f"当前电流 {_i_cur:.1f} A 不在已标定区间内：概率层会按对数距离"
                            f"自动降低远工况档的权重，RUL 区间偏宽属正常；"
                            f"把它当「外推」而不是实测结论。", icon="🧭")
            else:
                st.caption(f"有电流信息的历史档不足（同家族 ≥3 档才能拟合 α，"
                           f"当前 {_sens.get('n', 0)} 档），暂不给出。")
            st.caption("概率层合并不是简单平均：每档历史按「电流对数距离」加权"
                       "（exp(−(ln(I_d/I_cur)/0.7)²)），所以同一份数据在不同工况下"
                       "会得到不同的 RUL 区间——这就是模型对环境特征值的敏感度。")

        else:
            st.markdown("**全阈值表**（预设 3.5–6.0%）")

            def _fmt(v, nd):
                return "—" if (v is None or not np.isfinite(v)) else ("%.*f" % (nd, v))

            _head = "".join(f"<th>{h}</th>" for h in
                            ("FT%", "阈值 V", "P5 h", "P50 h", "P95 h", "删失%"))
            _body = []
            for row in mc["stats"]:
                _body.append(
                    "<tr>"
                    f"<td>{_fmt(row['FT'], 1)}</td>"
                    f"<td>{_fmt(row['U_th'], 4)}</td>"
                    f"<td>{_fmt(row['P5'], 0)}</td>"
                    f"<td><b>{_fmt(row['P50'], 0)}</b></td>"
                    f"<td>{_fmt(row['P95'], 0)}</td>"
                    f"<td>{'0' if abs(row['cens']) < 0.5 else _fmt(row['cens'], 0)}</td>"
                    "</tr>")
            st.markdown(
                '<style>.dsh-thr th,.dsh-thr td{white-space:nowrap;padding:3px 6px;}</style>'
                '<table class="dsh-thr" style="width:100%;border-collapse:collapse;'
                'text-align:center;font-size:13px;font-variant-numeric:tabular-nums;line-height:1.6;">'
                '<thead><tr style="color:#8b95a5;font-weight:600;border-bottom:1px solid #3a4250;">'
                f"{_head}</tr></thead><tbody>{''.join(_body)}</tbody></table>",
                unsafe_allow_html=True)

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
    _tr_step = max(1, np.asarray(prep["t"]).size // 600)
    _idx = np.asarray(prep["t"], float)[::_tr_step]
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

            _vh = list(vrows[0].keys())
            _vl = ["| " + " | ".join(_vh) + " |",
                   "|" + "|".join(["---"] * len(_vh)) + "|"]
            _vl += ["| " + " | ".join(str(_r.get(k, "")) for k in _vh) + " |" for _r in vrows]
            st.markdown("\n".join(_vl))
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
