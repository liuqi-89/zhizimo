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
import model_warm as mw
import nn_library as nl
import resource_guard as rg
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

# ---- 2026-11-05 排版（第三版）：侧栏恢复默认宽度，「其余数据」改成微信式浮层面板
# 侧栏宽度不再动（min-width 会破坏折叠按钮的收起动画），6 列表格改为点按钮后浮出在页面之上。
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
</style>
"""
st.markdown(_PANEL_CSS, unsafe_allow_html=True)


# ============================================================ 缓存与模型
@st.cache_resource(show_spinner=False)
def _model():
    return rp.load_model()


MODEL, MODEL_INFO = None, {"ok": False, "lazy": True}


def _get_model():
    """取模型（优先用页面加载时后台预热好的那一个）。

    为什么这么写：本机首次 `import torch` 需 5–10 s（实测），同步放在点预测时会直接
    让浏览器"无响应"。`model_warm` 在页面一打开就后台加载，点预测时基本已就绪；
    若还没好，这里最多等 60 s（等的是同一个加载，不会重复加载）。
    """
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
        except Exception as ex:  # noqa: BLE001
            MODEL_INFO = {"ok": False, "tried": True, "error": str(ex)}
    return MODEL


# 后台预热模型：必须放在 st.stop() 之前（首屏无数据时会 st.stop()，放文件末尾就永远执行不到）。
# 首次 import torch 在本机要 5–10 s，提前在后台线程加载，用户点「开始预测」时基本已就绪。
try:
    mw.warm(rp.load_model)
except Exception as _ex:  # noqa: BLE001
    log_exc("模型预热", _ex)


# ============================================================ 加载遮罩（屏幕变暗 + 加载中）
# 性能要点（2026-11-04 晚，针对"点预测后浏览器像卡死、小蓝点不动"）：
# 1) 不用 backdrop-filter 全屏模糊（集成显卡合成预算会被吃掉，叠加 DOM 更新直接拖死主线程）；
# 2) 动画全部用 opacity/transform（GPU 合成属性），遮罩一出现就自己动，不依赖服务器刷新；
# 3) 遮罩文案 + 详情合并进同一段 HTML，一次写入（原版本每 tick 写两次全屏 DOM）。
def _overlay_html(text: str = "加载中", detail: str = "", handover: bool = False) -> str:
    _sub = f'<div class="dsh-sub">{detail}</div>' if detail else ""
    # handover=True：这层遮罩不是给 Python 阶段用的，而是要交接给 iframe 里的"预加载自检"
    # （同一层 #dsh-dim 一直盖着，直到 23 档 FT 的图都备好才由前端从父页面 DOM 删掉）。
    # 因此它必须挡住整页交互；同时挂一个 240 s 自动隐藏动画兜底，万一前端接不上也不会永久锁死页面。
    _cls = "dsh-hand" if handover else ""
    _hn = ("#dsh-dim.dsh-hand{pointer-events:auto;animation:dshfade .16s ease,"
           "dshfail .8s linear 240s forwards}"
           "@keyframes dshfail{to{opacity:0;visibility:hidden}}") if handover else ""
    # 重活（分析 / 画布打包）才显示"预计耗时两分钟 + 让子弹飞一会"（队员的彩蛋文案后续可轮播）
    # 后续（队员定稿后）：把下面这行换成每 7 秒轮换的彩蛋文案列表 _FUN_LINES，
    # 例如 "真诚建议，别学C语言" / "快试试 deep seek harness 吧" / "燃料电池有可能是氢氧根离子在传导"。
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


# 揭遮罩时写进去的不可见占位块（配合 _clear_loading：绝不能 slot.empty()，
# 否则会在同一次 run 里减少子节点，触发 Streamlit 的幽灵子树 bug）。
_OFF_HTML = ('<div id="dsh-dim-off" style="display:none;height:0;width:0;'
             'margin:0;padding:0;overflow:hidden"></div>')


def _show_loading(slot, text: str = "加载中", detail: str = "",
                  handover: bool = False) -> None:
    try:
        slot.markdown(_overlay_html(text, detail, handover=handover),
                      unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        pass


def _clear_loading(slot) -> None:
    """揭掉遮罩：写入一个不可见占位块，**不要**用 slot.empty()。

    ★ 2026-11-05 事故修复（浏览器红框
      `NotFoundError: Failed to execute 'removeChild' on 'Node'` + 整页卡死）：
      同一个 st.empty() 在一次 script run 内被「先写满 → 再写更少的子节点（含清空）」时，
      Streamlit 的 AppRoot.addBlock 会沿用上一棵子树的引用、而 ClearStaleNodeVisitor
      只按 scriptRunId 清理（上游 issue #12069，1.47+ 复现），于是多出来的幽灵子节点
      带着当前 run id 清不掉。React fiber 认为某个节点还挂在容器里、真实 DOM 里却已经
      没有它，下一次提交在 commitMutationEffects → removeChild 上就抛 NotFoundError，
      错误边界把主区顶成红框、页面卡死。
      这里改成「写一层不可见 div」＝把原来的 1 个子节点换成另 1 个，
      子节点数量永远不减少，React 只走正常的替换路径，不会留下幽灵子树。
    """
    try:
        slot.markdown(
            _OFF_HTML, unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        pass


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
    # 2026-11-05：曲线改画「逐点中位路径」，与交互画布口径一致。
    # det_curve（乘子恒为 1）只是概率分布的下包络，历史增强后它的穿越时间会比中位 RUL
    # 大好几倍，曾导致「红竖线 / 阈值线的交点不在曲线上」。
    _band = mc.get("band") or {}
    _b50 = _band.get(50) if isinstance(_band, dict) else None
    det = np.asarray(_b50 if _b50 is not None else mc["det_curve"], float)
    t_ext = t0 + np.arange(len(det)) * rp.WIN_H

    rr = rp.rul_for_threshold(np.asarray(mc["U"], float), uth=uth, u0=u0)

    fig, ax = plt.subplots(figsize=(11.2, 4.3))
    t, u = prep["t"], prep["u_s"]
    lo = max(0.0, t0 - 700.0)
    m = t >= lo
    _step = max(1, int(m.sum() // 1200))          # 降采样到 ~1200 点，渲染更快
    ax.plot(t[m][::_step], u[m][::_step], color="#2c6fbb", lw=1.1,
            label=f"历史电压（10 h 平滑，0–{t0:.0f} h）")
    ax.plot(t_ext, det, color="k", lw=2.4, label="中位交付曲线（P50 路径）")
    ax.axhline(uth, color="#c0392b", lw=1.4, ls="--",
               label=f"FT={ft:.1f}% 阈值 = {uth:.4f} V")
    ax.axvline(t0, color="#7f8c8d", lw=1.0, ls=":")
    ax.text(t0, u0, " 预测起点", color="#7f8c8d", fontsize=9, va="bottom")

    valid = rr[np.isfinite(rr)]
    if len(valid) > 0:
        p50 = float(np.percentile(valid, 50))
        # 竖线画在「中位曲线与阈值线的交点」上（与交互画布同一口径）；取不到交点才退回统计 P50
        _idx = np.where(det <= uth)[0]
        _xc = None
        if len(_idx):
            _i = int(_idx[0])
            if _i == 0:
                _xc = 0.0
            else:
                _up, _un = float(det[_i - 1]), float(det[_i])
                _xc = float((_i - 1 + (_up - uth) / max(_up - _un, 1e-12)) * rp.WIN_H)
        if _xc is None:
            _xc = p50
        ax.axvline(t0 + _xc, color="#c0392b", lw=1.2, ls="--", alpha=0.85)
        # 标注直接贴着交点写（而不是贴 y 轴底部），保证不会因 ylim 裁剪而看不见
        ax.annotate(f"P50 穿越 ≈ {_xc:.0f} h", xy=(t0 + _xc, uth), xytext=(9, -17),
                    textcoords="offset points", color="#c0392b", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))

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
    # 神经网络库：打包"本次完整序列"供模型继续训练；相同内容指纹只入一次（一份数据只看一次）
    nn = nl.archive(name, prep, extra=dict(quality="; ".join(quality), source="终端预测"))
    return dict(ok=True, name=name, prep=prep, g=g, cal=cal, cal_use=cal_use,
                mc=mc, ai=ai, quality=quality, ds_id=ds_id, nn=nn,
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
    """「覆盖」发生前，把上一份结果自动留档到 输出\\（不让旧数据凭空消失）。

    新数据预测成功时，旧结果（含 4000×300 路径矩阵）会被从 session 里清掉；
    如果用户还没点过「💾 保存本次结果」，这里补一次 PNG + CSV，保证输出目录里
    始终留得住被覆盖的那一份。失败只写诊断日志，绝不打断预测。
    """
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
    except Exception as ex:  # noqa: BLE001
        log_exc("覆盖前自动留档", ex)


# ============================================================ 侧边栏
blob, data_name = None, ""
df_input, parse_err = None, None
blobs = []
_tu = "auto"
_overlay_slot = st.empty()          # 全屏加载遮罩的宿主（fixed 定位，不占版式）

with st.sidebar:
    st.header("⚙️ 设置")
    src = st.radio("数据来源", ["上传数据", "内置演示数据"], horizontal=True,
                   help="默认不加载任何数据；内置演示数据会占用数十 MB 内存，按需再选。")

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
            _show_loading(_overlay_slot,
                          f"正在读取演示数据（{len(blob) / 1048576:.1f} MB，请稍候）")
            try:
                df_input = _load_demo(blob, _tu)
            except Exception as ex:  # noqa: BLE001
                parse_err = f"演示数据读取失败：{ex}"
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
            except Exception as ex:  # noqa: BLE001
                parse_err = str(ex)
            finally:
                _clear_loading(_overlay_slot)
        elif blobs_all:
            # ---------- 单文件路径：覆盖模式，或拼接模式但只选了一份 ----------
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
                    # 覆盖模式：多选时只分析这一份；新上传的文件会自动成为默认选项（= 覆盖）
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
            except Exception as ex:  # noqa: BLE001
                parse_err = str(ex)
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

    # 数据变化时走一次加载闸门：解析/合并属于高危操作，等资源降到红线 20% 再放行
    import hashlib as _hl
    _sig_src = b"".join(blobs) if blobs else (blob or b"")
    _data_sig = (f"{src}|{_tu}|{len(blobs)}|"
                 f"{_hl.md5(_sig_src).hexdigest()[:12] if _sig_src else 'none'}")
    if df_input is not None and st.session_state.get("__data_sig") != _data_sig:
        st.session_state["__data_sig"] = _data_sig
        # 数据被替换（覆盖）：先把上一份还没保存的结果留档到 输出\，再把它的结果缓存清掉，
        # 保证「上一次的数据 + 图表」留在输出目录里，而当前页面只面向这一份新数据。
        for _k in [k for k in list(st.session_state.keys()) if str(k).startswith("res_")]:
            _autosave_previous(_k)
            st.session_state.pop(_k, None)
        rg.note_op("数据解析")
        rg.log_event("数据就绪", note=_data_sig)
        rg.release_hold(_overlay_slot, "数据已解析，正在释放内存")

    st.divider()
    with st.expander("🎚️ 指标与保存阈值（图内联动不经过这里）", expanded=False):
        # 放进 form：拖动滑条本身不触发页面刷新，只有点「应用」才重算指标/保存口径。
        # 这样即使快速来回拖，也不会排队几十次 rerun → 浏览器不会卡死。
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
        # 2026-11-05 第三版：不再用 tabs 把表格塞进窄侧栏，改成「点一下 → 浮层面板」。
        # segmented_control 点已选中的那一项会取消选择（返回 None），正好等于「再点一次关闭」。
        st.caption("点一下按钮，右侧浮出对应面板（盖在页面之上，不挤占布局）；再点一次即关闭。")
        _other_sel = st.segmented_control(
            "其余数据面板", ["📋 概览", "📐 衰减律", "📊 阈值表"],
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
        # 轻量 markdown 表（比 st.dataframe 每次 rerun 重建 arrow 表便宜得多）
        _mrows = rg.matrix_rows(_rs)
        _hdr = list(_mrows[0].keys())
        _lines = ["| " + " | ".join(_hdr) + " |",
                  "|" + "|".join(["---"] * len(_hdr)) + "|"]
        _lines += ["| " + " | ".join(str(r.get(h, "")) for h in _hdr) + " |" for r in _mrows]
        st.markdown("\n".join(_lines))
        st.caption(f"规则：负载分 ≥ 1.0 = 越过红线；高危操作结束后需降到 "
                   f"{rg.RELEASE_RATIO:.0%}（即 {rg.RELEASE_RATIO:.2f}）才解除「加载中」，"
                   f"最长等待约 3.5 s。记录见 `终端资源.log`。")

# ============================================================ 兼容旧变量名
hist_ids = chosen


# ============================================================ 主界面
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
    st.write("")                      # 与标题基线对齐
    with st.popover("👥 制作组简介", width="stretch"):
        st.markdown("### 制作组简介")
        # 注意：这是 HTML 注释，必须带 unsafe_allow_html=True，否则会被当成普通文本
        # 原样显示在页面上（与 resource_guard 旧版遮罩是同一类 bug）。
        st.markdown("牛逼轰轰的mj，hyg，zqy，js",
                    unsafe_allow_html=True)
        st.caption("美军很勇敢，在企业健身")
    # 画布分辨率：只控制 canvas 的像素密度（DPR），不影响任何计算与指标。
    # 「自动」= 按设备像素比，但封顶 1.5×（4K/2K 屏上 2× 会让画布像素翻 4 倍，是浏览器吃力的主因之一）。
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

# 结果缓存键不包含"历史记忆库选择"（那会随新数据自动变化，导致结果忽然消失）；
# 历史增强是否开启已包含，选中的历史数据集在点击预测时生效。
cache_key = f"res_{key}_{map_sig}_{_tu}_{bool(use_history)}"

res = st.session_state.get(cache_key)

# 只有点击「开始预测」才做分析（不做任何自动分析：避免打开页面/切换控件时触发重活）
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
            # 只保留最近一次结果，避免多份结果（含 4000×300 路径矩阵）常驻内存。
            # 覆盖前先把上一份还没保存的结果自动留档到 输出\（用户没点"保存"也不会丢）。
            for _k in [k for k in list(st.session_state.keys())
                       if str(k).startswith("res_") and k != cache_key]:
                _autosave_previous(_k)
                st.session_state.pop(_k, None)
        except Exception as ex:  # noqa: BLE001
            log_exc("点击预测", ex)
            st.error(f"分析失败：{ex}")
            with st.expander("查看错误详情（已写入 终端诊断.log）", expanded=False):
                st.code("见 终端诊断.log")
        finally:
            # 高危操作：不立即解除，等负载降到红线 20% 再放行（"向上管理"用户节奏）。
            # 分析成功时 keep=True：这层遮罩不揭，直接交给后面的"交互画布预加载"继续盖着，
            # 保证整条链路（分析 → 打包画布 → 前端预加载 23 档图）始终是同一层遮罩；
            # 等前端把图都备好（图片准备率 ≥90%）再由 iframe 自己把这层从 DOM 删掉。
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

# 极端兜底：即使自动改用建议阈值，仍可能一条有效路径都没有（valid_eff 为空）。
# 典型场景：把不同电池/不同分段的数据拼成一条曲线后，末端电压反而比起始还高。
# 此时若直接 np.percentile(空数组) 会抛 IndexError、整页红屏，所以必须先判断再算。
_no_path = len(valid_eff) == 0

# 把这次实际使用的阈值/路径记进结果缓存，供"被覆盖前自动留档"复用
res["ft_eff"], res["uth_eff"], res["rr_eff"] = ft_eff, uth_eff, rr_eff

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

# ---------- 交互画布：结构 A 只画一次，结构 B 由浏览器端 FT 参数本地联动 ----------
# 设计见 live_canvas.py：静态层=离屏 canvas 缓存（贴图 <1ms），动态层=阈值红线/P50/直方图
# 全部在浏览器本地 rAF 合帧重绘，拖动滑条不触发任何服务器计算 → 前端不会卡死。
_u_eq_full = (prep.get("P_init", 0.0) / prep["I_ref"]) if (prep.get("has_I") and prep.get("I_ref")) \
    else float(prep.get("u_start", prep["u_anchor"]))
_live_key = f"{cache_key}|{res.get('fp', 'x')}|rs{_res_scale}"
_live_cache = st.session_state.setdefault("live_cache", {})
try:
    _theme = lc.pick_theme(getattr(st.context, "theme", None))
except Exception:  # noqa: BLE001
    _theme = lc.pick_theme(None)

if _live_key not in _live_cache:
    if _no_path:
        # 0 条有效路径时画布没有可用数据（全 0 RUL 直方图没有意义，也容易再次触发前端异常）：
        # 直接留空，由下面给出解释性提示，不写遮罩、不打包。
        _live_cache[_live_key] = dict(live=None, html=None)
    else:
        # 先亮出屏幕变暗的「加载中」遮罩，再做一次性打包（约 0.3–1 s）。
        # handover=True：这层遮罩**不揭**，交给 iframe 里的"预加载自检"继续盖着（同一层 #dsh-dim），
        # 等 23 档 FT 的缩略图都备好（图片准备率 ≥90%）才由前端从父页面 DOM 删掉 —— 用户全程动不了。
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
        except Exception as ex:  # noqa: BLE001
            log_exc("构建交互画布", ex)
            st.error(f"交互画布生成失败：{ex}")
            _live_cache[_live_key] = dict(live=None, html=None)
        finally:
            # 只有"没生成出 html"时才在这里揭遮罩；生成成功就交给前端预加载阶段接棒。
            if _live_cache.get(_live_key, {}).get("html") is None:
                _clear_loading(_overlay_slot)
else:
    # 画布已缓存（同一份数据重复点预测 / 只改了分辨率）：这一步不会再写交接遮罩，
    # 而 release_hold(keep=True) 又把分析遮罩留在了页面上 —— 必须在这里揭掉，
    # 否则整页会被一层没人负责的 #dsh-dim 永久糊住（用户点哪都没反应）。
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
        _saved_dir = str(_save_outputs(_png1, _png2, res, ft_eff, uth_eff, rr_eff))
        st.session_state["last_out_dir"] = _saved_dir
        # 记下"已手动保存"，避免这份结果被下一份数据覆盖时又被自动留档一次（重复出图）
        res["saved"] = _saved_dir
out_dir = Path(st.session_state.get("last_out_dir", OUTPUT_ROOT))

# ---------------- 浮层面板内容（「📊 其余数据」三个主题） ----------------
# 2026-11-05 第三版：只有点了侧栏的分段按钮才渲染；面板用 CSS position:fixed 浮在页面右侧，
# 盖在所有内容之上，不挤占布局、也不强撑着侧栏变宽。再点一次同一个按钮 → _other_sel 变 None → 面板消失。
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
    _tr_step = max(1, np.asarray(prep["t"]).size // 600)   # 降采样，避免把 9 万点丢给前端
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
            # 轻量 markdown 表（避免每次 rerun 重建 arrow 表）
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
