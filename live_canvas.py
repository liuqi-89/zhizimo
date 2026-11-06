# -*- coding: utf-8 -*-
"""交互式预测画布（纯 Canvas + 原生 JS，无外部依赖）

设计目标（对应需求："结构 A 固定、结构 B 本地联动 / 前端不许卡死"）
--------------------------------------------------------------------------
* **结构 A（静态层）**：历史电压曲线、约束后交付曲线、坐标轴与网格。服务器端只算
  一次，浏览器端画到一张**离屏画布**上缓存；之后每次拖滑条只做一次 ``drawImage``
  把这张底图整体贴回来（GPU 级操作，<1 ms）。
* **结构 B（动态层）**：FT 阈值红线、阈值文字、P50 穿越竖线、RUL 直方图。全部在
  **浏览器本地**用 ``requestAnimationFrame`` 合帧重绘，只画几十条线，不需要服务器
  参与，也不重建 iframe/图表 → 拖动是丝滑的。
* **后端不猛**：FT 网格上的 RUL 分位/直方图只在"换数据"时算一次（约 0.2–1 s），
  期间前端显示 CSS 闪烁的「加载中……」占位，页面不会假死。

用法::

    live = build_live_data(prep, g, mc)            # 服务器端，每份数据算一次
    st.components.v1.html(build_html(live), height=..., scrolling=False)
"""
from __future__ import annotations

import json
import string
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

import rul_pipeline as rp

FT_MIN, FT_MAX, FT_STEP = 3.0, 8.0, 0.1
FT_SPAN_MAX = 2.2          # 滑条最多覆盖 2.2 个百分点（避免低衰减档被拉得过于稀疏）
BIN_TARGET = 72           # 直方图目标柱数（用于自动选好看的分箱宽度）
PRED_H = 400               # 上图 CSS 高度
HIST_H = 290               # 下图 CSS 高度
HIST_WIN_H = 700.0         # 历史曲线展示的时间窗（小时）

# ---- 浅色主题（与 Streamlit 默认浅色主题一致） ----
THEME_LIGHT = dict(
    bg="#ffffff", fg="#31333f", subfg="#5c5f6b", grid="#eceef2", axis="#c9ccd4",
    hist="#2c6fbb", det="#111111", red="#c0392b", bar="#5b9bd5", p5="#7b5ea7",
    start="#8a8f99",
)
# ---- 深色主题（用户截图里的暗色模式） ----
THEME_DARK = dict(
    bg="#0e1117", fg="#fafafa", subfg="#b9bec7", grid="#262b36", axis="#3b4150",
    hist="#6ba3e8", det="#e8e8e8", red="#ff6b6b", bar="#4f8fd6", p5="#b39ddb",
    start="#8a8f99",
)


# ============================================================ 数据准备（服务器端）
def ft_grid() -> np.ndarray:
    return np.round(np.arange(FT_MIN, FT_MAX + 1e-9, FT_STEP), 2)


def _downsample_view(prep: dict, window_h: float = HIST_WIN_H, target: int = 760):
    """取历史末段 window_h 小时并降采样到 ~target 点（静态结构 A 的数据）。"""
    t = np.asarray(prep["t"], float)
    u = np.asarray(prep["u_s"], float)
    t0 = float(prep["t_end"])
    m = t >= max(0.0, t0 - window_h)
    n = int(m.sum())
    step = max(1, n // target)
    return t[m][::step], u[m][::step]


def _rect_clip(p0, p1, x_lo, x_hi, y_lo, y_hi):
    """Liang-Barsky：把线段 p0-p1 裁剪到矩形内，返回 ((x0,y0),(x1,y1)) 或 None。"""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - x_lo), (dx, x_hi - x0), (-dy, y0 - y_lo), (dy, y_hi - y0)):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
    return ((x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy))


def _clip_polyline(xs: Iterable[float], ys: Iterable[float],
                   x_lo, x_hi, y_lo, y_hi) -> list[list[list[float]]]:
    """把折线裁剪成若干段（每段是 [[x,y], ...]），供前端直接 stroke。"""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    segs: list[list[list[float]]] = []
    cur: list[list[float]] = []
    for i in range(len(xs) - 1):
        a = (float(xs[i]), float(ys[i]))
        b = (float(xs[i + 1]), float(ys[i + 1]))
        r = _rect_clip(a, b, x_lo, x_hi, y_lo, y_hi)
        if r is None:
            if len(cur) > 1:
                segs.append(cur)
            cur = []
            continue
        if len(cur) == 0:
            cur = [[round(r[0][0], 2), round(r[0][1], 4)]]
        elif abs(cur[-1][0] - r[0][0]) > 1e-9 or abs(cur[-1][1] - r[0][1]) > 1e-9:
            if len(cur) > 1:
                segs.append(cur)
            cur = [[round(r[0][0], 2), round(r[0][1], 4)]]
        if abs(cur[-1][0] - r[1][0]) > 1e-2 or abs(cur[-1][1] - r[1][1]) > 1e-4:
            cur.append([round(r[1][0], 2), round(r[1][1], 4)])
    if len(cur) > 1:
        segs.append(cur)
    return segs


def _nice_step(x: float) -> float:
    if x <= 0:
        return 1.0
    e = 10.0 ** np.floor(np.log10(x))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if x <= m * e:
            return float(m * e)
    return float(10.0 * e)


def _rul_grid(U: np.ndarray, u0: float, uths: np.ndarray, win: float = 10.0) -> np.ndarray:
    """一次性向量化求出每个阈值下 4000 条路径的 RUL（未穿越 = NaN）。"""
    U = np.asarray(U, float)
    n_mc, n_col = U.shape
    tstep = win
    R = np.full((len(uths), n_mc), np.nan, dtype=np.float64)
    U1 = U[:, 1:]
    for i, uth in enumerate(uths):
        below = U1 <= uth
        has = below.any(axis=1)
        if not has.any():
            continue
        idx = below.argmax(axis=1)             # 首次穿越的列（window 结束处）
        rows = np.nonzero(has)[0]
        j = idx[rows]
        u_prev = U[rows, j]
        u_hit = U1[rows, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (u_prev - uth) / (u_prev - u_hit)
        frac = np.where(np.isfinite(frac), frac, 0.5)
        R[i, rows] = (j + np.clip(frac, 0.0, 1.0)) * tstep
    return R


def build_live_data(prep: dict, g: dict, mc: dict, u_eq: float | None = None) -> dict:
    """服务器端一次性预计算：静态曲线 + FT 网格上的 RUL 分位与直方图。"""
    U = np.asarray(mc["U"], float)
    u0 = float(prep["u_anchor"])
    t0 = float(prep["t_end"])
    win = rp.WIN_H

    grid = ft_grid()
    uths = np.array([rp.uth_for_ft(prep, ft) for ft in grid], float)
    R = _rul_grid(U, u0, uths, win)

    fin = np.isfinite(R)
    n_valid = fin.sum(axis=1)
    cens = 100.0 * (1.0 - n_valid / max(R.shape[1], 1))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pct = np.array([np.nanpercentile(np.where(fin, R, np.nan), q, axis=1)
                        for q in (5, 25, 50, 75, 95)], float)   # (5, n_ft)

    # 有效区间：首个"未失效"档 → 视界内仍能穿越的最高档，且跨度不超过 FT_SPAN_MAX
    _p95 = pct[4]
    ok = np.isfinite(_p95) & (pct[2] > 0.0)
    i_lo = int(np.argmax(ok)) if ok.any() else 0
    finite95 = np.isfinite(_p95)
    i_hi = int(np.nonzero(finite95)[0][-1]) if finite95.any() else len(grid) - 1
    ft_lo = float(grid[i_lo])
    ft_hi = float(min(grid[i_hi], ft_lo + FT_SPAN_MAX))
    i_hi = int(np.nonzero(grid <= ft_hi + 1e-9)[0][-1])
    i_hi = max(i_hi, i_lo)
    sl = slice(i_lo, i_hi + 1)
    sub = dict(
        grid=grid[sl], uths=uths[sl],
        p5=pct[0][sl], p25=pct[1][sl], p50=pct[2][sl], p75=pct[3][sl], p95=pct[4][sl],
        cens=cens[sl],
    )
    ft0_sel = float(sub["grid"][0])

    # ---- 直方图：统一分箱（宽度按时域挑选好看的值），每档按自身峰值归一 ----
    _p95_vis = sub["p95"]
    x_hi_t = float(min(max(420.0, float(np.nanmax(_p95_vis)) * 1.12 + 60.0), rp.HORIZON_H))
    bin_w = _nice_step(x_hi_t / BIN_TARGET)
    edges = np.arange(0.0, x_hi_t + bin_w * 1.5, bin_w)

    n_ft, n_bin = len(sub["grid"]), len(edges) - 1
    rel = np.zeros((n_ft, n_bin), float)
    cnt = np.zeros((n_ft, n_bin), int)
    share = np.zeros((n_ft, n_bin), float)
    for i in range(n_ft):
        r = R[i_lo + i][np.isfinite(R[i_lo + i])]
        if len(r) == 0:
            continue
        c, _ = np.histogram(r, bins=edges)
        cnt[i] = c
        ymax = int(c.max()) if len(c) else 0
        rel[i] = c / max(ymax, 1)
        share[i] = c / max(len(r), 1)

    # ---- 静态曲线（结构 A）：裁剪到坐标域内，前端直接 stroke ----
    # 2026-11-05 修：图上「约束后交付曲线」改画 4000 条路径的**逐点中位路径**（band P50）。
    # 旧代码画的是「乘子恒为 1」的确定性路径——那只是概率分布的下包络，并不代表典型电池；
    # 历史增强把乘子分布拉宽后，它的穿越时间会比中位 RUL 大好几倍，于是图 1 里
    # 红竖线（P50）与红横线（阈值）的交点完全落在曲线外面（用户实机反馈「红虚线飘开了」）。
    th, uh = _downsample_view(prep)
    _band = mc.get("band") or {}
    _b50 = _band.get(50) if isinstance(_band, dict) else None
    det = np.asarray(_b50 if _b50 is not None else mc["det_curve"], float)
    t_ext = t0 + np.arange(len(det)) * win

    # 交点：中位路径穿越各档阈值的时刻（前端红竖线就画在这里，交点必然在曲线上）
    cross = []
    for _u in sub["uths"]:
        _idx = np.where(det <= _u)[0]
        if len(_idx) == 0:
            cross.append(None)
            continue
        _i = int(_idx[0])
        if _i == 0:
            cross.append(0.0)
            continue
        _up, _un = float(det[_i - 1]), float(det[_i])
        cross.append(round(float((_i - 1 + (_up - _u) / max(_up - _un, 1e-12)) * win), 1))

    x_lo = max(0.0, t0 - HIST_WIN_H)
    x_hi = float(min(t_ext[-1], t0 + x_hi_t))
    vis = (t_ext <= x_hi + 1e-9)
    det_vis = det[vis] if vis.any() else det[:1]
    det_t_vis = t_ext[vis] if vis.any() else t_ext[:1]
    ueq = float(u_eq if u_eq is not None else prep.get("u_start", u0))
    uth_lo = float(np.nanmin(sub["uths"]))
    uth_hi = float(np.nanmax(sub["uths"]))
    y_hi = max(float(np.nanmax(uh)), uth_hi, float(np.nanmax(det_vis))) + 0.006
    y_lo = min(float(np.nanmin(uh)), uth_lo, float(np.nanmin(det_vis))) - 0.006

    hist_segs = _clip_polyline(th, uh, x_lo, x_hi, y_lo, y_hi)
    det_segs = _clip_polyline(det_t_vis, det_vis, x_lo, x_hi, y_lo, y_hi)

    return dict(
        t0=t0, u_eq=ueq, n_paths=int(mc.get("n_mc", U.shape[0])),
        x_lo=float(x_lo), x_hi=float(x_hi), y_lo=float(y_lo), y_hi=float(y_hi),
        hist_segs=hist_segs, det_segs=det_segs,
        hx0=0.0, hx1=float(x_hi_t), edges=[round(float(v), 2) for v in edges],
        ft=[round(float(v), 2) for v in sub["grid"]],
        uth=[round(float(v), 5) for v in sub["uths"]],
        cross=cross,
        p5=[round(float(v), 1) for v in sub["p5"]], p25=[round(float(v), 1) for v in sub["p25"]],
        p50=[round(float(v), 1) for v in sub["p50"]], p75=[round(float(v), 1) for v in sub["p75"]],
        p95=[round(float(v), 1) for v in sub["p95"]],
        cens=[round(float(v), 1) for v in sub["cens"]],
        rel=[[round(float(v), 4) for v in row] for row in rel],
        cnt=[[int(v) for v in row] for row in cnt],
        share=[[round(float(v), 4) for v in row] for row in share],
        n_paths_valid=[int(v) for v in n_valid[sl]],
        ft_default=ft0_sel,
        bin_w=float(bin_w),
    )


# ============================================================ 主题
def pick_theme(theme=None) -> dict:
    """把 Streamlit 的 theme 信息转成画布配色；未知时用浅色。"""
    base = None
    try:
        t = theme if theme is not None else None
        if t is None:
            import streamlit as st
            t = getattr(st.context, "theme", None)
        if isinstance(t, dict):
            base = t.get("type") or t.get("base")
        else:
            base = getattr(t, "type", None) or getattr(t, "base", None)
    except Exception:  # noqa: BLE001
        base = None
    return dict(THEME_DARK if str(base).lower() == "dark" else THEME_LIGHT)


# ============================================================ HTML / JS
_HTML = string.Template("""<!DOCTYPE html>
<html lang="zh-CN" translate="no" class="notranslate"><head><meta charset="utf-8">
<meta name="google" content="notranslate"><meta name="microsoft" content="notranslate">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: $BG; color: $FG;
         font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; }
  #wrap { width: 100%; padding: 2px 2px 4px; }
  .card { position: relative; background: $CARD; border: 1px solid $BORDER;
          border-radius: 12px; padding: 10px 12px 12px; margin: 6px 2px 12px;
          box-shadow: 0 1px 3px rgba(15,30,50,.06); }
  .hdr { font-size: 14.5px; font-weight: 600; margin: 2px 0 8px 2px; color: $FG; }
  .cvbox { position: relative; }
  .ctl { display: flex; align-items: center; gap: 10px; margin: 10px 2px 6px 2px; }
  .ctl input[type=range] { flex: 1; accent-color: $RED; height: 20px; }
  .ctl .val { font-size: 13px; font-weight: 700; color: $RED; min-width: 86px; text-align: right; }
  .ctl .lab { font-size: 12.5px; color: $SUBFG; white-space: nowrap; }
  .stat { font-size: 12.5px; color: $SUBFG; margin: 2px 2px 0; }
  canvas { display: block; width: 100%; }
  /* 独立灰屏加载层：两张图各自一份，互不影响 */
  .ovl { position: absolute; inset: 0; display: none; align-items: center;
         justify-content: center; gap: 2px; flex-direction: column;
         background: rgba(20,28,40,.42); backdrop-filter: blur(1.6px);
         -webkit-backdrop-filter: blur(1.6px); border-radius: 8px;
         color: $FG; font-size: 13.5px; font-weight: 600; letter-spacing: 1px; }
  .ovl.on { display: flex; }
  .ovl .dots { display: inline-block; width: 26px; text-align: left; }
  @keyframes __blink { 0%,20% {opacity:1;} 50% {opacity:.15;} 80%,100% {opacity:1;} }
  .ovl .dots span { animation: __blink 1.35s infinite; }
  .ovl .dots span:nth-child(2) { animation-delay: .18s; }
  .ovl .dots span:nth-child(3) { animation-delay: .36s; }
  /* 整块画布的总闸门：预测结束后先内部自检快速拖动，自检通过才允许用户接管 */
  #gate { position: fixed; inset: 0; z-index: 60; display: none; align-items: center;
          justify-content: center; flex-direction: column; gap: 12px; padding: 0 20px;
          background: rgba(8,14,24,.66); color: $FG; font-size: 14.5px;
          letter-spacing: .6px; text-align: center; }
  #gate.on { display: flex; }
  #gate .ring { width: 26px; height: 26px; border-radius: 50%;
                border: 2.5px solid rgba(111,183,255,.25); border-top-color: #6fb7ff;
                animation: __spin .85s linear infinite; will-change: transform; }
  #gate .sub { font-size: 12.5px; color: $FG; opacity: .92; line-height: 1.85; max-width: 620px;
               text-shadow: 0 1px 3px rgba(0,0,0,.6); }
  #gate #gatemsg { color: $FG; font-weight: 600; text-shadow: 0 1px 3px rgba(0,0,0,.6); }
  /* 一旦进入加载/保护状态：整块画布切断人机交互（哪怕用户正按着滑条也要强制中断） */
  body.locked #c1, body.locked #c2 { pointer-events: none; }
  body.locked #rng { pointer-events: none; }
  body.locked .hdr, body.locked .stat { opacity: .72; }
  #gate .bar { position: relative; width: min(340px, 70vw); height: 5px; border-radius: 4px;
               background: rgba(255,255,255,.16); overflow: hidden; }
  #gate .bar i { position: absolute; inset: 0; width: 40%; border-radius: 4px;
                 background: linear-gradient(90deg, rgba(111,183,255,0), #6fb7ff, rgba(111,183,255,0));
                 will-change: transform; animation: __slide 1.2s ease-in-out infinite; }
  @keyframes __spin { to { transform: rotate(360deg); } }
  @keyframes __slide { 0% { transform: translateX(-120%); } 100% { transform: translateX(260%); } }
</style></head>
<body><div id="wrap">
  <div class="card" id="c1">
    <div class="hdr">图 1 · 预测曲线（拖动滑条：红线 / P50 本地即时联动）</div>
    <div class="cvbox">
      <canvas id="cv"></canvas>
      <div class="ovl" id="povl"><span>刷新有点跟不上，正在降级</span>
        <span class="dots"><span>·</span><span>·</span><span>·</span></span></div>
    </div>
    <div class="ctl">
      <span class="lab">FT（失效阈值 %）</span>
      <input id="rng" type="range" min="0" max="1" step="1" value="0">
      <span class="val" id="ftv">--</span>
    </div>
    <div class="stat" id="stat">--</div>
  </div>
  <div class="card" id="c2">
    <div class="hdr">图 2 · 当前 FT 下的 RUL 概率分布（柱高按该档峰值归一）</div>
    <div class="cvbox">
      <canvas id="ch"></canvas>
      <div class="ovl" id="hovl"><span>柱状图加载中</span>
        <span class="dots"><span>·</span><span>·</span><span>·</span></span></div>
    </div>
    <div class="stat" id="tip">&nbsp;</div>
  </div>
  <div id="gate">
    <div class="ring"></div>
    <div id="gatemsg">预测完成 · 正在提前加载图表，请稍候…</div>
    <div class="bar"><i></i></div>
    <div class="sub">预计耗时两分钟：正在把每一档 FT 的图表提前渲染进内置 output，之后随便拖都不卡。</div>
    <div class="sub" style="margin-top:4px;font-size:24px;letter-spacing:6px;font-weight:700;opacity:.95;">让子弹飞一会</div>
  </div>
</div>
<script>
var DATA = $DATA;
var COL = $COL;
// 画布像素密度：-1 = 自动（按屏幕 DPR，封顶 1.5×）；>0 = 用户在侧栏「画布分辨率」里选的固定倍率。
// 只影响 canvas 画多少像素，不影响任何计算；调低能显著减少画布内存与每帧填充成本。
var RSCALE = $RSCALE;

var PRED_H = $PRED_H, HIST_H = $HIST_H, ML = 64, MR = 22, MT = 30, MB = 46;
var HIST_ML = 64, HIST_MR = 22, HIST_MT = 26, HIST_MB = 46;

var wrap = document.getElementById('wrap');
var cv = document.getElementById('cv'), ctx = cv.getContext('2d');
var ch = document.getElementById('ch'), hctx = ch.getContext('2d');
var rng = document.getElementById('rng');
var ftv = document.getElementById('ftv');
var stat = document.getElementById('stat');
var tip = document.getElementById('tip');
var povl = document.getElementById('povl'), hovl = document.getElementById('hovl');

var offA = document.createElement('canvas'), actx = offA.getContext('2d');
var W = 800, HIST_W = 800,
    DPR = (RSCALE > 0) ? RSCALE : Math.min(1.5, Math.max(1, window.devicePixelRatio || 1));
var curK = 0, rafId = 0, needPaint = false;
var gate = document.getElementById('gate'), gateMsg = document.getElementById('gatemsg');
var dragging = false, dragTimer = 0, DEGRADE = false, selfTestDone = false;

function setupCanvas(el, c2, hcss) {
  el.style.width = '100%';
  el.style.height = hcss + 'px';
  el.width = Math.max(1, Math.round(W * DPR));
  el.height = Math.round(hcss * DPR);
  c2.setTransform(DPR, 0, 0, DPR, 0, 0);
}

function fmt(v, d) { return (v === null || v === undefined || isNaN(v)) ? '--' : v.toFixed(d); }

function ticks(lo, hi, n) {
  if (!(hi > lo)) { return [lo]; }
  var raw = (hi - lo) / Math.max(1, n);
  var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
  var norm = raw / mag;
  var step = (norm < 1.5 ? 1 : (norm < 3 ? 2 : (norm < 7 ? 5 : 10))) * mag;
  var t = [], v = Math.ceil(lo / step - 1e-9) * step;
  for (; v <= hi + 1e-9; v += step) { t.push(v); }
  return t;
}

function tx(x) { return ML + (x - DATA.x_lo) / (DATA.x_hi - DATA.x_lo) * (W - ML - MR); }
function ty(y) { return MT + (DATA.y_hi - y) / (DATA.y_hi - DATA.y_lo) * (PRED_H - MT - MB); }
function hx(x) { return HIST_ML + (x - DATA.hx0) / (DATA.hx1 - DATA.hx0) * (HIST_W - HIST_ML - HIST_MR); }
function hy(v) { return HIST_MT + (1.06 - v) / 1.06 * (HIST_H - HIST_MT - HIST_MB); }

function lineSegs(c2, segs, color, width) {
  c2.strokeStyle = color; c2.lineWidth = width; c2.lineJoin = 'round'; c2.lineCap = 'round';
  for (var i = 0; i < segs.length; i++) {
    var s = segs[i];
    if (s.length < 2) { continue; }
    c2.beginPath();
    c2.moveTo(tx(s[0][0]), ty(s[0][1]));
    for (var j = 1; j < s.length; j++) { c2.lineTo(tx(s[j][0]), ty(s[j][1])); }
    c2.stroke();
  }
}

function histSegs(c2, segs) {
  c2.strokeStyle = COL.hist; c2.lineWidth = 1.9; c2.lineJoin = 'round';
  for (var i = 0; i < segs.length; i++) {
    var s = segs[i];
    if (s.length < 2) { continue; }
    c2.beginPath();
    c2.moveTo(tx(s[0][0]), ty(s[0][1]));
    for (var j = 1; j < s.length; j++) { c2.lineTo(tx(s[j][0]), ty(s[j][1])); }
    c2.stroke();
  }
}

/* ---------- 结构 A：坐标轴 + 两条静态曲线（只画一次，存离屏画布） ---------- */
function buildStatic() {
  actx.setTransform(DPR, 0, 0, DPR, 0, 0);
  actx.clearRect(0, 0, W, PRED_H);
  actx.fillStyle = COL.bg; actx.fillRect(0, 0, W, PRED_H);
  actx.font = '11.5px "Microsoft YaHei", Arial, sans-serif';

  var xt = ticks(DATA.x_lo, DATA.x_hi, 9), yt = ticks(DATA.y_lo, DATA.y_hi, 6);
  var i, v;
  // 网格
  actx.strokeStyle = COL.grid; actx.lineWidth = 1;
  for (i = 0; i < xt.length; i++) {
    actx.beginPath(); actx.moveTo(tx(xt[i]), MT); actx.lineTo(tx(xt[i]), PRED_H - MB); actx.stroke();
  }
  for (i = 0; i < yt.length; i++) {
    actx.beginPath(); actx.moveTo(ML, ty(yt[i])); actx.lineTo(W - MR, ty(yt[i])); actx.stroke();
  }
  // 轴标签
  actx.strokeStyle = COL.axis; actx.lineWidth = 1;
  actx.beginPath(); actx.moveTo(ML, MT); actx.lineTo(ML, PRED_H - MB); actx.lineTo(W - MR, PRED_H - MB); actx.stroke();
  actx.fillStyle = COL.subfg; actx.textAlign = 'center'; actx.textBaseline = 'top';
  for (i = 0; i < xt.length; i++) { actx.fillText(fmt(xt[i], 0), tx(xt[i]), PRED_H - MB + 7); }
  actx.textAlign = 'right'; actx.textBaseline = 'middle';
  for (i = 0; i < yt.length; i++) { actx.fillText(fmt(yt[i], 3), ML - 8, ty(yt[i])); }
  actx.textAlign = 'left'; actx.textBaseline = 'top';
  actx.fillText('时间 t（h）', W - MR - 74, PRED_H - MB + 24);

  // 预测起点竖线（静态）
  if (DATA.t0 >= DATA.x_lo && DATA.t0 <= DATA.x_hi) {
    actx.strokeStyle = COL.start; actx.lineWidth = 1.1; actx.setLineDash([3, 4]);
    actx.beginPath(); actx.moveTo(tx(DATA.t0), MT); actx.lineTo(tx(DATA.t0), PRED_H - MB); actx.stroke();
    actx.setLineDash([]);
    actx.fillStyle = COL.start; actx.textAlign = 'left'; actx.textBaseline = 'top';
    actx.fillText('预测起点', tx(DATA.t0) + 4, MT + 2);
  }

  // 历史曲线 + 交付曲线（结构 A，线宽已加粗）
  histSegs(actx, DATA.hist_segs);
  lineSegs(actx, DATA.det_segs, COL.det, 3.0);

  // 图例
  var lx = ML + 10, ly = MT + 8;
  actx.lineWidth = 1.9; actx.strokeStyle = COL.hist;
  actx.beginPath(); actx.moveTo(lx, ly + 5); actx.lineTo(lx + 22, ly + 5); actx.stroke();
  actx.fillStyle = COL.fg; actx.font = '12px "Microsoft YaHei", Arial, sans-serif';
  actx.textAlign = 'left'; actx.textBaseline = 'middle'; actx.fillText('历史电压（10 h 平滑）', lx + 28, ly + 5);
  actx.lineWidth = 3.0; actx.strokeStyle = COL.det;
  actx.beginPath(); actx.moveTo(lx, ly + 23); actx.lineTo(lx + 22, ly + 23); actx.stroke();
  actx.fillStyle = COL.fg; actx.fillText('约束后交付曲线', lx + 28, ly + 23);
}

/* ---------- 结构 B：阈值红线 + P50 竖线（每次拖动只画这一层） ---------- */
function drawDynamic(k) {
  DYN_SEEN[k] = 1;                         /* 记录：该档的动态层已真实渲染过（图片就绪口径之一） */
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, PRED_H);
  ctx.drawImage(offA, 0, 0, W, PRED_H);      // 贴回结构 A 的静态底图（<1ms）

  var ft = DATA.ft[k], uth = DATA.uth[k], p50 = DATA.p50[k];
  /* 2026-11-05 修「红虚线飘开」：
     横线用真实阈值 uth（与 u_eq*(1-ft/100) 同一个数）；
     竖线用「中位路径与阈值线的交点」DATA.cross[k]，不再用统计 P50 当位置 ——
     两者口径不同时（历史增强把乘子分布拉宽后差得很多），交点会跑到曲线外面。
     取不到交点（该档在视界内穿不过阈值）才退回统计 P50。 */
  var xm = (DATA.cross && DATA.cross.length > k) ? DATA.cross[k] : null;
  if (xm === null || xm === undefined || !isFinite(xm)) { xm = null; }
  /* 2026-11-06（无为而治）：交点在视界外/不存在时，干脆不画竖线——不再退回统计 P50，
     否则会出现「横线与曲线的交点在图外、竖线却画在图内」的自相矛盾。 */
  if (uth >= DATA.y_lo && uth <= DATA.y_hi) {
    ctx.strokeStyle = COL.red; ctx.lineWidth = 2.4; ctx.setLineDash([8, 5]);
    ctx.beginPath(); ctx.moveTo(ML, ty(uth)); ctx.lineTo(W - MR, ty(uth)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COL.red; ctx.font = 'bold 12.5px "Microsoft YaHei", Arial, sans-serif';
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText('FT = ' + fmt(ft, 1) + '%   阈值 ' + fmt(uth, 4) + ' V', W - MR - 6, ty(uth) - 6);
  }
  if (xm !== null) {
    var xp = DATA.t0 + xm;
    if (xp >= DATA.x_lo && xp <= DATA.x_hi) {
      ctx.strokeStyle = COL.red; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(tx(xp), MT); ctx.lineTo(tx(xp), PRED_H - MB); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COL.red; ctx.font = '12.5px "Microsoft YaHei", Arial, sans-serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('P50 穿越 ≈ ' + fmt(xm, 0) + ' h', tx(xp) + 5, PRED_H - MB - 6);
    }
  }
  VLINE.shown = !!(xm !== null && (DATA.t0 + xm) >= DATA.x_lo && (DATA.t0 + xm) <= DATA.x_hi);
  VLINE.x = xm;
  ftv.textContent = fmt(ft, 1) + '%';
  stat.textContent = '当前 FT = ' + fmt(ft, 1) + '%   阈值 ' + fmt(uth, 4) + ' V   ｜   '
      + 'P5 ' + fmt(DATA.p5[k], 0) + ' h · P50 ' + fmt(DATA.p50[k], 0) + ' h · P95 '
      + fmt(DATA.p95[k], 0) + ' h   ｜   删失 ' + fmt(DATA.cens[k], 1) + '%'
      + '   ｜   ' + _readyPctText();
}

/* ---------- 直方图（全分辨率只画一次，画完立刻压成 1/4 缩略图） ---------- */
function drawHist(k) {
  if (k === HIST_DRAWN_K) {                /* 当前档已经是全分辨率图，别重复画 */
    tip.textContent = _histTip(k);
    return;
  }
  hctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  hctx.clearRect(0, 0, HIST_W, HIST_H);
  hctx.fillStyle = COL.bg; hctx.fillRect(0, 0, HIST_W, HIST_H);
  hctx.font = '11.5px "Microsoft YaHei", Arial, sans-serif';

  var xt = ticks(DATA.hx0, DATA.hx1, 9), i;
  hctx.strokeStyle = COL.grid; hctx.lineWidth = 1;
  for (i = 0; i < xt.length; i++) {
    hctx.beginPath(); hctx.moveTo(hx(xt[i]), HIST_MT); hctx.lineTo(hx(xt[i]), HIST_H - HIST_MB); hctx.stroke();
  }
  var yt = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  for (i = 0; i < yt.length; i++) {
    hctx.beginPath(); hctx.moveTo(HIST_ML, hy(yt[i])); hctx.lineTo(HIST_W - HIST_MR, hy(yt[i])); hctx.stroke();
  }
  hctx.strokeStyle = COL.axis;
  hctx.beginPath(); hctx.moveTo(HIST_ML, HIST_MT); hctx.lineTo(HIST_ML, HIST_H - HIST_MB);
  hctx.lineTo(HIST_W - HIST_MR, HIST_H - HIST_MB); hctx.stroke();
  hctx.fillStyle = COL.subfg; hctx.textAlign = 'center'; hctx.textBaseline = 'top';
  for (i = 0; i < xt.length; i++) { hctx.fillText(fmt(xt[i], 0), hx(xt[i]), HIST_H - HIST_MB + 7); }
  hctx.textAlign = 'right'; hctx.textBaseline = 'middle';
  for (i = 0; i < yt.length; i++) { hctx.fillText(fmt(yt[i], 1), HIST_ML - 8, hy(yt[i])); }
  hctx.textAlign = 'left'; hctx.fillText('相对频率（该档按峰值归一）', HIST_W - HIST_MR - 178, HIST_H - HIST_MB + 24);

  // 柱
  var edges = DATA.edges, rel = DATA.rel[k];
  hctx.fillStyle = COL.bar;
  for (i = 0; i < rel.length; i++) {
    if (rel[i] <= 0) { continue; }
    var x0 = hx(edges[i]), x1 = hx(edges[i + 1]);
    var y0 = hy(0), y1 = hy(Math.min(1.0, rel[i]));
    hctx.fillRect(x0 + 0.5, y1, Math.max(1.0, x1 - x0 - 1.0), y0 - y1);
  }
  // P5 / P50 / P95 竖线
  var marks = [[DATA.p5[k], COL.p5, 1.8], [DATA.p50[k], COL.red, 2.2], [DATA.p95[k], COL.p5, 1.8]];
  var names = ['P5', 'P50', 'P95'];
  hctx.textAlign = 'center'; hctx.textBaseline = 'top';
  for (i = 0; i < marks.length; i++) {
    if (isNaN(marks[i][0])) { continue; }
    hctx.strokeStyle = marks[i][1]; hctx.lineWidth = marks[i][2];
    hctx.beginPath(); hctx.moveTo(hx(marks[i][0]), HIST_MT); hctx.lineTo(hx(marks[i][0]), HIST_H - HIST_MB); hctx.stroke();
    hctx.fillStyle = marks[i][1]; hctx.fillText(names[i], hx(marks[i][0]), HIST_MT + 2);
  }
  tip.textContent = _histTip(k);
  HIST_DRAWN_K = k;
  PREVIEW_K = -1;
  _thumbPut(k);          /* 关键：画完只留 1/4 缩略图，绝不留全分辨率位图（防渲染进程 OOM） */
}

/* 拖动中的"贴缩略图"：1/4 分辨率放大贴回（<0.3 ms，略糊但跟手），松手后再补全分辨率 */
function previewHist(k) {
  if (PREVIEW_K === k) { return true; }
  var t = _outGet(k);
  if (!t) { return false; }
  try {
    hctx.setTransform(1, 0, 0, 1, 0, 0);
    hctx.clearRect(0, 0, ch.width, ch.height);
    hctx.drawImage(t, 0, 0, ch.width, ch.height);
    HIST_DRAWN_K = -1;
    PREVIEW_K = k;
    tip.textContent = _histTip(k);
    return true;
  } catch (e) { _outDrop(k); return false; }
}

function _histTip(k) {
  return '当前档有效路径 ' + DATA.n_paths_valid[k] + ' / ' + DATA.n_paths
      + '（' + fmt(100 - DATA.cens[k], 1) + '% 在 3000 h 内穿越）｜ 分箱宽度 '
      + fmt(DATA.bin_w, 0) + ' h ｜ ' + _readyPctText();
}

/* ---------- 单层全局遮罩（跨父页面，2026-11-04 22:54 用户需求）----------
   需求原话："把图一的加载屏糊在最上层，图二加载的时候也把同一层糊着，图备好以后才揭开，
   只要保证加载屏一直糊着不动，就让用户动不了" —— 也就是整页只允许有一层遮罩。
   实现：Streamlit 的 components.html 是 srcdoc iframe（sandbox 含 allow-same-origin），
   与本页同源，所以这里直接接管**父页面**的 DOM：
     1) 优先复用 Python 侧已经写好的 #dsh-dim（点预测时就在了），只改它的文案 → 不闪、不移动；
     2) 没有就自己建 #dsh-global-ovl（例如 rerun 时画布走缓存、Python 没再画遮罩）；
     3) 揭遮罩 = 从父页面 DOM 删掉这层节点，一次删干净，不会残留第二层。
   极端情况下拿不到父页面（理论上不会）→ 退回 iframe 内的 #gate（只盖画布区域），
   Python 侧那层遮罩自带 240 s 自动隐藏动画兜底，绝不让整页永久锁死。 */
var PO = { ok: false, el: null, msg: null, own: false, wd: 0, text: '', ever: false };

function _poStyle(pd) {
  if (pd.getElementById('dsh-po-style')) { return; }
  var st = pd.createElement('style');
  st.id = 'dsh-po-style';
  st.textContent =
    '#dsh-global-ovl{position:fixed;inset:0;z-index:2147483000;display:flex;flex-direction:column;'
    + 'align-items:center;justify-content:center;gap:14px;background:rgba(8,14,24,.82);'
    + 'color:#eaf2ff;pointer-events:auto;contain:strict;'
    + 'font-family:system-ui,"Microsoft YaHei",Arial,sans-serif;transition:opacity .26s ease}'
    + '#dsh-global-ovl .po-ring{width:34px;height:34px;border-radius:50%;'
    + 'border:2.5px solid rgba(111,183,255,.25);border-top-color:#6fb7ff;'
    + 'animation:dshposp .85s linear infinite;will-change:transform}'
    + '#dsh-global-ovl .po-msg{font-size:15px;font-weight:600;letter-spacing:.6px;'
    + 'max-width:min(680px,88vw);text-align:center;line-height:1.8}'
    + '#dsh-global-ovl .po-sub{font-size:12.5px;color:#a9b8cc;line-height:1.8;'
    + 'text-align:center;max-width:min(640px,86vw)}'
    + '#dsh-global-ovl .po-bar{position:relative;width:min(340px,66vw);height:5px;'
    + 'border-radius:4px;background:rgba(255,255,255,.16);overflow:hidden}'
    + '#dsh-global-ovl .po-bar i{position:absolute;inset:0;width:40%;border-radius:4px;'
    + 'background:linear-gradient(90deg,rgba(111,183,255,0),#6fb7ff,rgba(111,183,255,0));'
    + 'will-change:transform;animation:dshposlide 1.2s ease-in-out infinite}'
    + '@keyframes dshposp{to{transform:rotate(360deg)}}'
    + '@keyframes dshposlide{0%{transform:translateX(-120%)}100%{transform:translateX(260%)}}';
  pd.head.appendChild(st);
}

/* ---------- 父页面 React DOM 防崩补丁（2026-11-05 事故加固）----------
   事故现象：浏览器主区顶部出现红色错误块
     NotFoundError: Failed to execute 'removeChild' on 'Node':
     The node to be removed is not a child of this node.
   堆栈是 React 的 commitMutationEffects → commitDeletionEffectsOnFiber（删除子树时
   调 removeChild），说明 React fiber 认为某个节点还挂在容器下，真实 DOM 里却没有它；
   错误边界接管后整页卡死。
   这种「React 树与 DOM 不一致」可能来自：① Streamlit 1.47+ 的 st.empty() 幽灵子树
   （同一次 script run 里后写的子节点更少时旧节点清不掉，issue #12069）；② 本 iframe
   为了做单层遮罩接管了父页面 #dsh-dim 节点；③ 浏览器扩展/翻译改动页面 DOM。
   不管来源是谁，只要 React 再调一次 removeChild 就会整页崩，用户只能刷新。
   所以这里给父窗口的 Node.prototype 打一个「只在真出问题时才让路」的最小补丁：
     - removeChild(c)：c 不是 this 的子节点 → 不抛错，直接把 c 返回（并 warn 一次）；
     - insertBefore(n, ref)：ref 不在 this 下 → 退化成 appendChild。
   正常路径与原生行为完全一致，只有已经不一致时才会走到兜底分支：
   最坏情况是页面上多/少一个块，但不会再把整页顶死。 */
function _poGuard(pd) {
  try {
    var W = (pd && pd.defaultView) || (window.parent && window.parent);
    if (!W || !W.Node || !W.Node.prototype || W.__dshDomGuard) { return; }
    W.__dshDomGuard = true;
    var NP = W.Node.prototype;
    var _rc = NP.removeChild, _ib = NP.insertBefore;
    if (typeof _rc === 'function') {
      NP.removeChild = function (c) {
        try { if (c && c.parentNode === this) { return _rc.call(this, c); } } catch (e) {}
        try { console.warn('[dsh] 跳过一次 removeChild：节点已不在父节点下', c); } catch (e2) {}
        return c;
      };
    }
    if (typeof _ib === 'function') {
      NP.insertBefore = function (n, ref) {
        try { if (!ref || ref.parentNode === this) { return _ib.call(this, n, ref); } } catch (e) {}
        try { return this.appendChild(n); } catch (e3) { return n; }
      };
    }
  } catch (e) {}
}

function _poEnsure() {
  if (PO.ok && PO.el && PO.el.parentNode) { return true; }
  var pd = null;
  try { pd = window.parent && window.parent.document; } catch (e) { pd = null; }
  if (!pd || !pd.body) { PO.ok = false; return false; }
  try {
    _poStyle(pd);
    _poGuard(pd);                              /* 父页面 React removeChild 兜底 */
    var el = pd.getElementById('dsh-dim');     /* Python 侧那层：直接接管，只改文案 */
    if (el) {
      el.style.opacity = '1';                /* 上一轮揭遮罩只是隐藏，这里重新点亮 */
      el.style.visibility = 'visible';
      el.style.pointerEvents = 'auto';
      el.style.zIndex = '2147483000';
      var m = el.querySelector('.dsh-sub');
      if (!m) { m = pd.createElement('div'); m.className = 'dsh-sub'; el.appendChild(m); }
      PO.el = el; PO.msg = m; PO.own = false; PO.ok = true; PO.ever = true;
    } else {
      el = pd.getElementById('dsh-global-ovl');
      if (!el) {
        el = pd.createElement('div');
        el.id = 'dsh-global-ovl';
        el.innerHTML = '<div class="po-ring"></div><div class="po-msg">加载中</div>'
          + '<div class="po-bar"><i></i></div>'
          + '<div class="po-sub" style="margin-top:6px;font-size:22px;letter-spacing:6px;'
          + 'font-weight:700;opacity:.95">让子弹飞一会</div>';
        pd.body.appendChild(el);
      }
      /* 复用上一轮隐藏的自建节点：重新点亮（自建节点始终留在 body 上，不再删除，
         避免任何一次 removeChild 落在 React 正在提交的时间窗口里） */
      el.style.display = 'flex';
      el.style.opacity = '1';
      el.style.visibility = 'visible';
      el.style.pointerEvents = 'auto';
      PO.el = el; PO.msg = el.querySelector('.po-msg'); PO.own = true; PO.ok = true; PO.ever = true;
    }
    gate.classList.remove('on');               /* 单层：父页面遮罩在，就不再用 iframe 内闸门 */
    if (PO.wd) { clearTimeout(PO.wd); }
    PO.wd = setTimeout(function () { _poHide(); }, 240000);   /* 兜底：不永久锁死页面 */
    return true;
  } catch (e) { PO.ok = false; return false; }
}

function _poMsg(txt) {
  if (!txt || txt === PO.text) { return; }
  PO.text = txt;
  if (PO.ok && PO.msg) {
    try { PO.msg.textContent = txt; } catch (e) {}
  }
  try { gateMsg.textContent = txt; } catch (e) {}   /* 同一句也写进内层闸门（降级时看得到） */
}

function _poHide() {
  var el = PO.el, own = PO.own;
  PO.ok = false; PO.el = null; PO.msg = null; PO.text = ''; PO.own = false;
  if (PO.wd) { clearTimeout(PO.wd); PO.wd = 0; }
  if (!el || !el.style) { return; }
  try {
    el.style.opacity = '0';
    el.style.visibility = 'hidden';
    el.style.pointerEvents = 'none';
  } catch (e) {}
  if (own) {
    /* 自建节点只隐藏、不删除：删除动作可能刚好落在 React 提交 DOM 的时间窗口里
       （2026-11-05 的 removeChild 事故），留着隐藏节点下一轮还能直接复用。 */
    setTimeout(function () {
      try { if (el.style) { el.style.display = 'none'; } } catch (e) {}
    }, 320);
  }
  /* Python 写的那层（#dsh-dim）交给 Streamlit 自己清，绝不从 iframe 里动 React 管的节点 */
}

function _poFallback() {                       /* 拿不到父页面：退回 iframe 内 #gate */
  try { gate.classList.add('on'); } catch (e) {}
}

/* ---------- 缩略图预加载缓存（2026-11-04 22:54 需求 B/C）----------
   旧方案把 23 档全分辨率 canvas 位图（DPR 1.5 时每张 ≈2.9 MB，合计 66–80 MB）囤在 JS 堆里，
   Chrome 渲染进程直接 OOM → 整页崩。现在改成"缩略图 / 可见画布预渲染"：
     - 预加载时在**可见画布**上逐档真实绘制（整页被遮罩挡着，用户看不到）；
     - 每档画完立刻压成 1/4 线性分辨率的缩略图（≈0.1 MB/张，23 张 ≈2.5 MB），
       非当前档只留缩略图，全分辨率位图一张都不留；
     - 拖动时先贴缩略图（<0.3 ms，模糊但跟手），松手后再按全分辨率重绘当前档；
     - 内存红线（需求 C）：JS 堆**相对预加载基线**增长 >180 MB 或 canvas 总像素 >40 MB → 清缓存 + 降级直绘。
   "图片准备率"口径不变：统计"真实有位图的缩略图数量"与"图 1 动态层真实渲染档数"的较小值。 */
var THUMB = {}, THUMB_ORDER = [], THUMB_LIMIT = 26;
var THUMB_SCALE = 0.25;            /* 1/4 线性分辨率 = 1/16 像素 */
var HIST_DRAWN_K = -1;             /* 当前可见画布上已经是哪一档的全分辨率图 */
/* 放行闸门的硬指标：图片准备率（真实位图数量，不是数值/结果数量）≥ 90% */
var OUT_READY_MIN = 0.90, OUT_RETRY_MAX = 8;
var DYN_SEEN = {};                 /* 图 1 动态层已真实渲染过的档集合 */
var OUT_TEST_FAIL = false;         /* 仅测试用：模拟位图缓存不可用 */
var MEM_LIMIT = false;             /* 撞上内存红线：停止预加载，明确降级后放行 */
/* 2026-11-05：内存红线改看"增量"而不是绝对值。
   真实 Streamlit 页面本身就有 100–250 MB 的 JS 堆（Streamlit 前端 + 数据表），
   performance.memory.usedJSHeapSize 在 iframe 里读到的是**整个渲染进程**的堆，
   用绝对值 180 MB 判定会在正常页面上误伤（预加载只跑 1 档就降级）。
   基线在 init() 里、预加载真正开始前采集；红线仍是需求 C 的 180 MB/40 MB。 */
var HEAP_BASE = -1;

function _outGet(k) { return THUMB[k] || null; }
function _outCount() { return THUMB_ORDER.length; }
function _outDrop(k) {
  if (THUMB[k]) { delete THUMB[k]; }
  var i = THUMB_ORDER.indexOf ? THUMB_ORDER.indexOf(k) : -1;
  if (i >= 0) { THUMB_ORDER.splice(i, 1); }
}
function _outClear() { THUMB = {}; THUMB_ORDER.length = 0; HIST_DRAWN_K = -1; }

function _thumbPut(k) {
  if (_outGet(k)) { return; }
  if (OUT_TEST_FAIL) { return; }             /* 测试钩子：模拟浏览器禁用 canvas 位图缓存 */
  try {
    if (!ch || !ch.width || !ch.height) { return; }
    var tw = Math.max(24, Math.round(ch.width * THUMB_SCALE));
    var th = Math.max(18, Math.round(ch.height * THUMB_SCALE));
    var c = document.createElement('canvas');
    c.width = tw; c.height = th;
    var x = c.getContext('2d');
    if (!x) { return; }
    x.drawImage(ch, 0, 0, tw, th);
    THUMB[k] = c; THUMB_ORDER.push(k);
    while (THUMB_ORDER.length > THUMB_LIMIT) { delete THUMB[THUMB_ORDER.shift()]; }
  } catch (e) { /* 隐私模式等拿不到 canvas：自动关缓存，不影响功能 */ }
}

/* 内存红线（需求 C）：堆**相对预加载基线**增长 >180 MB 或 canvas 总像素 >40 MB → 清缓存并降级直绘 */
function _heapDelta() {
  var h = _heapMB();
  if (h < 0 || HEAP_BASE < 0) { return -1; }
  return Math.max(0, h - HEAP_BASE);
}
function _canvasPixelsMB() {
  var px = 0, i, t;
  try { px += (cv.width * cv.height) || 0; } catch (e) {}
  try { px += (ch.width * ch.height) || 0; } catch (e) {}
  try { px += (offA.width * offA.height) || 0; } catch (e) {}
  for (i = 0; i < THUMB_ORDER.length; i++) {
    t = THUMB[THUMB_ORDER[i]];
    if (t) { px += (t.width * t.height) || 0; }
  }
  return px * 4 / 1048576;                   /* RGBA 字节 → MB */
}

function _memGuard() {
  var hd = _heapDelta(), pm = _canvasPixelsMB();
  if ((hd >= 0 && hd > 180) || pm > 40) {
    MEM_LIMIT = true;
    DEGRADE = true;
    _outClear();
    return true;
  }
  return false;
}

/* ---------- 图片准备率校验（用户硬要求：必须"有图片"，不是"有数值"，<90% 不放行） ---------- */
function _dynCount() {
  var c = 0, k;
  for (k in DYN_SEEN) { if (DYN_SEEN.hasOwnProperty(k)) { c += 1; } }
  return c;
}
function _outVerify() {
  var have = 0, i, c, total = DATA.ft.length;
  for (i = 0; i < THUMB_ORDER.length; i++) {
    c = THUMB[THUMB_ORDER[i]];
    if (c && c.width > 0 && c.height > 0) { have += 1; }   /* 必须是真的有像素的位图 */
  }
  var dyn = _dynCount();
  return { img_have: have, dyn_have: dyn, total: total,
           ratio: Math.min(have / Math.max(1, total), dyn / Math.max(1, total)),
           mem: MEM_LIMIT, heap_mb: _heapMB(), heap_base: HEAP_BASE,
           heap_delta: _heapDelta(), px_mb: _canvasPixelsMB() };
}
function _readyPctText() {
  var v = _outVerify();
  return '图片准备率 ' + v.img_have + '/' + v.total + '（' + Math.round(100 * v.ratio) + '%）';
}


/* ---------- 两图独立调度：函数图即时 rAF；直方图拖动期间不画、松手后再画 ----------
   关键改动（2026-11-04 晚，用户反馈"快速拖 FT 仍卡"）：
   - 拖动中（pointerdown 直到松手后 240ms）**完全不重画直方图**，只联动图 1 的红线/P50，
     因此拖动路径上的每帧成本恒定在 1~3 ms；
   - 松手（或停顿 240ms）后只画一次直方图，并显示它自己的灰屏"柱状图刷新中"；
   - 全部动作都在 rAF 里合帧，快速拖动也最多每帧一次。 */
var histTimer = 0, histQueued = 0, povlT = 0, evTimes = [];
var PREVIEW_K = -1;                     /* 画布上目前贴的是哪一档缩略图（-1=全分辨率或无） */
var VLINE = { shown: false, x: null };  /* 图1 竖线（P50 穿越）是否真的画出来了，供验收脚本读取 */

/* ---------- 红线矩阵监视器（浏览器侧）----------
   和服务器侧 resource_guard 的"红线矩阵"同构：每帧记录"绘制耗时 / 帧间隔 / JS 堆增长"，
   归一化到各自红线得到 0~1 的负载分。一旦逼近红线（>=0.8）就**强制把整块画布切到
   "加载中…"保护模式**（盖 #gate、暂停一切重绘，动画交给 CSS 合成层），
   等负载回落到红线的 20% 并稳定 0.8s 后才解除——这正是用户要的"向上管理"。 */
var MON = {
  cost_ema: 0.0,     /* 每帧绘制耗时 EMA（ms） */
  gap_ema: 0.0,      /* 相邻 rAF 间隔 EMA（ms） */
  load: 0.0,         /* 0~1 负载分 */
  protect: false,    /* 保护模式 */
  hold: 0, ticks: 0,
  cost_red: 12.0,    /* 红线：单帧绘制 12ms（60fps 预算 16.7ms） */
  gap_red: 60.0,     /* 红线：帧间隔 60ms（掉到 16fps 以下） */
  heap_red: 320.0,   /* 红线：JS 堆相对初始值增长 320MB（拿得到时才算） */
  heap_base: -1,
  t_last: 0, started: false
};
var MON_TICK = 420, PROTECT_ON = 0.95, PROTECT_OFF = 0.20;
MON.grace = 12;       /* 自检结束后先观察 ~5s，再允许触发保护（避免刚加载完误触发） */
MON.cooldown = 0;     /* 保护解除后的冷却周期数 */
MON.sustain = 0;      /* 连续超线周期数 */

function _heapMB() {
  try {
    var m = window.performance && window.performance.memory;
    return (m && m.usedJSHeapSize) ? (m.usedJSHeapSize / 1048576) : -1;
  } catch (e) { return -1; }
}

/* 调试用：整张 canvas 的像素校验和（跨 4 字节取一点，步长必须是 4 的倍数，否则只会采到 alpha 通道） */
function _canvasSig(c) {
  try {
    var im = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    var s = 0, i;
    for (i = 0; i < im.length; i += 4100) { s = (s * 31 + im[i]) % 1000003; }
    return s;
  } catch (e) { return -1; }
}

function _monSnap() {
  var lc = MON.cost_ema / MON.cost_red;
  var lg = MON.gap_ema / MON.gap_red;
  var lh = 0.0, h = _heapMB();
  if (h >= 0 && MON.heap_base >= 0) {
    lh = Math.max(0.0, (h - MON.heap_base) / MON.heap_red);
  }
  MON.load = Math.max(lc, lg, lh);
  return MON.load;
}

function _protectOn(reason) {
  if (MON.protect) { return; }
  MON.protect = true;
  DEGRADE = true;                       /* 解除后也只降级运行 */
  _lockInput(true);                     /* 强制切断人机交互（含进行中的拖动） */
  povl.classList.remove('on');
  hovl.classList.remove('on');
  gateMsg.textContent = '检测到页面负载逼近红线（' + reason + '），已进入保护模式（加载中…），'
      + '已暂停交互，负载回落后自动恢复';
  gate.classList.add('on');             /* 全屏"加载中" */
}

function _protectOff() {
  MON.protect = false;
  MON.hold = 0;
  MON.ticks = 0;
  MON.sustain = 0;
  MON.cooldown = 6;     /* 解除后 6 个周期（≈2.5s）内不再触发，防止反复弹保护 */
  gate.classList.remove('on');
  _lockInput(false);                    /* 恢复交互 */
  /* 恢复后补画一次当前状态 */
  drawDynamic(parseInt(rng.value, 10));
  queueHist(true);
}

function _monTick() {
  if (!MODATA_OK()) { return; }         /* 闸门/自检期间不重复报 */
  if (MON.protect) {
    /* 保护期间没有绘制，帧耗时/帧间隔按空闲衰减，代表"页面正在恢复" */
    MON.cost_ema = (MON.cost_ema || 0) * 0.35;
    MON.gap_ema = (MON.gap_ema || 0) * 0.35;
  } else if (MON.grace > 0) {
    MON.grace -= 1;
    return;                              /* 宽容期：自检刚结束，先观察，不触发保护 */
  } else if (MON.cooldown > 0) {
    MON.cooldown -= 1;
    MON.cost_ema = (MON.cost_ema || 0) * 0.7;
    MON.gap_ema = (MON.gap_ema || 0) * 0.7;
    return;                              /* 冷却期：解除保护后的降速窗口 */
  }
  var l = _monSnap();
  if (!MON.protect) {
    if (l >= PROTECT_ON) {
      MON.sustain += 1;
      if (MON.sustain >= 3) {            /* 连续 3 个周期（≈1.3s）超线才动手 */
        MON.sustain = 0;
        _protectOn('负载分 ' + l.toFixed(2));
      }
    } else {
      MON.sustain = 0;
    }
  } else {
    if (l <= PROTECT_OFF) {
      MON.hold += 1;
      if (MON.hold >= 5) { _protectOff(); }  /* 稳定 ≈2.1s 后才解除 */
    } else {
      MON.hold = 0;
    }
  }
}

function MODATA_OK() { return selfTestDone; }

function _monReset() {
  MON.cost_ema = 0.0; MON.gap_ema = 0.0; MON.load = 0.0;
  MON.sustain = 0; MON.hold = 0; MON.cooldown = 0;
  MON.grace = 12;
  MON.heap_base = _heapMB();
  MON.t_last = 0;
}

function _monStart() {
  if (MON.started) { return; }
  MON.started = true;
  MON.heap_base = _heapMB();
  var loop = function () {
    try { _monTick(); } catch (e) {}
    setTimeout(loop, MON_TICK);
  };
  setTimeout(loop, MON_TICK);
}

/* ---------- 人机交互闸门（用户要求：一旦进入加载状态，必须切断交互，进行中的也要强制停止） ----------
   humanLocked=true 时：
     - 输入滑条 input/change/pointer/touch 全部在捕获阶段拦掉；
     - 正在进行的拖动立即作废（dragging=false / 定时器清空 / rAF 取消 / 滑条失焦）；
     - 两卡 pointer-events:none、滑条 disabled —— 彻底不产生新事件，CPU 不会再被打满。 */
var humanLocked = false;

function _lockInput(on) {
  humanLocked = !!on;
  try { rng.disabled = !!on; } catch (e) {}
  if (!document.body || !document.body.classList) { return; }
  if (on) {
    document.body.classList.add('locked');
    dragging = false;
    if (dragTimer) { clearTimeout(dragTimer); dragTimer = 0; }
    if (histTimer) { clearTimeout(histTimer); histTimer = 0; }
    if (povlT) { clearTimeout(povlT); povlT = 0; }
    histQueued = 0; needPaint = false;
    povl.classList.remove('on');
    hovl.classList.remove('on');
    if (rafId) {
      try { (window.cancelAnimationFrame || function () {})(rafId); } catch (e) {}
      rafId = 0;
    }
    try { rng.blur(); } catch (e) {}
  } else {
    document.body.classList.remove('locked');
    rng.disabled = false;
  }
}

/* 捕获阶段拦截一切指针/键盘/输入事件（iframe 内 document 级，不会漏到滑条上） */
(function () {
  var names = ['pointerdown', 'pointermove', 'pointerup', 'pointercancel',
               'mousedown', 'mousemove', 'mouseup',
               'touchstart', 'touchmove', 'touchend',
               'wheel', 'keydown', 'keyup', 'input', 'change', 'click'];
  var block = function (e) {
    if (!humanLocked) { return; }
    try { e.stopPropagation(); } catch (err) {}
    try { e.stopImmediatePropagation(); } catch (err) {}
    if (e.preventDefault) { try { e.preventDefault(); } catch (err) {} }
  };
  if (document.addEventListener) {
    for (var i = 0; i < names.length; i++) {
      document.addEventListener(names[i], block, { capture: true, passive: false });
    }
  }
})();

function markDragging() {
  if (humanLocked || MON.protect) { return; }
  dragging = true;
  if (dragTimer) { clearTimeout(dragTimer); }
  dragTimer = setTimeout(function () {
    dragTimer = 0; dragging = false; queueHist(true);
  }, 240);
}

function onInput() {
  if (humanLocked) { return; }             /* 加载/保护状态：直接丢弃输入 */
  if (!selfTestDone) { return; }          /* 自检期间不接受输入（闸门还盖着） */
  var now = (window.performance && performance.now) ? performance.now() : Date.now();
  evTimes.push(now);
  while (evTimes.length && now - evTimes[0] > 600) { evTimes.shift(); }
  if (evTimes.length > 24) {              /* 极端快速拖动：图 1 自己也短暂降级提示 */
    povl.classList.add('on');
    if (povlT) { clearTimeout(povlT); }
    povlT = setTimeout(function () { povlT = 0; povl.classList.remove('on'); }, 420);
  }
  markDragging();
  schedule();
}

function queueHist(force) {
  if (humanLocked) { return; }
  if (!selfTestDone) { return; }
  var kk = parseInt(rng.value, 10);
  if (previewHist(kk)) {                  /* 有 1/4 缩略图：拖动中也立刻贴（<0.3ms） */
    if (histTimer) { clearTimeout(histTimer); histTimer = 0; }
    histQueued = 0;
    hovl.classList.remove('on');
    if (force || !dragging) { drawHist(kk); }   /* 松手/停顿后补一张全分辨率，保证看得清 */
    return;
  }
  if (dragging && !force) { return; }     /* 连缩略图都没有的档：拖动期间不画（保丝滑） */
  histQueued += 1;
  if (histQueued >= 2) { hovl.classList.add('on'); }
  if (histTimer) { return; }
  histTimer = setTimeout(function () {
    histTimer = 0;
    drawHist(parseInt(rng.value, 10));
    histQueued = 0;
    hovl.classList.remove('on');
  }, DEGRADE ? 180 : 110);
}

function schedule() {
  if (humanLocked) { return; }             /* 加载/保护状态：不再排队任何重绘 */
  needPaint = true;
  if (rafId) { return; }
  var raf = window.requestAnimationFrame || function (f) { return setTimeout(f, 16); };
  rafId = raf(function () {
    rafId = 0;
    var now = _now();
    if (MON.t_last) {
      var gap = now - MON.t_last;
      /* 降速：EMA 系数 0.12；>350ms 视为切页/长任务恢复，不计入，避免误判 */
      if (gap > 0 && gap <= 350) {
        MON.gap_ema = MON.gap_ema ? (MON.gap_ema * 0.88 + gap * 0.12) : gap;
      }
    }
    MON.t_last = now;
    if (!needPaint) { return; }
    needPaint = false;
    if (!selfTestDone || humanLocked) { return; }
    if (MON.protect) { return; }            /* 保护模式：暂停一切重绘 */
    var k = parseInt(rng.value, 10);
    var t0 = _now();
    drawDynamic(k);                         /* 函数图：红线 / P50 即时联动 */
    var cost = Math.min(40, _now() - t0);
    MON.cost_ema = MON.cost_ema ? (MON.cost_ema * 0.88 + cost * 0.12) : cost;
    queueHist(false);                       /* 缓存命中会立即贴图；未命中在拖动中自动跳过 */
  });
}

function relayout() {
  var newW = Math.max(320, wrap.clientWidth || 800);
  var changed = Math.abs(newW - W) > 2;
  W = newW; HIST_W = newW;
  if (changed) { _outClear(); }            /* 画布像素尺寸变了，旧位图缓存作废 */
  setupCanvas(cv, ctx, PRED_H);
  setupCanvas(offA, actx, PRED_H);
  setupCanvas(ch, hctx, HIST_H);
  if (changed || true) { buildStatic(); }
  drawDynamic(parseInt(rng.value, 10));
  drawHist(parseInt(rng.value, 10));
}

function _now() {
  return (window.performance && window.performance.now) ? window.performance.now() : Date.now();
}

/* 内部自检（用户要求：预测结束后先自己跑一下，确认快速拖动不崩，再让用户接管）：
   把整圈 FT 快速扫一遍（相当于模拟快速拖动），逐帧计时；无异常且每帧够快才揭开闸门。
   自检期间闸门盖着，屏幕继续显示"预测完成 · 内部自检中"，用户无法操作。 */
var ST = { i: 0, runs: 0, tot: 0, times: [], err: null, phase: 0, retry: 0 };

/* 用户要求（m04702）：先说"预计耗时两分钟"，在这段时间里把每一档 FT 的**图片**都提前做好；
   放行闸门的硬指标是"图片准备率 ≥ 90%"（统计内置 output 里真实可用的位图数量，不是数值/结果数量）。
   分两阶段：
     阶段 0（预加载图片）：逐档 drawDynamic + drawHist，直方图立即进 output 位图缓存；
                          图片准备率不足 90% 就继续补（最多 8 轮），**绝不放行**；
     阶段 1（拖动自检）：随机 8 档再画一遍（应全部命中缓存），确认拖动路径足够快。
   每片只画 3 档就让出主线程（setTimeout 0），CSS 动画（灰点/加载圈）永远在动。 */
function selfTest() {
  var n = DATA.ft.length;
  ST.i = 0; ST.runs = n; ST.tot = 0; ST.times = []; ST.err = null;
  ST.phase = 0; ST.retry = 0;
  /* 单层遮罩（需求 A）：优先接管父页面那层（点预测时就在，只改文案、不闪）；
     拿不到父页面就退回 iframe 内 #gate，保证用户始终点不动。 */
  _poEnsure();
  if (!PO.ok) { _poFallback(); }
  _poMsg('预计耗时两分钟 · 正在把每一档 FT 的图片渲染进缩略图缓存（' + _readyPctText() + '）');
  try {
    drawDynamic(parseInt(rng.value, 10));
    drawHist(parseInt(rng.value, 10));
  } catch (e) { ST.err = e; }
  _stChunk();
}

function _stChunk() {
  var n = DATA.ft.length, c, k, t0, dt;
  if (!ST.err && !MEM_LIMIT) {
    try {
      if (ST.phase === 0) {
        /* 阶段 0（预加载图片）：逐档在**可见画布**上真实绘制（整页被遮罩盖着，用户看不到），
           画完立刻压成 1/4 缩略图；不留任何全分辨率位图。每片 3 档就让出主线程。 */
        for (c = 0; c < 3 && ST.i < ST.runs; c++) {
          if (_memGuard()) { break; }     /* 需求 C：逼近内存红线就停手，降级放行 */
          k = ST.i; ST.i += 1;
          if (_outGet(k) && DYN_SEEN[k]) { continue; }
          drawDynamic(k);
          drawHist(k);
        }
      } else if (ST.phase === 1) {
        /* 阶段 1（拖动自检）：走一遍真实拖动路径 —— 贴缩略图 + 松手补全分辨率，逐档计时 */
        for (c = 0; c < 4 && ST.i < ST.runs; c++) {
          k = Math.round((ST.i * (n - 1)) / Math.max(1, ST.runs - 1));
          t0 = _now();
          drawDynamic(k);
          previewHist(k);                 /* 拖动中：贴 1/4 缩略图 */
          drawHist(k);                    /* 松手后：补一张全分辨率 */
          dt = _now() - t0;
          ST.tot += dt; ST.times.push(dt);
          ST.i += 1;
        }
      }
    } catch (e) { ST.err = e; }
  }
  if (!ST.err && !MEM_LIMIT && ST.i < ST.runs) {
    if (ST.phase === 0) {
      _poMsg('预计耗时两分钟 · 正在把每一档 FT 的图片渲染进缩略图缓存（' + _readyPctText() + '）');
    } else {
      _poMsg('预计耗时两分钟 · 图片已备好，正在做拖动自检（' + ST.i + '/' + ST.runs + '）');
    }
    setTimeout(_stChunk, 0);
    return;
  }
  if (!ST.err && !MEM_LIMIT && ST.phase === 0) {
    var v0 = _outVerify();
    if (v0.ratio < OUT_READY_MIN && ST.retry < OUT_RETRY_MAX) {
      ST.retry += 1;
      ST.i = 0; ST.runs = n;
      _poMsg(_readyPctText() + ' · 低于 90%，正在补齐缺失档（第 ' + ST.retry + ' 轮）…');
      setTimeout(_stChunk, 160);
      return;
    }
    ST.phase = 1; ST.i = 0; ST.runs = 8; ST.tot = 0; ST.times = [];
    _poMsg('预计耗时两分钟 · 图片已备好，正在做拖动自检…');
    setTimeout(_stChunk, 0);
    return;
  }
  _stFinish();
}

/* 唯一的揭遮罩入口：父页面那层隐藏（自己建的直接删），内层 #gate 一定摘掉 */
function _reveal(delay) {
  setTimeout(function () {
    if (PO.ok) { _poHide(); }
    try { gate.classList.remove('on'); } catch (e) {}
  }, delay || 0);
}

function _stFinish() {
  var err = ST.err, runs = ST.runs, tot = ST.tot, times = ST.times.slice(0);
  var avg = tot / Math.max(1, runs);
  times.sort(function (a, b) { return a - b; });
  var p90 = times.length ? times[Math.min(times.length - 1, Math.floor(times.length * 0.9))] : 0;
  var worst = times.length ? times[times.length - 1] : 0;
  var v = _outVerify();
  var ok_img = v.ratio >= OUT_READY_MIN;
  drawDynamic(parseInt(rng.value, 10));
  drawHist(parseInt(rng.value, 10));
  if (!err && !ok_img && ST.retry < OUT_RETRY_MAX && !MEM_LIMIT) {
    /* 图片准备率不足 90%：遮罩不放行，继续补齐缺失档 */
    ST.retry += 1; ST.phase = 0; ST.i = 0; ST.runs = DATA.ft.length;
    _poMsg(_readyPctText() + ' · 未到 90%，遮罩不放行，正在继续补齐（第 ' + ST.retry + ' 轮）…');
    setTimeout(_stChunk, 200);
    return;
  }
  if (MEM_LIMIT) {
    /* 需求 C：撞上内存红线 —— 明确降级，绝不让浏览器崩 */
    DEGRADE = true;
    var _hd = v.heap_delta, _hm;
    if (_hd >= 0) { _hm = '增量 ' + Math.round(_hd) + ' MB（基线 ' + Math.round(v.heap_base) + ' MB）'; }
    else if (v.heap_mb >= 0) { _hm = '当前 ' + Math.round(v.heap_mb) + ' MB'; }
    else { _hm = '不可读'; }
    _poMsg('已接近浏览器内存红线（JS 堆 ' + _hm + ' / 画布 ' + v.px_mb.toFixed(1)
        + ' MB），为保证不崩溃已停手：拖动时贴缩略图，松手后再补全分辨率');
    _reveal(1200);
  } else if (err) {
    DEGRADE = true;
    _poMsg('图表预加载出现异常，已降级：拖动时只联动红线，松手后再刷新直方图');
    _reveal(900);
  } else if (!ok_img) {
    /* 位图缓存彻底不可用（极少见，例如浏览器禁止创建 canvas）：明确告知降级后放行 */
    DEGRADE = true;
    _poMsg(_readyPctText() + ' · 浏览器无法保存图片缓存，已降级为直绘模式（可能略慢）');
    _reveal(1400);
  } else if (p90 >= 40 || avg >= 16 || worst >= 45) {
    DEGRADE = true;
    _poMsg(_readyPctText() + ' · 自检偏慢（平均 ' + avg.toFixed(1) + ' ms），已降级：拖动时只联动红线');
    _reveal(900);
  } else {
    _poMsg(_readyPctText() + ' · 全部图片已就绪，现在随便拖吧');
    _reveal(420);
  }
  selfTestDone = true;                      /* 图片已≥90%：之后拖动都是贴缩略图，不再有第一次卡顿 */
  _monStart();                              /* 负载监视仅作最后兜底 */
  _monReset();                              /* 刚交接时清空 EMA/加长宽容期，避免被保护模式二次冻结 */
  _lockInput(false);
}

function init() {
  var n = DATA.ft.length;
  rng.min = 0; rng.max = n - 1; rng.step = 1;
  var k0 = 0, best = 1e9, i;
  for (i = 0; i < n; i++) {
    var d = Math.abs(DATA.ft[i] - DATA.ft_default);
    if (d < best) { best = d; k0 = i; }
  }
  rng.value = k0;
  _poEnsure();                             /* 单层遮罩（需求 A）：先接管父页面那层，再锁交互 */
  if (!PO.ok) { _poFallback(); }           /* 拿不到父页面：退回 iframe 内闸门 */
  _lockInput(true);                        /* 遮罩/自检期间彻底切断交互 */

  try {
    window.__lc = {
      verify: function () { return _outVerify(); },
      count: function () { return _outCount(); },
      ready: function () { return selfTestDone; },
      gateOn: function () { return !!(gate && gate.classList && gate.classList.contains) ? gate.classList.contains('on') : false; },
      msg: function () { return gateMsg ? gateMsg.textContent : ''; },
      fail: function (on) { OUT_TEST_FAIL = !!on; },
      err: function () { return ST.err ? ('' + ST.err) : ''; },
      retry: function () { return ST.retry; },
      poOk: function () { return !!PO.ok; },
      poUsed: function () { return !!PO.ever; },
      poMsg: function () { return PO.text; },
      thumb: function () { return _outCount(); },
      heap: function () { return _heapMB(); },
      heapBase: function () { return HEAP_BASE; },
      heapDelta: function () { return _heapDelta(); },
      px: function () { return _canvasPixelsMB(); },
      mem: function () { return MEM_LIMIT; },
      reveal: function () { _reveal(0); },
      histK: function () { return HIST_DRAWN_K; },
      prevK: function () { return PREVIEW_K; },
      vline: function () { return { shown: VLINE.shown, x: VLINE.x }; },
      sig: function () { return _canvasSig(ch); },
      probe: function (k) { rng.value = k; drawDynamic(k); drawHist(k); return _histTip(k); }
    };
  } catch (e) {}
  rng.addEventListener('input', onInput);
  rng.addEventListener('change', function () { dragging = false; queueHist(true); });
  if (rng.addEventListener) {
    rng.addEventListener('pointerdown', markDragging, { passive: true });
    rng.addEventListener('touchstart', markDragging, { passive: true });
  }
  var _release = function () { if (humanLocked) { return; } dragging = false; queueHist(true); };
  if (window.addEventListener) {
    window.addEventListener('pointerup', _release, { passive: true });
    window.addEventListener('mouseup', _release, { passive: true });
    window.addEventListener('touchend', _release, { passive: true });
    var tmr = 0;
    window.addEventListener('resize', function () {
      if (tmr) { clearTimeout(tmr); }
      tmr = setTimeout(function () { tmr = 0; relayout(); }, 120);
    });
  }
  HEAP_BASE = _heapMB();                   /* 预加载基线：红线只看本次预加载造成的增量 */
  relayout();
  setTimeout(selfTest, 90);                /* 先让闸门画出来，再做内部试跑 */
}
init();
</script></body></html>""")

_LOADING = string.Template("""<!DOCTYPE html><html lang="zh-CN" translate="no" class="notranslate"><head><meta charset="utf-8">
<meta name="google" content="notranslate"><meta name="microsoft" content="notranslate">
<style>
  body { margin:0; background:$BG; color:$SUBFG; height:100%;
         font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif; }
  .box { display:flex; align-items:center; justify-content:center; height:160px;
         font-size:14.5px; letter-spacing:1px; }
  .dots { display:inline-block; width:22px; text-align:left; }
  @keyframes blink { 0%, 20% {opacity:1;} 50% {opacity:0.15;} 80%, 100% {opacity:1;} }
  .dots span { animation: blink 1.4s infinite; }
  .dots span:nth-child(2) { animation-delay: 0.2s; }
</style></head><body>
<div class="box"><span>正在准备交互画布，加载中</span>
  <span class="dots"><span>·</span><span>·</span><span>·</span></span></div>
</body></html>""")


def component_height() -> int:
    """两张卡片 + 边距后的 iframe 高度（比纯画布高约 210px）。"""
    return PRED_H + HIST_H + 230


def loading_html(theme=None) -> str:
    c = pick_theme(theme)
    return _LOADING.substitute(BG=c["bg"], SUBFG=c["subfg"])


def build_html(live: dict, ft_default: float | None = None, theme=None,
               render_scale: float | None = None) -> str:
    """生成自包含的交互画布 HTML（结构 A 离屏缓存 + 结构 B 本地重绘）。

    render_scale：
        None / <=0  → 自动（按设备像素比，封顶 1.5×）
        1.5 / 2.0 等 → 固定倍率（用户在侧栏「画布分辨率」里选的）
    只影响画布像素密度，不影响预测结果与指标；调低可显著减少画布内存与填充耗时。
    """
    c = pick_theme(theme)
    try:
        _rs = float(render_scale) if render_scale is not None else -1.0
    except (TypeError, ValueError):
        _rs = -1.0
    if not np.isfinite(_rs) or _rs <= 0:
        _rs = -1.0
    else:  # 合理范围内夹一下，避免 1e9 这种离谱倍率把画布撑爆
        _rs = float(np.clip(_rs, 0.5, 2.0))
    data = {k: v for k, v in live.items() if k != "theme"}
    if ft_default is not None:
        data["ft_default"] = float(ft_default)
    # 只传前端用到的字段（去掉内部用的大数组）
    keep = ("t0", "u_eq", "n_paths", "x_lo", "x_hi", "y_lo", "y_hi",
            "hist_segs", "det_segs", "hx0", "hx1", "edges", "ft", "uth", "cross",
            "p5", "p25", "p50", "p75", "p95", "cens", "rel", "cnt", "share",
            "n_paths_valid", "ft_default", "bin_w")
    data = {k: data[k] for k in keep if k in data}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    dark = str(c["bg"]).lower() in ("#0e1117", "#0d1117", "#000000") or         float(sum(int(str(c["bg"]).lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))) < 220
    card = "#151b24" if dark else "#fbfcfe"
    border = "#263243" if dark else "#dfe7f1"
    return _HTML.substitute(
        BG=c["bg"], FG=c["fg"], SUBFG=c["subfg"], RED=c["red"],
        CARD=card, BORDER=border,
        DATA=js, COL=json.dumps(c, ensure_ascii=False),
        PRED_H=str(PRED_H), HIST_H=str(HIST_H),
        RSCALE=repr(_rs),
    )

