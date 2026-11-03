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
from typing import Iterable

import numpy as np
import pandas as pd

import rul_pipeline as rp

FT_MIN, FT_MAX, FT_STEP = 3.0, 8.0, 0.1
FT_SPAN_MAX = 2.2          # 滑条最多覆盖 2.2 个百分点（避免低衰减档被拉得过于稀疏）
BIN_TARGET = 120           # 直方图目标柱数（用于自动选好看的分箱宽度）
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


def _downsample_view(prep: dict, window_h: float = HIST_WIN_H, target: int = 900):
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
            cur = [[r[0][0], r[0][1]]]
        elif abs(cur[-1][0] - r[0][0]) > 1e-9 or abs(cur[-1][1] - r[0][1]) > 1e-9:
            if len(cur) > 1:
                segs.append(cur)
            cur = [[r[0][0], r[0][1]]]
        if abs(cur[-1][0] - r[1][0]) > 1e-9 or abs(cur[-1][1] - r[1][1]) > 1e-9:
            cur.append([r[1][0], r[1][1]])
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
    with np.errstate(all="ignore"):
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
    th, uh = _downsample_view(prep)
    det = np.asarray(mc["det_curve"], float)
    t_ext = t0 + np.arange(len(det)) * win

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
        hx0=0.0, hx1=float(x_hi_t), edges=[float(v) for v in edges],
        ft=[float(v) for v in sub["grid"]],
        uth=[float(v) for v in sub["uths"]],
        p5=[float(v) for v in sub["p5"]], p25=[float(v) for v in sub["p25"]],
        p50=[float(v) for v in sub["p50"]], p75=[float(v) for v in sub["p75"]],
        p95=[float(v) for v in sub["p95"]],
        cens=[float(v) for v in sub["cens"]],
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
<html lang="zh-CN"><head><meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: $BG; color: $FG;
         font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; }
  #wrap { width: 100%; }
  .hdr { font-size: 14.5px; font-weight: 600; margin: 2px 0 4px 2px; color: $FG; }
  .ctl { display: flex; align-items: center; gap: 10px; margin: 8px 2px 6px 2px; }
  .ctl input[type=range] { flex: 1; accent-color: $RED; height: 18px; }
  .ctl .val { font-size: 13px; font-weight: 700; color: $RED; min-width: 86px; text-align: right; }
  .ctl .lab { font-size: 12.5px; color: $SUBFG; white-space: nowrap; }
  .stat { font-size: 12.5px; color: $SUBFG; margin: 0 2px 8px 2px; }
  canvas { display: block; width: 100%; }
  .loading { display:flex; align-items:center; justify-content:center; height:120px;
             font-size:14px; color:$SUBFG; letter-spacing:1px; }
  .dots { display:inline-block; width: 26px; text-align:left; }
</style></head>
<body><div id="wrap">
  <div class="hdr">预测曲线（拖动FT滑条：红线 / P50 / 直方图 全部本地即时联动）</div>
  <canvas id="cv"></canvas>
  <div class="ctl">
    <span class="lab">FT（失效阈值 %）</span>
    <input id="rng" type="range" min="0" max="1" step="1" value="0">
    <span class="val" id="ftv">--</span>
  </div>
  <div class="stat" id="stat">--</div>
  <div class="hdr">当前 FT 下的 RUL 概率分布（柱高按该档峰值归一；悬停查看路径数）</div>
  <canvas id="ch"></canvas>
  <div class="stat" id="tip">&nbsp;</div>
</div>
<script>
var DATA = $DATA;
var COL = $COL;

var PRED_H = $PRED_H, HIST_H = $HIST_H, ML = 64, MR = 22, MT = 30, MB = 46;
var HIST_ML = 64, HIST_MR = 22, HIST_MT = 26, HIST_MB = 46;

var wrap = document.getElementById('wrap');
var cv = document.getElementById('cv'), ctx = cv.getContext('2d');
var ch = document.getElementById('ch'), hctx = ch.getContext('2d');
var rng = document.getElementById('rng');
var ftv = document.getElementById('ftv');
var stat = document.getElementById('stat');
var tip = document.getElementById('tip');

var offA = document.createElement('canvas'), actx = offA.getContext('2d');
var W = 800, HIST_W = 800, DPR = Math.max(1, window.devicePixelRatio || 1);
var curK = 0, rafId = 0, needPaint = false;

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
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, PRED_H);
  ctx.drawImage(offA, 0, 0, W, PRED_H);      // 贴回结构 A 的静态底图（<1ms）

  var ft = DATA.ft[k], uth = DATA.uth[k], p50 = DATA.p50[k];
  var ythr = DATA.u_eq * (1 - ft / 100);
  if (ythr >= DATA.y_lo && ythr <= DATA.y_hi) {
    ctx.strokeStyle = COL.red; ctx.lineWidth = 2.4; ctx.setLineDash([8, 5]);
    ctx.beginPath(); ctx.moveTo(ML, ty(ythr)); ctx.lineTo(W - MR, ty(ythr)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COL.red; ctx.font = 'bold 12.5px "Microsoft YaHei", Arial, sans-serif';
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText('FT = ' + fmt(ft, 1) + '%   阈值 ' + fmt(ythr, 4) + ' V', W - MR - 6, ty(ythr) - 6);
  }
  if (!isNaN(p50)) {
    var xp = DATA.t0 + p50;
    if (xp >= DATA.x_lo && xp <= DATA.x_hi) {
      ctx.strokeStyle = COL.red; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(tx(xp), MT); ctx.lineTo(tx(xp), PRED_H - MB); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COL.red; ctx.font = '12.5px "Microsoft YaHei", Arial, sans-serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('P50 穿越 ≈ ' + fmt(p50, 0) + ' h', tx(xp) + 5, PRED_H - MB - 6);
    }
  }
  ftv.textContent = fmt(ft, 1) + '%';
  stat.textContent = '当前 FT = ' + fmt(ft, 1) + '%   阈值 ' + fmt(uth, 4) + ' V   ｜   '
      + 'P5 ' + fmt(DATA.p5[k], 0) + ' h · P50 ' + fmt(DATA.p50[k], 0) + ' h · P95 '
      + fmt(DATA.p95[k], 0) + ' h   ｜   删失 ' + fmt(DATA.cens[k], 1) + '%';
}

/* ---------- 直方图（每次拖动重画，但只有 ~120 根柱） ---------- */
function drawHist(k) {
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
  tip.textContent = '当前档有效路径 ' + DATA.n_paths_valid[k] + ' / ' + DATA.n_paths
      + '（' + fmt(100 - DATA.cens[k], 1) + '% 在 3000 h 内穿越）｜ 分箱宽度 ' + fmt(DATA.bin_w, 0) + ' h';
}

function paint(k) { curK = k; drawDynamic(k); drawHist(k); }

function schedule() {
  needPaint = true;
  if (rafId) { return; }
  var raf = window.requestAnimationFrame || function (f) { return setTimeout(f, 16); };
  rafId = raf(function () { rafId = 0; if (needPaint) { needPaint = false; paint(parseInt(rng.value, 10)); } });
}

function relayout() {
  var newW = Math.max(320, wrap.clientWidth || 800);
  var changed = Math.abs(newW - W) > 2;
  W = newW; HIST_W = newW;
  setupCanvas(cv, ctx, PRED_H);
  setupCanvas(offA, actx, PRED_H);
  setupCanvas(ch, hctx, HIST_H);
  if (changed || true) { buildStatic(); }
  paint(parseInt(rng.value, 10));
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
  rng.oninput = schedule;
  rng.onchange = schedule;
  relayout();
  if (window.addEventListener) {
    var tmr = 0;
    window.addEventListener('resize', function () {
      if (tmr) { clearTimeout(tmr); }
      tmr = setTimeout(function () { tmr = 0; relayout(); }, 120);
    });
  }
}
init();
</script></body></html>""")

_LOADING = string.Template("""<!DOCTYPE html><html><head><meta charset="utf-8">
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


def loading_html(theme=None) -> str:
    c = pick_theme(theme)
    return _LOADING.substitute(BG=c["bg"], SUBFG=c["subfg"])


def build_html(live: dict, ft_default: float | None = None, theme=None) -> str:
    """生成自包含的交互画布 HTML（结构 A 离屏缓存 + 结构 B 本地重绘）。"""
    c = pick_theme(theme)
    data = {k: v for k, v in live.items() if k != "theme"}
    if ft_default is not None:
        data["ft_default"] = float(ft_default)
    # 只传前端用到的字段（去掉内部用的大数组）
    keep = ("t0", "u_eq", "n_paths", "x_lo", "x_hi", "y_lo", "y_hi",
            "hist_segs", "det_segs", "hx0", "hx1", "edges", "ft", "uth",
            "p5", "p25", "p50", "p75", "p95", "cens", "rel", "cnt", "share",
            "n_paths_valid", "ft_default", "bin_w")
    data = {k: data[k] for k in keep if k in data}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return _HTML.substitute(
        BG=c["bg"], FG=c["fg"], SUBFG=c["subfg"], RED=c["red"],
        DATA=js, COL=json.dumps(c, ensure_ascii=False),
        PRED_H=str(PRED_H), HIST_H=str(HIST_H),
    )
