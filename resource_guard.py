# -*- coding: utf-8 -*-
"""资源红线矩阵 + 加载闸门（"向上管理"用户操作节奏）。

设计目标（用户 2026-11-04 需求）：
- 终端崩溃/浏览器无响应的根源是"高危操作叠加过快"（上传/解析/预测/画布重建）。
- 本模块把系统负载量化成一个 0~1 的“负载分”：
      load = max( RSS 增量 / (红线RSS - 空闲RSS),
                  CPU 占用 / 红线CPU,
                  线程数  / 红线线程 )
  红线 = 1.0。
- 高危操作开始时显示全屏遮罩；操作结束后**不立即解除**，而是持续采样，
  只有当负载分降到“红线的 20%”（0.2）或超时（默认 6 s）后才解除遮罩。
- 所有采样写 `终端资源.log`，便于排查“到底是谁把内存/CPU 拉爆”。
"""
from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil 缺失时退化
    psutil = None  # type: ignore

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "终端资源.log"

# ---------------- 红线矩阵（可按机器调整） ----------------
RED = {
    "rss_mb": 3000.0,     # 进程常驻内存红线（MB）；超过这里浏览器/系统开始吃紧
    "cpu_pct": 85.0,      # 进程 CPU 红线（%）
    "threads": 128,       # 线程数红线
    "ops_10s": 2.0,       # 10 秒内"高危操作"次数红线（这才是浏览器卡死的主因）
}
RELEASE_RATIO = 0.20      # 解除加载的阈值：红线的 20%
OPS_WINDOW_S = 10.0
IDLE = {"rss_mb": None, "cpu_pct": 0.0, "threads": 1}  # 空闲基线，首次采样时记录


def note_op(kind: str = "高危操作") -> None:
    """记一次高危操作（上传/开始预测/切数据），用于统计操作频率。"""
    try:
        import streamlit as st
        ops = st.session_state.setdefault("__rg_ops", [])
        ops.append([time.time(), kind])
        st.session_state["__rg_ops"] = ops[-40:]
        log_event(kind, note="op")
    except Exception:
        pass


def ops_in_window(window: float = OPS_WINDOW_S) -> int:
    try:
        import streamlit as st
        ops = st.session_state.get("__rg_ops") or []
        now = time.time()
        return sum(1 for t, _k in ops if now - float(t) <= window)
    except Exception:
        return 0


def _proc():
    if psutil is None:
        return None
    try:
        return psutil.Process()
    except Exception:
        return None


def sample(cpu_interval: float = 0.0) -> dict:
    """采样一次负载，返回原始值 + 负载分。"""
    p = _proc()
    if p is None:  # 退化：没有 psutil 时只报告未知
        return {"rss_mb": 0.0, "cpu_pct": 0.0, "threads": 0, "load": 0.0,
                "available": False, "ts": time.time()}
    try:
        rss = p.memory_info().rss / 1048576.0
    except Exception:
        rss = 0.0
    try:
        cpu = float(p.cpu_percent(interval=cpu_interval))
    except Exception:
        cpu = 0.0
    try:
        th = int(p.num_threads())
    except Exception:
        th = 1
    if IDLE["rss_mb"] is None:
        IDLE["rss_mb"] = max(rss, 1.0)
    idle_rss = float(IDLE["rss_mb"])
    span = max(RED["rss_mb"] - idle_rss, 64.0)
    l_rss = max(0.0, (rss - idle_rss) / span)
    l_cpu = max(0.0, cpu / RED["cpu_pct"])
    l_th = max(0.0, th / RED["threads"])
    n_ops = ops_in_window()
    l_ops = max(0.0, n_ops / RED["ops_10s"])
    load = max(l_rss, l_cpu, l_th, l_ops)
    return {"rss_mb": rss, "cpu_pct": cpu, "threads": th, "load": load,
            "l_rss": l_rss, "l_cpu": l_cpu, "l_thread": l_th, "l_ops": l_ops,
            "ops_10s": n_ops,
            "available": True, "ts": time.time(),
            "idle_rss_mb": idle_rss}


def matrix_rows(s: Optional[dict] = None) -> list[dict]:
    """给界面展示用的“红线矩阵”行。"""
    s = s or sample()
    ratio = RELEASE_RATIO
    rows = [
        {"资源": "进程内存 RSS", "当前": f"{s['rss_mb']:.0f} MB",
         "红线": f"{RED['rss_mb']:.0f} MB",
         f"释放线(×{ratio:.0%})": f"{RED['rss_mb'] * ratio:.0f} MB",
         "负载分": f"{s.get('l_rss', 0):.2f}"},
        {"资源": "进程 CPU", "当前": f"{s['cpu_pct']:.0f}%",
         "红线": f"{RED['cpu_pct']:.0f}%",
         f"释放线(×{ratio:.0%})": f"{RED['cpu_pct'] * ratio:.0f}%",
         "负载分": f"{s.get('l_cpu', 0):.2f}"},
        {"资源": "线程数", "当前": f"{s['threads']}",
         "红线": f"{RED['threads']}",
         f"释放线(×{ratio:.0%})": f"{RED['threads'] * ratio:.0f}",
         "负载分": f"{s.get('l_thread', 0):.2f}"},
        {"资源": "高危操作频率(10s)", "当前": f"{s.get('ops_10s', 0)} 次",
         "红线": f"{RED['ops_10s']:.0f} 次",
         f"释放线(×{ratio:.0%})": f"{RED['ops_10s'] * ratio:.1f} 次",
         "负载分": f"{s.get('l_ops', 0):.2f}"},
    ]
    return rows


def summary_text(s: Optional[dict] = None) -> str:
    s = s or sample()
    return (f"负载 {s['load']:.2f}（红线 1.00，释放线 {RELEASE_RATIO:.2f}）"
            f" ｜ RSS {s['rss_mb']:.0f} MB ｜ CPU {s['cpu_pct']:.0f}% ｜ 线程 {s['threads']}"
            f" ｜ 近 10s 高危操作 {s.get('ops_10s', 0)} 次")


def log_event(action: str, s: Optional[dict] = None, note: str = "") -> None:
    s = s or sample()
    try:
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (f"[{ts}] {action} | load={s['load']:.3f} rss={s['rss_mb']:.0f}MB "
                f"cpu={s['cpu_pct']:.0f}% threads={s['threads']} {note}\n")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _overlay_html(text: str, theme: str = "light", detail: str = "") -> str:
    """全屏变暗遮罩（与 app.py 中的内联版本一致的观感）。

    性能要点：不用 backdrop-filter 全屏模糊；动画只用 opacity/transform（GPU 合成），
    遮罩出现后浏览器独立循环，即使服务器在算东西"小蓝点"也不会停。
    """
    dark = theme == "dark"
    bg = "rgba(8,12,18,0.80)" if dark else "rgba(10,18,30,0.66)"
    fg = "#e8eef7" if dark else "#ffffff"
    sub = "#a9b8cc" if dark else "#dbe6f5"
    det = f'<div class="s">{detail}</div>' if detail else ""
    return f"""
    <style>
      #__rg_ovl {{ position: fixed; inset: 0; z-index: 999999; background: {bg};
                  display: flex; flex-direction: column; align-items: center;
                  justify-content: center; gap: 12px; contain: strict; }}
      #__rg_ovl .t {{ color: {fg}; font-size: 16px; font-weight: 700; letter-spacing: 1px;
                      display: flex; align-items: center; gap: 8px; }}
      #__rg_ovl .ring {{ width: 16px; height: 16px; border-radius: 50%;
                         border: 2px solid rgba(111,183,255,.28); border-top-color: #6fb7ff;
                         animation: __rgspin .9s linear infinite; will-change: transform; }}
      #__rg_ovl .d {{ display: inline-block; width: 26px; }}
      #__rg_ovl .d span {{ animation: __rgb 1.2s infinite; opacity: .3; }}
      #__rg_ovl .d span:nth-child(2) {{ animation-delay: .15s; }}
      #__rg_ovl .d span:nth-child(3) {{ animation-delay: .3s; }}
      #__rg_ovl .s {{ color: {sub}; font-size: 12.5px; line-height: 1.75;
                      text-align: center; white-space: pre-line; max-width: min(600px, 86vw); }}
      #__rg_ovl .bar {{ position: relative; width: min(360px, 60vw); height: 5px;
                        border-radius: 4px; background: rgba(255,255,255,.16); overflow: hidden; }}
      #__rg_ovl .bar i {{ position: absolute; inset: 0; width: 38%; border-radius: 4px;
                          background: linear-gradient(90deg, rgba(111,183,255,0), #6fb7ff,
                                                      rgba(111,183,255,0));
                          will-change: transform; animation: __rgslide 1.25s ease-in-out infinite; }}
      @keyframes __rgspin {{ to {{ transform: rotate(360deg); }} }}
      @keyframes __rgslide {{ 0% {{ transform: translateX(-120%); }}
                              100% {{ transform: translateX(280%); }} }}
      @keyframes __rgb {{ 0%,20% {{opacity:1;}} 50% {{opacity:.15;}} 80%,100% {{opacity:1;}} }}
    </style>
    <div id="__rg_ovl">
      <div class="t"><span class="ring"></span>{text}
        <span class="d"><span>·</span><span>·</span><span>·</span></span></div>
      {det}
      <div class="bar"><i></i></div>
    </div>"""


def show(slot, text: str, theme: str = "light", detail: str = "") -> None:
    """把遮罩写进 st.empty() 占位符（单次写入，详情合并进同一段 HTML）。"""
    try:
        slot.markdown(_overlay_html(text, theme, detail), unsafe_allow_html=True)
    except Exception:
        pass


# 揭遮罩写进去的不可见占位块。
# ★ 2026-11-05 事故修复：不要对 st.empty() 调 slot.empty()。
#   同一个占位符在一次 script run 内「先写内容、再清空」＝第二次子节点比第一次少，
#   Streamlit AppRoot 在复用旧子树时会漏清多出来的节点（上游 issue #12069），
#   React fiber 与真实 DOM 因此不一致，下一次提交在 removeChild 上抛
#   NotFoundError: The node to be removed is not a child of this node → 整页卡死。
#   改成「用 1 个不可见节点换掉 1 个遮罩节点」，子节点数量恒定，走正常替换路径。
_OFF_HTML = ('<div id="dsh-dim-off" style="display:none;height:0;width:0;'
             'margin:0;padding:0;overflow:hidden"></div>')


def clear(slot) -> None:
    try:
        slot.markdown(_OFF_HTML, unsafe_allow_html=True)
    except Exception:
        pass


def release_hold(slot, action: str, theme: str = "light",
                 min_hold: float = 0.5, target: float = RELEASE_RATIO,
                 timeout: float = 3.5, tick: float = 0.28,
                 keep: bool = False) -> dict:
    """高危操作结束后调用：继续显示遮罩，直到负载降到红线 × 20% 或超时。

    ★ 重要（2026-11-04 深夜，用户点名"决定了以后会不会崩"）：
      这里**只写一次** markdown，之后纯 sleep 轮询负载，**不再每 tick 重写整段遮罩 HTML**。
      原因：每调用一次 slot.markdown(html) 都会替换主区一棵 DOM 子树并触发重排；
      叠加 iframe 画布时，0.28 s 一次的整屏替换会把主线程拖垮 —— 这正是之前
      "点预测后浏览器像卡死"的根因之一。遮罩自身的动效在浏览器合成线程里跑，
      不靠 Python 刷新也一直在动；负载数字只写进 终端资源.log，不写进 DOM。

    keep=True（2026-11-04 晚新增）：降完负载**不揭遮罩**，把它原样留给下一个阶段。
    用于「分析完成 → 交互画布预加载」的交接：整条链路上始终是同一层 #dsh-dim，
    由 iframe 里的预加载自检在「图片准备率 ≥90%」之后自己从父页面 DOM 删掉。
    """
    t0 = time.time()
    # 旧版文案 bug：调用方传 "分析完成，正在释放资源"，模板又追加 " · 正在释放资源"，
    # 于是界面上出现"分析完成，正在释放资源 · 正在释放资源"。这里去重。
    label = action if "释放" in action else f"{action} · 正在释放资源"
    if not keep:
        # keep=True 时连这一下都不写：上层马上会写"正在准备交互画布"的交接遮罩，
        # 让"分析遮罩 → 释放遮罩 → 画布遮罩"三次整屏替换变成"分析遮罩 → 画布遮罩"两次。
        show(slot, label, theme, "正在等待系统降温后自动继续…")
    last = sample()
    while True:
        el = time.time() - t0
        last = s = sample(cpu_interval=0.02)
        ok = (s["load"] <= target and el >= min_hold)
        if ok or el >= timeout:
            last["release_ok"] = ok
            last["waited_s"] = el
            log_event(action, s, note=f"release={'ok' if ok else 'timeout'} wait={el:.1f}s")
            if not keep:
                clear(slot)      # keep=True：不揭遮罩，留给下一阶段（画布预加载）继续盖着
            return last
        time.sleep(tick)


def cooldown_ok(key: str, seconds: float = 2.0) -> bool:
    """简易防连点：同一高危动作在 `seconds` 秒内重复触发则拒绝。"""
    now = time.time()
    try:
        import streamlit as st  # 延迟导入，避免非 Streamlit 环境报错
        last = st.session_state.get(f"__rg_last_{key}", 0.0)
        if now - last < seconds:
            return False
        st.session_state[f"__rg_last_{key}"] = now
    except Exception:
        pass
    return True
