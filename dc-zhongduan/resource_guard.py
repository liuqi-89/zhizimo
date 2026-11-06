from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
except Exception:
    psutil = None

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "终端资源.log"

RED = {
    "rss_mb": 3000.0,
    "cpu_pct": 85.0,
    "threads": 128,
    "ops_10s": 2.0,
}
RELEASE_RATIO = 0.20
OPS_WINDOW_S = 10.0
IDLE = {"rss_mb": None, "cpu_pct": 0.0, "threads": 1}

def note_op(kind: str = "高危操作") -> None:
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
    p = _proc()
    if p is None:
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
    try:
        slot.markdown(_overlay_html(text, theme, detail), unsafe_allow_html=True)
    except Exception:
        pass

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
    t0 = time.time()

    label = action if "释放" in action else f"{action} · 正在释放资源"
    if not keep:

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
                clear(slot)
            return last
        time.sleep(tick)

def cooldown_ok(key: str, seconds: float = 2.0) -> bool:
    now = time.time()
    try:
        import streamlit as st
        last = st.session_state.get(f"__rg_last_{key}", 0.0)
        if now - last < seconds:
            return False
        st.session_state[f"__rg_last_{key}"] = now
    except Exception:
        pass
    return True
