# -*- coding: utf-8 -*-
"""后台预热模型（页面一打开就开始，不阻塞首屏）。

为什么要它（2026-11-04 晚实测）：
- 这台 Windows 机器上**首次 `import torch` 约 5–10 s**（Defender 实时扫描 + 冷启动），
  而真正的 `load_state_model` 只要 0.05 s。用户点「开始预测」时同步等这 5–10 s，
  浏览器就会显示"无响应"（即使我们的遮罩动画在转，用户也会以为卡死）。
- 解决办法：页面加载后立刻在后台线程里 import torch + 加载模型，把这段时间藏到
  "用户看页面 / 选文件 / 拖滑条"的空闲里；点预测时模型通常已就绪，链路从 ~17 s 降到 ~6 s。

实现要点：状态放在**模块级 dict**里（Python 模块只 import 一次，跨 Streamlit rerun 保留），
不能在 app.py 里用全局变量（脚本每次 rerun 都会重新执行，全局会重置）。
"""
from __future__ import annotations

import threading
import time

STORE = {
    "started": False,
    "done": False,
    "model": None,
    "info": None,
    "error": None,
    "t0": None,
    "t1": None,
}
_LOCK = threading.Lock()


def get() -> dict:
    """返回预热状态（dict 原地更新，可直接引用）。"""
    return STORE


def warm(loader) -> None:
    """启动后台预热线程（重复调用无副作用；loader 返回 (model, info)）。"""
    with _LOCK:
        if STORE["started"]:
            return
        STORE["started"] = True
    STORE["t0"] = time.time()

    def _bg() -> None:
        try:
            model, info = loader()
            STORE["model"] = model
            STORE["info"] = info
        except Exception as ex:  # noqa: BLE001 - 任何失败都不该影响主流程
            STORE["error"] = f"{type(ex).__name__}: {ex}"
        finally:
            STORE["t1"] = time.time()
            STORE["done"] = True

    threading.Thread(target=_bg, name="pemfc-model-warm", daemon=True).start()


def wait(timeout: float = 60.0) -> bool:
    """阻塞等待预热结束（返回是否已拿到模型）。仅在主线程、确实需要模型时调用。"""
    t0 = time.time()
    while not STORE["done"] and time.time() - t0 < timeout:
        time.sleep(0.1)
    return STORE["model"] is not None


def elapsed() -> float | None:
    """预热耗时（秒，未完成返回 None）。"""
    if STORE.get("t0") is None or STORE.get("t1") is None:
        return None
    return float(STORE["t1"]) - float(STORE["t0"])
