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
    return STORE

def warm(loader) -> None:
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
        except Exception as ex:
            STORE["error"] = f"{type(ex).__name__}: {ex}"
        finally:
            STORE["t1"] = time.time()
            STORE["done"] = True

    threading.Thread(target=_bg, name="pemfc-model-warm", daemon=True).start()

def wait(timeout: float = 60.0) -> bool:
    t0 = time.time()
    while not STORE["done"] and time.time() - t0 < timeout:
        time.sleep(0.1)
    return STORE["model"] is not None

def reset() -> None:
    with _LOCK:
        STORE.update({"started": False, "done": False, "model": None,
                      "info": None, "error": None, "t0": None, "t1": None})

def elapsed() -> float | None:
    if STORE.get("t0") is None or STORE.get("t1") is None:
        return None
    return float(STORE["t1"]) - float(STORE["t0"])
