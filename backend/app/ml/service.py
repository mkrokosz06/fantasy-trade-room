"""Background training and cached predictions for the web app."""
from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone

import pandas as pd

from ..data.store import DataStore, get_store
from . import model
from .dataset import label_feature


class NotTrained(Exception):
    pass


_state = {"status": "idle", "log": [], "started_at": None, "finished_at": None, "error": None}
_lock = threading.Lock()
_pred_cache: dict[tuple, pd.DataFrame] = {}


def status() -> dict:
    with _lock:
        s = dict(_state)
        s["log"] = list(_state["log"][-60:])
    meta = model.load_meta()
    if meta:
        # Labels may have improved since training; recompute from feature names.
        for feats in meta.get("importance", {}).values():
            for f in feats:
                f["label"] = label_feature(f["feature"])
    s["trained"] = meta is not None
    s["meta"] = meta
    if s["status"] == "idle" and meta:
        s["status"] = "ready"
    return s


def _log(msg: str):
    with _lock:
        _state["log"].append(msg)


def start_training() -> bool:
    with _lock:
        if _state["status"] == "training":
            return False
        _state.update(status="training", log=[], started_at=datetime.now(timezone.utc).isoformat(),
                      finished_at=None, error=None)

    def run():
        try:
            model.train_and_save(get_store(), log=_log)
            _pred_cache.clear()
            with _lock:
                _state["status"] = "ready"
        except Exception as e:  # noqa: BLE001
            _log("Training failed: " + "".join(traceback.format_exception_only(e)).strip())
            with _lock:
                _state["status"] = "error"
                _state["error"] = str(e)
        finally:
            with _lock:
                _state["finished_at"] = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=run, daemon=True, name="ml-train").start()
    return True


def predictions(store: DataStore, end_week: int) -> tuple[pd.DataFrame, dict]:
    meta = model.load_meta()
    if meta is None:
        raise NotTrained("The model hasn't been trained yet. Open Model lab and click Train model.")
    key = (store.loaded_at, meta["trained_at"], end_week)
    if key not in _pred_cache:
        _pred_cache.clear()
        scale = float(meta["backtest"].get("range_scale", 1.0))
        _pred_cache[key] = model.predict_future(store, end_week, scale)
    return _pred_cache[key], meta
