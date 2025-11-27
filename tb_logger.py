import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf
import yaml

_writer = None
_log_dir: Optional[str] = None
_enabled = False


class Histogram:
    """Container type to mark values that should be logged as histograms."""

    def __init__(self, data: Any):
        self.data = np.asarray(data)


def init(log_dir: str, enabled: bool = True):
    """Initialize a TensorBoard SummaryWriter."""
    global _writer, _log_dir, _enabled
    _log_dir = log_dir
    _enabled = enabled
    if enabled:
        os.makedirs(log_dir, exist_ok=True)
        _writer = tf.summary.create_file_writer(log_dir)
    else:
        _writer = None
    return _writer


def is_enabled() -> bool:
    return _enabled and _writer is not None


def get_log_dir() -> Optional[str]:
    return _log_dir


def save_config(config: Mapping[str, Any], tags: Optional[Sequence[str]] = None):
    """Persist the resolved config to disk and log it for reference."""
    if _log_dir is None:
        return
    plain_config = _to_plain_dict(config)
    config_path = os.path.join(_log_dir, "config.resolved.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(plain_config, f)

    if not is_enabled():
        return
    with _writer.as_default():
        tf.summary.text("config", tf.constant(yaml.safe_dump(plain_config)), step=0)
        if tags:
            tf.summary.text("tags", tf.constant(", ".join(tags)), step=0)
    _writer.flush()


def log_scalar(name: str, value: Any, step: int):
    if not is_enabled():
        return
    scalar = _to_scalar(value)
    with _writer.as_default():
        tf.summary.scalar(name, scalar, step=step)
    _writer.flush()


def log_text(name: str, text: str, step: int):
    if not is_enabled():
        return
    with _writer.as_default():
        tf.summary.text(name, tf.constant(text), step=step)
    _writer.flush()


def log_dict(values: Mapping[str, Any], step: int):
    if not is_enabled():
        return
    with _writer.as_default():
        for name, value in values.items():
            _write_value(name, value, step)
    _writer.flush()


def _write_value(name: str, value: Any, step: int):
    if isinstance(value, Histogram):
        tf.summary.histogram(name, value.data, step=step)
        return
    if isinstance(value, str):
        tf.summary.text(name, tf.constant(value), step=step)
        return
    tf.summary.scalar(name, _to_scalar(value), step=step)


def _to_scalar(value: Any) -> float:
    array = np.asarray(value)
    if array.shape == ():
        return float(array)
    return float(array.mean())


def _to_plain_dict(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return {k: _to_plain_dict(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [_to_plain_dict(v) for v in config]
    return config
