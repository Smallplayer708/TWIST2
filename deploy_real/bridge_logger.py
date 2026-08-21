import os
import time


class BridgeLogger:
    _instances = {}

    def __init__(self, component, log_dir=None):
        self.component = component
        if log_dir is None:
            log_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(log_dir, "twist2_orca_bridge.log")

    @classmethod
    def get(cls, component, log_dir=None):
        key = (component, log_dir or "")
        if key not in cls._instances:
            cls._instances[key] = cls(component, log_dir)
        return cls._instances[key]

    def _write(self, level, msg, **kwargs):
        ts = time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
        payload = " ".join(f"{k}={v}" for k, v in kwargs.items())
        line = f"[{ts}] [{self.component}] [{level}] {msg}"
        if payload:
            line += " " + payload
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()

    def info(self, msg, **kwargs):
        self._write("INFO", msg, **kwargs)

    def warn(self, msg, **kwargs):
        self._write("WARN", msg, **kwargs)

    def err(self, msg, **kwargs):
        self._write("ERR", msg, **kwargs)


get_bridge_logger = BridgeLogger.get
