from __future__ import annotations
import time
from dataclasses import dataclass
import requests

@dataclass
class HttpConfig:
    timeout_s: int = 60
    retries: int = 3
    backoff_s: float = 1.0
    user_agent: str = "trading-ml-pipeline/0.1"

def get_bytes(url: str, cfg: HttpConfig | None = None) -> bytes:
    cfg = cfg or HttpConfig()
    headers = {"User-Agent": cfg.user_agent}
    last_err: Exception | None = None

    for i in range(cfg.retries):
        try:
            r = requests.get(url, headers=headers, timeout=cfg.timeout_s)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            time.sleep(cfg.backoff_s * (2 ** i))

    raise RuntimeError(f"GET failed after {cfg.retries} tries: {url}") from last_err
