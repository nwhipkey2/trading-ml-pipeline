from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from typing import Optional
import pandas as pd

from src.utils.http import get_bytes, HttpConfig

ECB_BASE = "https://data-api.ecb.europa.eu/service/data"

@dataclass
class EcbQuery:
    flow: str               # e.g. "EXR" or "YC"
    key: str                # e.g. "D.USD.EUR.SP00.A" or "B.U2....SR_10Y" (NO "YC." prefix)
    start: Optional[str] = None  # YYYY-MM-DD
    end: Optional[str] = None    # YYYY-MM-DD
    format: str = "csvdata"

def _sanitize_key(flow: str, key: str) -> str:
    # Some portal pages show "Series key: YC.B...." but the API expects "B...." when flow=YC.
    # Same idea for EXR: API expects "D.USD.EUR.SP00.A" not "EXR.D...."
    if key.startswith(flow + "."):
        return key[len(flow) + 1 :]
    return key

def fetch_ecb_csv(q: EcbQuery, http: HttpConfig | None = None) -> pd.DataFrame:
    key = _sanitize_key(q.flow, q.key)

    url = f"{ECB_BASE}/{q.flow}/{key}?format={q.format}"
    if q.start:
        url += f"&startPeriod={q.start}"
    if q.end:
        url += f"&endPeriod={q.end}"

    raw = get_bytes(url, http)
    df = pd.read_csv(BytesIO(raw))
    return df
