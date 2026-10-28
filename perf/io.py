"""CSV templates and loading (returns file mandatory, sector file optional for attribution and risk contributions)."""
from __future__ import annotations

import io

import pandas as pd

from . import data


def to_csv(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode()


def load(returns_csv, sectors_csv=None):
    read = lambda f: pd.read_csv(io.BytesIO(f) if isinstance(f, (bytes, bytearray)) else f)  # noqa: E731
    r = read(returns_csv)
    s = read(sectors_csv) if sectors_csv is not None else None
    return data.validate(r, s)
