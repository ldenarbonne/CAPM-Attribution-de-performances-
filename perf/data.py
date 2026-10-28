"""
Synthetic monthly data set: risk-free rate, equity factors, sector indices of a benchmark, and several active
equity funds with known (planted) skills, so that every tool of the app has something to find.

Tables
* returns  : date × [rf, MKT_RF, SMB, HML, RMW, CMA, MOM, benchmark, <fund 1>, <fund 2>, ...]   (net fund returns)
* sectors  : long table date, fund, sector, w_port, w_bench, r_port, r_bench   (gross, for attribution)
             sector "Cash" has w_bench = 0 and r = rf.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
SECTORS = ["Energy", "Materials", "Industrials", "Consumer discretionary", "Consumer staples", "Health care",
           "Financials", "Information technology", "Communication services", "Utilities", "Real estate"]

F_MEAN = np.array([0.55, 0.12, 0.18, 0.30, 0.18, 0.50]) / 100
F_VOL = np.array([4.4, 2.8, 2.9, 2.0, 1.9, 4.2]) / 100
F_CORR = np.array([
    [1.00, 0.25, -0.10, -0.25, -0.30, -0.20],
    [0.25, 1.00, 0.05, -0.35, 0.00, -0.05],
    [-0.10, 0.05, 1.00, 0.15, 0.60, -0.35],
    [-0.25, -0.35, 0.15, 1.00, 0.10, 0.10],
    [-0.30, 0.00, 0.60, 0.10, 1.00, 0.05],
    [-0.20, -0.05, -0.35, 0.10, 0.05, 1.00]])

# sector loadings on [MKT, SMB, HML, RMW, CMA, MOM], idiosyncratic vol (monthly), initial benchmark weight
SECTOR_LOAD = pd.DataFrame([
    [1.05, 0.10, 0.55, -0.10, 0.20, -0.10, 0.045, 0.05],
    [1.15, 0.15, 0.30, -0.05, 0.05, 0.00, 0.035, 0.07],
    [1.10, 0.10, 0.05, 0.10, 0.00, 0.05, 0.020, 0.15],
    [1.20, 0.20, -0.10, 0.05, -0.20, 0.05, 0.030, 0.11],
    [0.60, -0.15, 0.05, 0.35, 0.20, 0.00, 0.022, 0.10],
    [0.75, -0.10, -0.20, 0.30, 0.10, 0.05, 0.025, 0.14],
    [1.15, 0.00, 0.60, -0.15, 0.10, -0.05, 0.030, 0.17],
    [1.25, 0.05, -0.45, 0.20, -0.35, 0.15, 0.035, 0.08],
    [0.95, 0.00, -0.05, 0.00, -0.05, 0.00, 0.030, 0.04],
    [0.55, -0.10, 0.25, 0.10, 0.25, -0.05, 0.025, 0.05],
    [0.85, 0.20, 0.35, -0.10, 0.15, -0.05, 0.035, 0.04]],
    index=SECTORS, columns=FACTORS + ["idio vol", "w0"])

FUNDS = {
    "Quality Growth": dict(
        tilt={"Information technology": 0.05, "Health care": 0.04, "Consumer staples": 0.03, "Energy": -0.04, "Financials": -0.06,
              "Utilities": -0.03, "Real estate": -0.02},
        style=[0.0, -0.10, -0.20, 0.35, 0.0, 0.05], sel_alpha={"Information technology": 0.0020, "Health care": 0.0020}, sel_vol=0.012, sel_all=0.0,
        fee=0.012, cash=0.02, timing=0.0),
    "Value Small Caps": dict(
        tilt={"Financials": 0.06, "Energy": 0.03, "Materials": 0.02, "Information technology": -0.05, "Health care": -0.04, "Communication services": -0.02},
        style=[0.05, 0.35, 0.30, -0.05, 0.05, -0.05], sel_alpha={"Industrials": 0.0010}, sel_vol=0.015, sel_all=0.0020, fee=0.014, cash=0.03, timing=0.0),
    "Tactical Allocator": dict(
        tilt={"Consumer discretionary": 0.02, "Utilities": -0.02}, style=[0.0, 0.0, 0.0, 0.0, 0.0, 0.10], sel_alpha={}, sel_vol=0.008,
        sel_all=0.0, fee=0.015, cash=0.12, timing=1.0),
    "Closet Indexer": dict(
        tilt={"Industrials": 0.01, "Real estate": -0.01}, style=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sel_alpha={}, sel_vol=0.003, sel_all=0.0,
        fee=0.016, cash=0.02, timing=0.0),
}


def generate(months: int = 144, seed: int = 21, start: str = "2014-11-30") -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=months, freq="ME")
    # risk-free: low then rising (annual 0 % → ~3.5 %)
    rf = np.clip(np.concatenate([np.full(months // 2, 0.0002), np.linspace(0.0002, 0.0030, months - months // 2)])
                 + rng.normal(0, 0.00005, months), 0, None)
    # multivariate Student-t factors (fat tails) + a crisis cluster
    L = np.linalg.cholesky(F_CORR)
    df = 5
    z = rng.standard_normal((months, 6)) @ L.T * np.sqrt((df - 2) / rng.chisquare(df, (months, 1)))
    f = F_MEAN + z * F_VOL
    crisis = slice(int(months * 0.38), int(months * 0.38) + 4)
    f[crisis, 0] += np.array([-0.06, -0.09, -0.03, 0.05])[: f[crisis, 0].shape[0]]
    f[crisis, 5] += np.array([0.02, 0.03, -0.08, -0.06])[: f[crisis, 0].shape[0]]
    f = f - f.mean(0) + F_MEAN          # sample premia set to their design values (no sampling luck in the demo)
    F = pd.DataFrame(f, index=dates, columns=FACTORS)
    load = SECTOR_LOAD[FACTORS].to_numpy()
    rb = rf[:, None] + f @ load.T + rng.standard_normal((months, len(SECTORS))) * SECTOR_LOAD["idio vol"].to_numpy()
    rb = pd.DataFrame(rb, index=dates, columns=SECTORS)
    # benchmark weights drift with relative performance (buy and hold), rebalanced to the start of each month
    wb = np.zeros((months, len(SECTORS)))
    w = SECTOR_LOAD["w0"].to_numpy() / SECTOR_LOAD["w0"].sum()
    for t in range(months):
        wb[t] = w
        w = w * (1 + rb.iloc[t].to_numpy())
        w /= w.sum()
    bench = (wb * rb.to_numpy()).sum(1)
    out = pd.DataFrame({"rf": rf}, index=dates).join(F)
    out["benchmark"] = bench
    rows = []
    mkt_next = f[:, 0]
    for name, p in FUNDS.items():
        tilt = np.array([p["tilt"].get(s, 0.0) for s in SECTORS])
        style = np.array(p["style"])
        sa = np.array([p["sel_alpha"].get(s, 0.0) for s in SECTORS]) + p["sel_all"]
        rp = rb.to_numpy() + (f @ style)[:, None] + sa + rng.standard_normal((months, len(SECTORS))) * p["sel_vol"]
        ret = np.zeros(months)
        for t in range(months):
            cash = p["cash"]
            if p["timing"] > 0:   # imperfect foresight of the market sign → cash between 0 and 2 × target
                sig = np.sign(mkt_next[t]) if rng.random() < 0.5 + p["timing"] / 2.5 else -np.sign(mkt_next[t])
                cash = float(np.clip(p["cash"] * (1 - 0.9 * sig), 0, 0.25))
            noise = rng.normal(0, 0.003, len(SECTORS))
            we = np.clip(wb[t] + tilt + noise, 0, None)
            we = we / we.sum() * (1 - cash)
            ret[t] = (we * rp[t]).sum() + cash * rf[t]
            for i, s in enumerate(SECTORS):
                rows.append((dates[t], name, s, we[i], wb[t, i], rp[t, i], rb.iat[t, i]))
            rows.append((dates[t], name, "Cash", cash, 0.0, rf[t], rf[t]))
        out[name] = ret - p["fee"] / 12
    sectors = pd.DataFrame(rows, columns=["date", "fund", "sector", "w_port", "w_bench", "r_port", "r_bench"])
    out.index.name = "date"
    return out, sectors


def fund_names(returns: pd.DataFrame) -> list[str]:
    return [c for c in returns.columns if c not in ["rf", "benchmark"] + FACTORS]


def validate(returns: pd.DataFrame, sectors: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if "date" in returns.columns:
        returns = returns.set_index("date")
    returns.index = pd.to_datetime(returns.index)
    need = ["rf", "benchmark", "MKT_RF"]
    miss = [c for c in need if c not in returns.columns]
    if miss:
        raise ValueError(f"returns: missing columns {', '.join(miss)}")
    returns = returns.apply(pd.to_numeric, errors="coerce").sort_index()
    if not fund_names(returns):
        raise ValueError("returns: no fund column found")
    if returns[need].isna().any().any():
        raise ValueError("returns: missing values in rf / benchmark / MKT_RF")
    if sectors is not None:
        miss = [c for c in ["date", "fund", "sector", "w_port", "w_bench", "r_port", "r_bench"] if c not in sectors.columns]
        if miss:
            raise ValueError(f"sectors: missing columns {', '.join(miss)}")
        sectors = sectors.copy()
        sectors["date"] = pd.to_datetime(sectors["date"])
        g = sectors.groupby(["fund", "date"])[["w_port", "w_bench"]].sum()
        if (g.sub(1).abs() > 0.02).any().any():
            raise ValueError("sectors: portfolio and benchmark weights must sum to 1 for every fund and date")
    return returns, sectors
