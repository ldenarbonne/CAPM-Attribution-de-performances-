"""
Brinson-Fachler sector attribution and multi-period linking.

Single period, sector i, total benchmark return R_b = Σ w_b,i r_b,i, portfolio return R_p = Σ w_p,i r_p,i:
  allocation   A_i = (w_p,i − w_b,i)(r_b,i − R_b)
  selection    S_i = w_b,i (r_p,i − r_b,i)
  interaction  I_i = (w_p,i − w_b,i)(r_p,i − r_b,i)
  Σ (A + S + I) = R_p − R_b                      (exact, arithmetic active return)
Option "selection incl. interaction": S_i = w_p,i (r_p,i − r_b,i).

Linking over T periods so that the effects add up to the compounded active return Π(1+R_p) − Π(1+R_b):
* Carino (1999): k_t = [ln(1+R_p,t) − ln(1+R_b,t)] / (R_p,t − R_b,t), K = same on cumulative returns;
  linked effect = Σ_t effect_t · k_t / K.
* Frongello (2002): G_t = A_t · Π_{j<t}(1+R_p,j) + R_b,t · Σ_{j<t} G_j  (applied per effect, recursive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EFFECTS = ["allocation", "selection", "interaction"]


def single_period(g: pd.DataFrame, merge_interaction: bool = False) -> pd.DataFrame:
    wp, wb, rp, rb = (g[c].to_numpy() for c in ["w_port", "w_bench", "r_port", "r_bench"])
    Rb = (wb * rb).sum()
    out = pd.DataFrame({"w_port": wp, "w_bench": wb, "active weight": wp - wb, "r_port": rp, "r_bench": rb}, index=g["sector"].to_numpy())
    out["allocation"] = (wp - wb) * (rb - Rb)
    if merge_interaction:
        out["selection"] = wp * (rp - rb)
        out["interaction"] = 0.0
    else:
        out["selection"] = wb * (rp - rb)
        out["interaction"] = (wp - wb) * (rp - rb)
    out["total"] = out[EFFECTS].sum(axis=1)
    out.attrs.update(Rp=float((wp * rp).sum()), Rb=float(Rb))
    return out


def by_period(sectors: pd.DataFrame, fund: str, merge_interaction: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (effects long: date × sector × effect, totals: date → Rp, Rb)."""
    s = sectors[sectors["fund"] == fund]
    frames, tot = [], []
    for d, g in s.groupby("date"):
        sp = single_period(g, merge_interaction)
        frames.append(sp[EFFECTS].assign(date=d).rename_axis("sector").reset_index())
        tot.append({"date": d, "Rp": sp.attrs["Rp"], "Rb": sp.attrs["Rb"]})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(tot).set_index("date")


def _k(rp, rb):
    rp, rb = np.asarray(rp, float), np.asarray(rb, float)
    d = rp - rb
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(np.abs(d) > 1e-12, (np.log1p(rp) - np.log1p(rb)) / d, 1 / (1 + rp))
    return k


def link(effects: pd.DataFrame, totals: pd.DataFrame, method: str = "Carino") -> pd.DataFrame:
    """Linked effects by sector (rows) × effect (columns). Sum of the table = compounded active return."""
    dates = totals.index
    Rp, Rb = totals["Rp"].to_numpy(), totals["Rb"].to_numpy()
    P = effects.pivot_table(index="date", columns=["sector"], values=EFFECTS, aggfunc="sum").reindex(dates).fillna(0.0)
    if method == "Arithmetic (unlinked)":
        L = P.sum()
    elif method == "Carino":
        Rpc, Rbc = np.prod(1 + Rp) - 1, np.prod(1 + Rb) - 1
        K = _k(Rpc, Rbc)
        k = _k(Rp, Rb)
        L = (P.mul(k, axis=0)).sum() / K
    elif method == "Frongello":
        cum_p = np.concatenate([[1.0], np.cumprod(1 + Rp)[:-1]])
        G = np.zeros(P.shape)
        A = P.to_numpy()
        for t in range(len(dates)):
            G[t] = A[t] * cum_p[t] + Rb[t] * G[:t].sum(0)
        L = pd.Series(G.sum(0), index=P.columns)
    else:
        raise ValueError(method)
    out = L.unstack(0).reindex(columns=EFFECTS)
    out["total"] = out.sum(axis=1)
    out.attrs["active (compounded)"] = float(np.prod(1 + Rp) - np.prod(1 + Rb))
    out.attrs["Rp"] = float(np.prod(1 + Rp) - 1)
    out.attrs["Rb"] = float(np.prod(1 + Rb) - 1)
    return out


def cumulative_effects(effects: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Frongello-linked effects cumulated through time (for a running chart), by effect."""
    dates = totals.index
    Rp, Rb = totals["Rp"].to_numpy(), totals["Rb"].to_numpy()
    P = effects.groupby("date")[EFFECTS].sum().reindex(dates).to_numpy()
    cum_p = np.concatenate([[1.0], np.cumprod(1 + Rp)[:-1]])
    G = np.zeros_like(P)
    for t in range(len(dates)):
        G[t] = P[t] * cum_p[t] + Rb[t] * G[:t].sum(0)
    out = pd.DataFrame(G.cumsum(0), index=dates, columns=EFFECTS)
    out["active (compounded)"] = np.cumprod(1 + Rp) - np.cumprod(1 + Rb)
    return out
