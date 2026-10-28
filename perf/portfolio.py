"""
Mean-variance frontier and risk contributions.

Frontier (Markowitz) on the benchmark sector indices:
* closed form without constraints (two-fund separation): global minimum variance, tangency portfolio, CML;
* long-only frontier by quadratic programming (SLSQP) on a grid of target returns;
* covariance: sample, optionally shrunk towards its diagonal (Ledoit-Wolf-style target, user intensity).

Risk contributions (ex ante, from the last holdings):
* active risk = √(a'Σa + Σ w_p,i² σ²_sel,i), a = w_p − w_b (allocation risk + stock/selection risk within sectors);
  contribution of sector i to allocation risk: a_i (Σa)_i / TE (Euler, sums to the allocation variance / TE);
* active risk from a factor model of the active return: β'Σ_f β (systematic) + residual variance (specific);
* total risk and VaR: parametric (normal) VaR with Euler component VaR by sector, historical VaR on the
  simulated portfolio history at current weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def cov_matrix(R: pd.DataFrame, shrink: float = 0.0, ppy: int = 12, ewma: float | None = None) -> pd.DataFrame:
    X = R - R.mean()
    if ewma:
        w = ewma ** np.arange(len(X))[::-1]
        w /= w.sum()
        S = (X.to_numpy() * w[:, None]).T @ X.to_numpy()
    else:
        S = np.cov(X.to_numpy(), rowvar=False, ddof=1)
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    return pd.DataFrame(S * ppy, index=R.columns, columns=R.columns)


def analytic(mu: np.ndarray, S: np.ndarray, rf: float) -> dict:
    Si = np.linalg.inv(S)
    one = np.ones(len(mu))
    a, b, c = one @ Si @ one, one @ Si @ mu, mu @ Si @ mu
    w_gmv = Si @ one / a
    w_tan = Si @ (mu - rf) / (one @ Si @ (mu - rf))
    targets = np.linspace(min(mu.min(), b / a) - 0.02, mu.max() + 0.04, 80)
    var = (a * targets ** 2 - 2 * b * targets + c) / (a * c - b ** 2)
    return {"gmv": w_gmv, "tangency": w_tan, "frontier": pd.DataFrame({"return": targets, "vol": np.sqrt(var)})}


def long_only(mu: np.ndarray, S: np.ndarray, rf: float, n: int = 30, w_max: float = 1.0) -> dict:
    k = len(mu)
    cons_sum = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0, w_max)] * k
    w0 = np.ones(k) / k
    gmv = minimize(lambda w: w @ S @ w, w0, bounds=bnds, constraints=[cons_sum], method="SLSQP").x
    lo = gmv @ mu
    hi = max(mu.max() if w_max >= 1 else np.sort(mu)[::-1][: int(np.ceil(1 / w_max))].mean(), lo)
    rows, ws = [], []
    for tgt in np.linspace(lo, hi * 0.999 + lo * 0.001, n):
        res = minimize(lambda w: w @ S @ w, w0, bounds=bnds, method="SLSQP",
                       constraints=[cons_sum, {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t}])
        if res.success:
            rows.append({"return": res.x @ mu, "vol": np.sqrt(res.x @ S @ res.x)})
            ws.append(res.x)
    tan = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ S @ w), w0, bounds=bnds, constraints=[cons_sum], method="SLSQP").x
    return {"gmv": gmv, "tangency": tan, "frontier": pd.DataFrame(rows), "weights": np.array(ws)}


def point(w, mu, S):
    return float(w @ mu), float(np.sqrt(w @ S @ w))


# ---------------------------------------------------------------- risk contributions
def active_risk(sectors: pd.DataFrame, fund: str, sector_returns: pd.DataFrame, rf: pd.Series, ppy: int = 12,
                shrink: float = 0.0, ewma: float | None = None) -> pd.DataFrame:
    s = sectors[sectors["fund"] == fund]
    last = s[s["date"] == s["date"].max()].set_index("sector")
    R = sector_returns.copy()
    R["Cash"] = rf
    S = cov_matrix(R, shrink, ppy, ewma).reindex(index=last.index, columns=last.index).fillna(0.0)
    a = (last["w_port"] - last["w_bench"]).to_numpy()
    wp = last["w_port"].to_numpy()
    Sa = S.to_numpy() @ a
    alloc_var = float(a @ Sa)
    sel = s.assign(d=s["r_port"] - s["r_bench"]).pivot_table(index="date", columns="sector", values="d").reindex(columns=last.index).fillna(0.0)
    sel_var = sel.var(ddof=1).to_numpy() * ppy * wp ** 2
    te = np.sqrt(alloc_var + sel_var.sum())
    out = pd.DataFrame({"w_port": wp, "w_bench": last["w_bench"].to_numpy(), "active weight": a,
                        "allocation contribution": a * Sa / te, "selection contribution": sel_var / te}, index=last.index)
    out["total contribution"] = out["allocation contribution"] + out["selection contribution"]
    out["% of TE²"] = out["total contribution"] / te
    out.attrs.update(TE=te, allocation=np.sqrt(alloc_var), selection=np.sqrt(sel_var.sum()), date=s["date"].max().strftime("%b %Y"))
    return out


def factor_risk(returns: pd.DataFrame, fund: str, factors: list[str], active: bool = True, ppy: int = 12) -> pd.Series:
    from .regression import ols
    y = returns[fund] - (returns["benchmark"] if active else returns["rf"])
    r = ols(y, returns[factors], factors, lags=0)
    b = r.coef[1:]
    Sf = np.cov(returns[factors].to_numpy(), rowvar=False, ddof=1) * ppy
    Sf = np.atleast_2d(Sf)
    contrib = b * (Sf @ b)
    resid = r.resid.var(ddof=1) * ppy
    tot = contrib.sum() + resid
    s = pd.Series(np.r_[contrib, resid] / np.sqrt(tot), index=factors + ["specific"])
    s.attrs["vol"] = float(np.sqrt(tot))
    s.attrs["realised"] = float(y.std(ddof=1) * np.sqrt(ppy))
    return s


def var_contributions(sectors: pd.DataFrame, fund: str, sector_returns: pd.DataFrame, rf: pd.Series, q: float = 0.99,
                      horizon_months: int = 1) -> tuple[pd.DataFrame, dict]:
    s = sectors[sectors["fund"] == fund]
    last = s[s["date"] == s["date"].max()].set_index("sector")
    R = sector_returns.copy()
    R["Cash"] = rf
    # fund-specific sector returns (benchmark sector + selection) for the historical simulation
    Rp = s.pivot_table(index="date", columns="sector", values="r_port").reindex(columns=last.index)
    w = last["w_port"].to_numpy()
    mu = Rp.mean().to_numpy() * horizon_months
    S = np.cov(Rp.to_numpy(), rowvar=False, ddof=1) * horizon_months
    sig = np.sqrt(w @ S @ w)
    z = norm.ppf(q)
    var_p = -(w @ mu) + z * sig
    comp = w * (-mu + z * (S @ w) / sig)
    hist = Rp.to_numpy() @ w
    hvar = -np.quantile(hist, 1 - q) if horizon_months == 1 else -np.quantile(hist, 1 - q) * np.sqrt(horizon_months)
    hes = -hist[hist <= np.quantile(hist, 1 - q)].mean()
    out = pd.DataFrame({"weight": w, "component VaR": comp, "% of VaR": comp / var_p}, index=last.index)
    return out, {"parametric VaR": float(var_p), "historical VaR": float(hvar), "historical ES": float(hes), "vol": float(sig)}
