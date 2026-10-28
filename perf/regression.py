"""
Time-series regressions for the CAPM and multifactor models, with Newey-West (HAC) standard errors.

CAPM / Jensen:          r − r_f = α + β (r_m − r_f) + ε
Treynor-Mazuy timing:   r − r_f = α + β x + γ x² + ε            (γ > 0: market timing)
Henriksson-Merton:      r − r_f = α + β x + γ max(x, 0) + ε     (γ > 0: market timing)
Fama-French 3, Carhart 4, Fama-French 5, FF5 + momentum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MODELS = {
    "CAPM": ["MKT_RF"],
    "Fama-French 3": ["MKT_RF", "SMB", "HML"],
    "Carhart 4": ["MKT_RF", "SMB", "HML", "MOM"],
    "Fama-French 5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
    "Fama-French 5 + momentum": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


@dataclass
class OLS:
    names: list
    coef: np.ndarray
    se: np.ndarray            # Newey-West
    se_ols: np.ndarray
    resid: np.ndarray
    fitted: np.ndarray
    r2: float
    r2_adj: float
    n: int
    lags: int

    @property
    def t(self):
        return self.coef / self.se

    @property
    def p(self):
        return 2 * stats.t.sf(np.abs(self.t), self.n - len(self.coef))

    def table(self, ppy: int = 12) -> pd.DataFrame:
        df = pd.DataFrame({"coefficient": self.coef, "NW std error": self.se, "t (NW)": self.t, "p-value": self.p,
                           "t (OLS)": self.coef / self.se_ols}, index=self.names)
        df.loc["alpha", "annualised"] = self.coef[0] * ppy
        return df


def nw_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def ols(y, X, names, lags: int | None = None) -> OLS:
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (n - k)
    se_ols = np.sqrt(np.diag(XtX_inv) * s2)
    L = nw_lags(n) if lags is None else lags
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * n / (n - k)
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e @ e) / tss
    return OLS(["alpha"] + list(names), b, np.sqrt(np.diag(V)), se_ols, e, X @ b, r2, 1 - (1 - r2) * (n - 1) / (n - k), n, L)


def capm(returns: pd.DataFrame, fund: str, market: str = "benchmark") -> OLS:
    y = returns[fund] - returns["rf"]
    x = returns[market] - returns["rf"] if market == "benchmark" else returns[market]
    return ols(y, x.to_frame(), ["beta"])


def timing(returns: pd.DataFrame, fund: str, kind: str = "Treynor-Mazuy") -> OLS:
    y = returns[fund] - returns["rf"]
    x = (returns["benchmark"] - returns["rf"]).to_numpy()
    z = x ** 2 if kind == "Treynor-Mazuy" else np.maximum(x, 0)
    return ols(y, np.column_stack([x, z]), ["beta", "timing γ"])


def factor_model(returns: pd.DataFrame, fund: str, model: str, active: bool = False) -> OLS:
    cols = MODELS[model]
    y = returns[fund] - (returns["benchmark"] if active else returns["rf"])
    return ols(y, returns[cols], cols)


def factor_attribution(returns: pd.DataFrame, fund: str, model: str, ppy: int = 12, active: bool = False) -> pd.Series:
    """Average (annualised) excess return = α + Σ β_k · mean(f_k) (+ zero-mean residual)."""
    res = factor_model(returns, fund, model, active)
    cols = MODELS[model]
    out = {"alpha": res.coef[0] * ppy}
    for i, c in enumerate(cols):
        out[c] = res.coef[i + 1] * returns[c].mean() * ppy
    s = pd.Series(out)
    s.attrs["total"] = ((returns[fund] - (returns["benchmark"] if active else returns["rf"])).mean()) * ppy
    return s


def rolling_betas(returns: pd.DataFrame, fund: str, model: str, window: int = 36) -> pd.DataFrame:
    cols = MODELS[model]
    y = (returns[fund] - returns["rf"]).to_numpy()
    X = returns[cols].to_numpy()
    rows = []
    for t in range(window, len(y) + 1):
        r = ols(y[t - window:t], X[t - window:t], cols, lags=0)
        rows.append(np.r_[r.coef[0] * 12, r.coef[1:]])
    return pd.DataFrame(rows, index=returns.index[window - 1:], columns=["alpha (ann.)"] + cols)


def sml(returns: pd.DataFrame, assets: pd.DataFrame, ppy: int = 12) -> pd.DataFrame:
    """Security market line: realised mean excess return against beta to the benchmark for each asset column."""
    exb = returns["benchmark"] - returns["rf"]
    rows = []
    for c in assets.columns:
        ex = assets[c] - returns["rf"]
        r = ols(ex, exb.to_frame(), ["beta"])
        rows.append({"asset": c, "beta": r.coef[1], "mean excess (ann.)": ex.mean() * ppy, "alpha (ann.)": r.coef[0] * ppy, "t alpha": r.t[0]})
    df = pd.DataFrame(rows).set_index("asset")
    df.attrs["market premium"] = exb.mean() * ppy
    return df
