"""
Return-based performance measures (periodic simple returns, annualised with `ppy` periods per year).

Sharpe, Sortino, Treynor, information ratio, tracking error, Jensen's alpha, M² (Modigliani), Calmar, Omega,
maximum drawdown, up/down capture, modified Sharpe (Cornish-Fisher VaR), skewness and kurtosis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def cum(r):
    return (1 + pd.Series(r)).cumprod()


def drawdown(r) -> pd.Series:
    w = cum(r)
    return w / w.cummax() - 1


def cagr(r, ppy=12):
    r = np.asarray(r)
    return float(np.prod(1 + r) ** (ppy / len(r)) - 1)


def downside_dev(r, mar=0.0):
    x = np.minimum(np.asarray(r) - mar, 0)
    return float(np.sqrt((x ** 2).mean()))


def cornish_fisher_var(r, q=0.95):
    x = np.asarray(r)
    z = norm.ppf(1 - q)
    s, k = skew(x), kurtosis(x)              # excess kurtosis
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return float(-(x.mean() + zcf * x.std(ddof=1)))


def beta(r, m):
    c = np.cov(r, m, ddof=1)
    return float(c[0, 1] / c[1, 1])


def capture(r, b):
    r, b = np.asarray(r), np.asarray(b)
    up, dn = b > 0, b < 0
    u = (np.prod(1 + r[up]) ** (1 / up.sum()) - 1) / (np.prod(1 + b[up]) ** (1 / up.sum()) - 1) if up.any() else np.nan
    d = (np.prod(1 + r[dn]) ** (1 / dn.sum()) - 1) / (np.prod(1 + b[dn]) ** (1 / dn.sum()) - 1) if dn.any() else np.nan
    return float(u), float(d)


def omega(r, mar=0.0):
    x = np.asarray(r) - mar
    return float(np.clip(x, 0, None).sum() / max(np.clip(-x, 0, None).sum(), 1e-12))


def summary(r, b, rf, ppy: int = 12, mar_annual: float = 0.0) -> dict:
    """All measures for one return series r against benchmark b and risk-free rf (same length, periodic)."""
    r, b, rf = (np.asarray(x, float) for x in (r, b, rf))
    ex, exb, act = r - rf, b - rf, r - b
    mar = (1 + mar_annual) ** (1 / ppy) - 1
    vol = ex.std(ddof=1) * np.sqrt(ppy)
    volb = exb.std(ddof=1) * np.sqrt(ppy)
    sr = ex.mean() * ppy / vol
    srb = exb.mean() * ppy / volb
    be = beta(ex, exb)
    te = act.std(ddof=1) * np.sqrt(ppy)
    dd = drawdown(r)
    u, d = capture(r, b)
    out = {
        "annual return (CAGR)": cagr(r, ppy),
        "annual volatility": r.std(ddof=1) * np.sqrt(ppy),
        "excess return (ann.)": ex.mean() * ppy,
        "Sharpe ratio": sr,
        "Sortino ratio": (r.mean() - mar) * ppy / (downside_dev(r, mar) * np.sqrt(ppy)),
        "beta": be,
        "Treynor ratio": ex.mean() * ppy / be,
        "Jensen alpha (ann.)": (ex.mean() - be * exb.mean()) * ppy,
        "tracking error": te,
        "information ratio": act.mean() * ppy / te if te > 0 else np.nan,
        "M² (vs benchmark)": (sr - srb) * volb,
        "max drawdown": float(dd.min()),
        "Calmar ratio": cagr(r, ppy) / abs(float(dd.min())) if dd.min() < 0 else np.nan,
        "Omega ratio": omega(r, mar),
        "modified Sharpe (CF VaR 95 %)": ex.mean() / cornish_fisher_var(r, 0.95),
        "up capture": u,
        "down capture": d,
        "skewness": float(skew(r)),
        "excess kurtosis": float(kurtosis(r)),
        "hit ratio vs benchmark": float((act > 0).mean()),
    }
    return out


FMT = {"annual return (CAGR)": "{:.2%}", "annual volatility": "{:.2%}", "excess return (ann.)": "{:.2%}", "Sharpe ratio": "{:.2f}",
       "Sortino ratio": "{:.2f}", "beta": "{:.2f}", "Treynor ratio": "{:.2%}", "Jensen alpha (ann.)": "{:.2%}", "tracking error": "{:.2%}",
       "information ratio": "{:.2f}", "M² (vs benchmark)": "{:.2%}", "max drawdown": "{:.1%}", "Calmar ratio": "{:.2f}",
       "Omega ratio": "{:.2f}", "modified Sharpe (CF VaR 95 %)": "{:.2f}", "up capture": "{:.0%}", "down capture": "{:.0%}",
       "skewness": "{:.2f}", "excess kurtosis": "{:.2f}", "hit ratio vs benchmark": "{:.0%}"}


def table(returns: pd.DataFrame, funds: list[str], ppy=12, mar_annual=0.0) -> pd.DataFrame:
    cols = {f: summary(returns[f], returns["benchmark"], returns["rf"], ppy, mar_annual) for f in funds}
    cols["benchmark"] = summary(returns["benchmark"], returns["benchmark"], returns["rf"], ppy, mar_annual)
    return pd.DataFrame(cols)


def rolling(returns: pd.DataFrame, fund: str, window: int = 36, ppy=12) -> pd.DataFrame:
    r, b, rf = returns[fund], returns["benchmark"], returns["rf"]
    ex, exb = r - rf, b - rf
    out = pd.DataFrame(index=returns.index)
    out["Sharpe"] = ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(ppy)
    out["benchmark Sharpe"] = exb.rolling(window).mean() / exb.rolling(window).std() * np.sqrt(ppy)
    out["beta"] = ex.rolling(window).cov(exb) / exb.rolling(window).var()
    act = r - b
    out["information ratio"] = act.rolling(window).mean() / act.rolling(window).std() * np.sqrt(ppy)
    out["alpha (ann.)"] = (ex.rolling(window).mean() - out["beta"] * exb.rolling(window).mean()) * ppy
    return out
