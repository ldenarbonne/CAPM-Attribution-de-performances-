"""
Statistical inference on performance.

* Standard error of the Sharpe ratio under non-normal i.i.d. returns (Mertens 2002 / Lo 2002):
  Var(SR) ≈ (1 + SR²/2 − γ₃·SR + (γ₄ − 3)/4 · SR²) / T     (per-period SR, γ₄ = raw kurtosis)
* Probabilistic Sharpe ratio (Bailey & López de Prado 2012):
  PSR(SR*) = Φ( (SR − SR*)·√(T − 1) / √(1 − γ₃·SR + (γ₄ − 1)/4 · SR²) )
* Deflated Sharpe ratio (Bailey & López de Prado 2014): PSR against the expected maximum Sharpe of N independent
  trials, SR₀ = √V[SR_n] · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))), γ = Euler-Mascheroni.
* Jobson-Korkie test of equal Sharpe ratios with Memmel's (2003) correction.
* Circular block bootstrap of the Sharpe difference (robust to autocorrelation and fat tails, in the spirit of
  Ledoit & Wolf 2008; percentile interval, not studentised).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649


def sharpe(x) -> float:
    x = np.asarray(x)
    return float(x.mean() / x.std(ddof=1))


def sharpe_se(x) -> float:
    x = np.asarray(x)
    sr, g3, g4 = sharpe(x), skew(x), kurtosis(x, fisher=False)
    return float(np.sqrt((1 + 0.5 * sr ** 2 - g3 * sr + (g4 - 3) / 4 * sr ** 2) / len(x)))


def psr(x, sr_star: float = 0.0) -> float:
    """sr_star per period."""
    x = np.asarray(x)
    sr, g3, g4 = sharpe(x), skew(x), kurtosis(x, fisher=False)
    return float(norm.cdf((sr - sr_star) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    if n_trials <= 1:
        return 0.0
    return float(np.sqrt(var_trials) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def dsr(x, n_trials: int, var_trials: float) -> tuple[float, float]:
    """Deflated Sharpe ratio; var_trials = variance of the per-period Sharpe ratios across the trials."""
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return psr(x, sr0), sr0


def jobson_korkie(x, y) -> dict:
    x, y = np.asarray(x), np.asarray(y)
    T = len(x)
    a, b = sharpe(x), sharpe(y)
    rho = np.corrcoef(x, y)[0, 1]
    theta = (2 - 2 * rho + 0.5 * (a ** 2 + b ** 2 - 2 * a * b * rho ** 2)) / T
    z = (a - b) / np.sqrt(theta)
    return {"SR diff (per period)": a - b, "z": float(z), "p-value": float(2 * norm.sf(abs(z))), "correlation": float(rho)}


def block_bootstrap_sharpe_diff(x, y, n_boot: int = 2000, block: int = 6, seed: int = 0, ppy: int = 12) -> dict:
    x, y = np.asarray(x), np.asarray(y)
    T = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    d = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        d[i] = (sharpe(x[idx]) - sharpe(y[idx])) * np.sqrt(ppy)
    obs = (sharpe(x) - sharpe(y)) * np.sqrt(ppy)
    # two-sided p-value: share of centred bootstrap draws more extreme than the observation
    p = float(np.mean(np.abs(d - d.mean()) >= abs(obs)))
    return {"SR diff (ann.)": obs, "CI 2.5 %": float(np.quantile(d, 0.025)), "CI 97.5 %": float(np.quantile(d, 0.975)), "p-value": p,
            "draws": d}


def min_track_record(x, sr_star: float = 0.0, conf: float = 0.95) -> float:
    """Minimum track record length (periods) for SR > sr_star at the given confidence (Bailey & López de Prado)."""
    x = np.asarray(x)
    sr, g3, g4 = sharpe(x), skew(x), kurtosis(x, fisher=False)
    if sr <= sr_star:
        return np.inf
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (norm.ppf(conf) / (sr - sr_star)) ** 2)
