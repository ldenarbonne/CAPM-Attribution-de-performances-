import numpy as np
import pandas as pd
import pytest

from perf import attribution as AT
from perf import data as D
from perf import inference as INF
from perf import io as IO
from perf import metrics as MT
from perf import portfolio as PF
from perf import regression as RG


@pytest.fixture(scope="module")
def ds():
    return D.generate(120, 7)


def test_beta_alpha_recovered():
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    m = rng.normal(0.006, 0.04, n)
    y = 0.002 + 1.3 * m + rng.normal(0, 0.01, n)
    df = pd.DataFrame({"rf": 0.0, "benchmark": m, "F": y}, index=idx)
    r = RG.capm(df, "F")
    assert r.coef[1] == pytest.approx(1.3, abs=0.03)
    assert r.coef[0] == pytest.approx(0.002, abs=0.0008)


def test_newey_west_zero_lags_equals_white():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = X @ [1, -0.5] + rng.normal(size=200)
    r = RG.ols(y, X, ["a", "b"], lags=0)
    Xc = np.column_stack([np.ones(200), X])
    inv = np.linalg.inv(Xc.T @ Xc)
    white = inv @ (Xc * r.resid[:, None]).T @ (Xc * r.resid[:, None]) @ inv * 200 / 197
    assert np.allclose(r.se, np.sqrt(np.diag(white)))


def test_ratios_known_values():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.0])
    b = np.array([0.01, -0.01, 0.02, -0.01, 0.01, 0.0])
    s = MT.summary(r, b, np.zeros(6), ppy=12)
    assert s["Sharpe ratio"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(12))
    assert s["max drawdown"] == pytest.approx(-0.02)
    assert s["Omega ratio"] == pytest.approx(0.06 / 0.03)
    assert s["Sortino ratio"] == pytest.approx(r.mean() * 12 / (np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(12)))


def test_planted_skills_detected(ds):
    ret, _ = ds
    tm = RG.timing(ret, "Tactical Allocator", "Treynor-Mazuy")
    assert tm.coef[2] > 0 and tm.t[2] > 2
    ci = RG.capm(ret, "Closet Indexer")
    assert ci.coef[0] < 0 and ci.t[0] < -3                      # fees
    vs = RG.factor_model(ret, "Value Small Caps", "Fama-French 5 + momentum")
    assert vs.coef[2] > 0.15 and vs.coef[3] > 0.15              # SMB, HML tilts
    qg = RG.factor_model(ret, "Quality Growth", "Fama-French 5")
    assert qg.coef[4] > 0.2                                     # RMW (quality)


def test_factor_attribution_adds_up(ds):
    ret, _ = ds
    fa = RG.factor_attribution(ret, "Value Small Caps", "Carhart 4")
    assert fa.sum() == pytest.approx(fa.attrs["total"], rel=1e-9)


def test_brinson_single_period_exact(ds):
    _, sec = ds
    g = sec[(sec["fund"] == "Quality Growth") & (sec["date"] == sec["date"].max())]
    sp = AT.single_period(g)
    assert sp["total"].sum() == pytest.approx(sp.attrs["Rp"] - sp.attrs["Rb"], abs=1e-12)
    sp2 = AT.single_period(g, merge_interaction=True)
    assert sp2["total"].sum() == pytest.approx(sp.attrs["Rp"] - sp.attrs["Rb"], abs=1e-12)


@pytest.mark.parametrize("method", ["Carino", "Frongello"])
def test_linking_exact(ds, method):
    _, sec = ds
    e, t = AT.by_period(sec, "Value Small Caps")
    L = AT.link(e, t, method)
    assert L["total"].sum() == pytest.approx(L.attrs["active (compounded)"], abs=1e-10)
    ce = AT.cumulative_effects(e, t)
    assert ce[AT.EFFECTS].iloc[-1].sum() == pytest.approx(ce["active (compounded)"].iloc[-1], abs=1e-10)


def test_fund_return_consistent_with_sectors(ds):
    ret, sec = ds
    _, t = AT.by_period(sec, "Value Small Caps")
    fee = D.FUNDS["Value Small Caps"]["fee"] / 12
    assert np.allclose(t["Rp"].to_numpy() - fee, ret["Value Small Caps"].to_numpy())


def test_sharpe_inference():
    rng = np.random.default_rng(3)
    x = rng.normal(0.01, 0.04, 100_00)
    assert INF.sharpe_se(x) == pytest.approx(np.sqrt((1 + 0.5 * 0.25 ** 2) / len(x)), rel=0.05)
    assert INF.psr(x, 0.0) > 0.99
    assert INF.expected_max_sharpe(1, 0.01) == 0 and INF.expected_max_sharpe(100, 0.01) > INF.expected_max_sharpe(10, 0.01)
    jk = INF.jobson_korkie(x, x)
    assert jk["z"] == pytest.approx(0.0)


def test_frontier(ds):
    _, sec = ds
    sr = sec[(sec["fund"] == "Closet Indexer") & (sec["sector"] != "Cash")].pivot(index="date", columns="sector", values="r_bench")
    S = PF.cov_matrix(sr, 0.2).to_numpy()
    mu = sr.mean().to_numpy() * 12
    lo = PF.long_only(mu, S, 0.01)
    an = PF.analytic(mu, S, 0.01)
    assert lo["gmv"].min() >= -1e-8 and lo["gmv"].sum() == pytest.approx(1)
    # long-only GMV variance ≥ unconstrained GMV variance
    assert lo["gmv"] @ S @ lo["gmv"] >= an["gmv"] @ S @ an["gmv"] - 1e-12
    srt = lambda w: (w @ mu - 0.01) / np.sqrt(w @ S @ w)  # noqa: E731
    assert srt(an["tangency"]) >= srt(lo["tangency"]) - 1e-6


def test_risk_contributions_add_up(ds):
    ret, sec = ds
    sr = sec[(sec["fund"] == "Closet Indexer") & (sec["sector"] != "Cash")].pivot(index="date", columns="sector", values="r_bench")
    ar = PF.active_risk(sec, "Quality Growth", sr, ret["rf"])
    assert ar["total contribution"].sum() == pytest.approx(ar.attrs["TE"])
    v, d = PF.var_contributions(sec, "Quality Growth", sr, ret["rf"], 0.99)
    assert v["component VaR"].sum() == pytest.approx(d["parametric VaR"])
    fr = PF.factor_risk(ret, "Value Small Caps", D.FACTORS)
    assert fr.sum() == pytest.approx(fr.attrs["vol"])


def test_csv_roundtrip(ds):
    ret, sec = ds
    r2, s2 = IO.load(ret.reset_index().to_csv(index=False).encode(), sec.to_csv(index=False).encode())
    assert np.allclose(r2.to_numpy(), ret.to_numpy())
    with pytest.raises(ValueError):
        IO.load(ret.drop(columns="benchmark").reset_index().to_csv(index=False).encode())
