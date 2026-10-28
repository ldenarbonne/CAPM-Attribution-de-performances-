"""CAPM & Performance Lab – CAPM, risk-adjusted ratios, Brinson attribution, factor models, frontier and risk contributions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from perf import attribution as AT
from perf import data as D
from perf import inference as INF
from perf import io as IO
from perf import metrics as MT
from perf import portfolio as PF
from perf import regression as RG

st.set_page_config(page_title="CAPM & Performance Lab", page_icon="📈", layout="wide")
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#3a5a8c", "#d4892b", "#3b8c5a", "#b8433a", "#9aa5b1", "#7b5ea7"
PAL = [BLUE, ORANGE, GREEN, PURPLE, RED, "#2a9d8f", "#8d6e63"]
PPY = 12
LEG = dict(orientation="h", y=-0.22)


def h(df):
    return min(35 * (len(df) + 1) + 3, 900)


def text(df: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    out = df.copy()
    out.attrs = {}
    for c in out.columns:
        f = fmt.get(c)
        if f is not None:
            out[c] = [("–" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f.format(v)) for v in df[c]]
    return out


def rowfmt(df: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    """Format a table whose rows are measures (fmt keyed by row label)."""
    out = df.astype(object).copy()
    out.attrs = {}
    for r in out.index:
        f = fmt.get(r, "{:.4f}")
        out.loc[r] = ["–" if (v is None or not np.isfinite(v)) else f.format(v) for v in df.loc[r]]
    return out


@st.cache_data(show_spinner="Generating the funds…")
def synthetic(seed, months):
    return D.generate(months, seed)


# ---------------------------------------------------------------- data and sidebar
with st.sidebar:
    st.header("Data")
    if "imported" in st.session_state:
        st.success(f"Imported: {st.session_state['imported_name']}")
        if st.button("Back to synthetic data"):
            del st.session_state["imported"]
            st.rerun()
    seed = st.number_input("Seed", 0, 9999, 21, disabled="imported" in st.session_state)
    months = st.slider("History (months)", 60, 240, 144, 12, disabled="imported" in st.session_state)

if "imported" in st.session_state:
    returns_all, sectors_all = st.session_state["imported"]
else:
    returns_all, sectors_all = synthetic(seed, months)

funds = D.fund_names(returns_all)
with st.sidebar:
    d0, d1 = returns_all.index.min().to_pydatetime(), returns_all.index.max().to_pydatetime()
    rng_ = st.slider("Analysis period", d0, d1, (d0, d1), format="MMM YYYY")
    window = st.slider("Rolling window (months)", 12, 60, 36, 6)
    mar = st.number_input("Minimum acceptable return, MAR (annual, for Sortino and Omega)", -0.10, 0.20, 0.0, 0.005, format="%.3f")
    st.caption("Monthly simple returns; annualisation × 12 (means) and × √12 (volatilities).")

returns = returns_all.loc[rng_[0]:rng_[1]]
sectors = None if sectors_all is None else sectors_all[(sectors_all["date"] >= rng_[0]) & (sectors_all["date"] <= rng_[1])]
if len(returns) < 24:
    st.error("Select at least 24 months.")
    st.stop()
has_factors = all(c in returns.columns for c in D.FACTORS)
sector_ret = None
if sectors is not None:
    sector_ret = sectors[(sectors["fund"] == sectors["fund"].iloc[0]) & (sectors["sector"] != "Cash")].pivot(index="date", columns="sector", values="r_bench")

st.title("CAPM & Performance Lab")
st.caption("Performance measurement of active equity funds against their benchmark: CAPM and market timing, risk-adjusted "
           "ratios with statistical tests, Brinson-Fachler attribution linked over time, multifactor models, the efficient "
           "frontier and ex ante risk contributions.")

tabs = st.tabs(["Overview", "CAPM", "Performance ratios", "Statistical tests", "Brinson attribution", "Factor models",
                "Efficient frontier", "Risk contributions", "Data & import", "Theory"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    tb = MT.table(returns, funds, PPY, mar)
    k = st.columns(len(funds) + 1)
    for i, f in enumerate(funds + ["benchmark"]):
        k[i].metric(f, f"{tb.loc['annual return (CAGR)', f]:.1%} p.a.")
        k[i].caption(f"Sharpe {tb.loc['Sharpe ratio', f]:.2f} · max drawdown {tb.loc['max drawdown', f]:.0%}")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for i, f in enumerate(funds + ["benchmark"]):
            fig.add_scatter(x=returns.index, y=MT.cum(returns[f]).values * 100, name=f,
                            line=dict(color=PAL[i] if f != "benchmark" else "black", width=2.5 if f == "benchmark" else 1.8,
                                      dash="dash" if f == "benchmark" else None))
        fig.update_layout(height=420, margin=dict(t=30, b=10), yaxis_title="growth of 100", legend=LEG, title="Cumulative performance (net of fees)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        for i, f in enumerate(funds + ["benchmark"]):
            fig.add_scatter(x=returns.index, y=MT.drawdown(returns[f]).values, name=f,
                            line=dict(color=PAL[i] if f != "benchmark" else "black", width=1.5, dash="dash" if f == "benchmark" else None))
        fig.update_layout(height=420, margin=dict(t=30, b=10), yaxis_tickformat=".0%", legend=LEG, title="Drawdown", showlegend=False)
        st.plotly_chart(fig, width="stretch")
    keys = ["annual return (CAGR)", "annual volatility", "Sharpe ratio", "Sortino ratio", "beta", "Jensen alpha (ann.)", "tracking error",
            "information ratio", "max drawdown"]
    st.dataframe(rowfmt(tb.loc[keys], MT.FMT), width="stretch", height=h(tb.loc[keys]))
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=[tb.loc["annual volatility", f] for f in funds + ["benchmark"]], y=[tb.loc["annual return (CAGR)", f] for f in funds + ["benchmark"]],
                         text=funds + ["benchmark"])
        fig.update_traces(marker=dict(size=13, color=PAL[: len(funds)] + ["black"]), textposition="top center")
        vols = [tb.loc["annual volatility", f] for f in funds + ["benchmark"]]
        fig.update_layout(height=360, margin=dict(t=30, b=10), xaxis_tickformat=".1%", yaxis_tickformat=".1%", xaxis_title="volatility",
                          yaxis_title="annual return", title="Return vs risk", xaxis_range=[min(vols) - 0.01, max(vols) + 0.01])
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.scatter(x=[tb.loc["tracking error", f] for f in funds], y=[(returns[f] - returns["benchmark"]).mean() * PPY for f in funds], text=funds)
        fig.update_traces(marker=dict(size=13, color=PAL[: len(funds)]), textposition="top center")
        for ir in [-0.5, 0.5]:
            xs = np.linspace(0, max(tb.loc["tracking error", funds]) * 1.2, 10)
            fig.add_scatter(x=xs, y=ir * xs, mode="lines", line=dict(color=GREY, dash="dot"), name=f"IR {ir:+.1f}", showlegend=False)
        fig.update_layout(height=360, margin=dict(t=30, b=10), xaxis_tickformat=".1%", yaxis_tickformat=".1%", xaxis_title="tracking error",
                          yaxis_title="active return (ann.)", title="Active return vs tracking error (dotted: IR ±0.5)")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- CAPM
with tabs[1]:
    fund = st.selectbox("Fund", funds, key="capm_fund")
    res = RG.capm(returns, fund)
    exb = returns["benchmark"] - returns["rf"]
    exf = returns[fund] - returns["rf"]
    k = st.columns(5)
    k[0].metric("Jensen's alpha (ann.)", f"{res.coef[0] * PPY:+.2%}", f"t = {res.t[0]:.2f} (Newey-West)", delta_color="off")
    k[1].metric("Beta", f"{res.coef[1]:.3f}", f"t = {res.t[1]:.1f}", delta_color="off")
    k[2].metric("R²", f"{res.r2:.1%}")
    k[3].metric("Residual risk (ann.)", f"{res.resid.std(ddof=2) * np.sqrt(PPY):.2%}")
    k[4].metric("Appraisal ratio", f"{res.coef[0] / res.resid.std(ddof=2) * np.sqrt(PPY):.2f}", help="α / σ(ε), annualised (Treynor-Black).")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=exb, y=exf, mode="markers", marker=dict(color=BLUE, opacity=0.6), name="monthly excess returns",
                        text=[d.strftime("%b %Y") for d in returns.index])
        xs = np.linspace(exb.min(), exb.max(), 20)
        fig.add_scatter(x=xs, y=res.coef[0] + res.coef[1] * xs, mode="lines", line=dict(color=RED), name=f"α + βx, β = {res.coef[1]:.2f}")
        fig.add_scatter(x=xs, y=xs, mode="lines", line=dict(color=GREY, dash="dot"), name="β = 1, α = 0")
        fig.update_layout(height=430, margin=dict(t=30, b=10), xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend=LEG,
                          xaxis_title="benchmark excess return", yaxis_title="fund excess return", title="Characteristic line")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Regression r − r_f = α + β(r_m − r_f) + ε**")
        t = res.table(PPY)
        st.dataframe(text(t, {"coefficient": "{:.5f}", "NW std error": "{:.5f}", "t (NW)": "{:.2f}", "p-value": "{:.3f}", "t (OLS)": "{:.2f}", "annualised": "{:.2%}"}),
                     width="stretch")
        st.caption(f"Newey-West HAC standard errors with {res.lags} lags; n = {res.n} months.")
        st.markdown("**Market timing**")
        rows = []
        for kind in ["Treynor-Mazuy", "Henriksson-Merton"]:
            r2 = RG.timing(returns, fund, kind)
            rows.append({"model": kind, "alpha (ann.)": r2.coef[0] * PPY, "beta": r2.coef[1], "timing γ": r2.coef[2], "t(γ)": r2.t[2], "p(γ)": r2.p[2]})
        tt = pd.DataFrame(rows).set_index("model")
        st.dataframe(text(tt, {"alpha (ann.)": "{:+.2%}", "beta": "{:.3f}", "timing γ": "{:.3f}", "t(γ)": "{:.2f}", "p(γ)": "{:.3f}"}), width="stretch")
        st.caption("γ > 0 and significant: the fund raises its market exposure before up-markets. A positive timing γ often "
                   "comes with a negative selection α in these models (the convexity is paid for).")
    c1, c2 = st.columns(2)
    with c1:
        ro = MT.rolling(returns, fund, window, PPY)
        fig = go.Figure()
        fig.add_scatter(x=ro.index, y=ro["beta"], name="beta", line_color=BLUE)
        fig.add_scatter(x=ro.index, y=ro["alpha (ann.)"], name="alpha (ann.)", line_color=ORANGE, yaxis="y2")
        fig.update_layout(height=360, margin=dict(t=30, b=10), legend=LEG, title=f"Rolling {window}-month beta and alpha",
                          yaxis=dict(title="beta"), yaxis2=dict(title="alpha", overlaying="y", side="right", tickformat=".0%", showgrid=False))
        st.plotly_chart(fig, width="stretch")
    with c2:
        assets = returns[funds].copy()
        if sector_ret is not None:
            assets = assets.join(sector_ret)
        sm = RG.sml(returns, assets, PPY)
        mp = sm.attrs["market premium"]
        fig = go.Figure()
        xs = np.linspace(0, max(1.6, sm["beta"].max() + 0.1), 10)
        fig.add_scatter(x=xs, y=mp * xs, mode="lines", line=dict(color=RED), name="SML: β × market premium")
        isf = sm.index.isin(funds)
        fig.add_scatter(x=sm.loc[~isf, "beta"], y=sm.loc[~isf, "mean excess (ann.)"], mode="markers+text", text=sm.index[~isf], textposition="top center",
                        marker=dict(color=GREY, size=9), name="sector indices", textfont=dict(size=9))
        fig.add_scatter(x=sm.loc[isf, "beta"], y=sm.loc[isf, "mean excess (ann.)"], mode="markers+text", text=sm.index[isf], textposition="bottom center",
                        marker=dict(color=BLUE, size=12, symbol="diamond"), name="funds")
        fig.add_scatter(x=[1], y=[mp], mode="markers", marker=dict(color="black", size=12), name="benchmark")
        fig.update_layout(height=360, margin=dict(t=30, b=10), yaxis_tickformat=".0%", legend=LEG, xaxis_title="beta",
                          yaxis_title="mean excess return (ann.)", title="Security market line (realised)")
        st.plotly_chart(fig, width="stretch")
    st.caption("Vertical distance to the SML = Jensen's alpha. With realised returns the SML is a test of the CAPM on this "
               "sample: assets above it earned more than their beta justified.")

# ---------------------------------------------------------------- ratios
with tabs[2]:
    tb = MT.table(returns, funds, PPY, mar)
    st.dataframe(rowfmt(tb, MT.FMT), width="stretch", height=h(tb))
    with st.expander("Definitions"):
        st.markdown(r"""
| Ratio | Formula | Reads as |
|---|---|---|
| Sharpe | $(\bar r - r_f)/\sigma$ | excess return per unit of total risk |
| Sortino | $(\bar r - \text{MAR})/\sigma_{down}$ | excess over the MAR per unit of downside deviation |
| Treynor | $(\bar r - r_f)/\beta$ | excess return per unit of systematic risk |
| Information ratio | $\overline{r-r_b}/\text{TE}$ | active return per unit of active risk |
| Jensen's α | $\bar r - r_f - \beta(\bar r_b - r_f)$ | return above the CAPM prediction |
| M² | $(SR_p - SR_b)\,\sigma_b$ | Sharpe gap expressed as a return at benchmark risk |
| Calmar | CAGR / \|max drawdown\| | return per unit of worst loss |
| Omega | $E[(r-L)^+]/E[(L-r)^+]$ | gains over losses around the threshold L = MAR |
| Modified Sharpe | $(\bar r - r_f)/\text{VaR}_{CF}$ | excess return per unit of Cornish-Fisher VaR (skew and kurtosis) |
| Up / down capture | geometric mean in up (down) benchmark months / benchmark | asymmetry of the participation |
""")
    c1, c2 = st.columns(2)
    with c1:
        what = st.selectbox("Rolling measure", ["Sharpe", "information ratio", "beta", "alpha (ann.)"])
        fig = go.Figure()
        for i, f in enumerate(funds):
            fig.add_scatter(x=returns.index, y=MT.rolling(returns, f, window, PPY)[what], name=f, line_color=PAL[i])
        if what == "Sharpe":
            fig.add_scatter(x=returns.index, y=MT.rolling(returns, funds[0], window, PPY)["benchmark Sharpe"], name="benchmark",
                            line=dict(color="black", dash="dash"))
        fig.update_layout(height=380, margin=dict(t=30, b=10), legend=LEG, title=f"Rolling {window}-month {what}",
                          yaxis_tickformat=".0%" if what == "alpha (ann.)" else None)
        st.plotly_chart(fig, width="stretch")
    with c2:
        f = st.selectbox("Return distribution of", funds, key="dist")
        r = returns[f]
        fig = go.Figure()
        fig.add_histogram(x=r, nbinsx=40, histnorm="probability density", marker_color=BLUE, opacity=0.75, name=f)
        xs = np.linspace(r.min(), r.max(), 200)
        from scipy.stats import norm as _n
        fig.add_scatter(x=xs, y=_n.pdf(xs, r.mean(), r.std()), line_color=RED, name="normal fit")
        fig.add_vline(x=-MT.cornish_fisher_var(r, 0.95), line_dash="dash", line_color=ORANGE, annotation_text="CF VaR 95 %")
        fig.add_vline(x=np.quantile(r, 0.05), line_dash="dot", line_color=GREY, annotation_text="hist. 5 %")
        fig.update_layout(height=380, margin=dict(t=30, b=10), xaxis_tickformat=".0%", legend=LEG, title="Monthly return distribution")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Ranking consistency")
    keys = ["Sharpe ratio", "Sortino ratio", "Treynor ratio", "information ratio", "Jensen alpha (ann.)", "M² (vs benchmark)", "Calmar ratio",
            "Omega ratio", "modified Sharpe (CF VaR 95 %)"]
    rk = tb.loc[keys, funds].rank(axis=1, ascending=False).astype(int)
    st.dataframe(rk, width="stretch", height=h(rk))
    st.caption("Rank of each fund (1 = best) under each measure. Rankings agree when risk is mostly systematic and returns "
               "close to normal; they diverge with very different betas, tracking errors or return asymmetry.")

# ---------------------------------------------------------------- statistical tests
with tabs[3]:
    st.markdown("Is an observed Sharpe ratio, alpha or Sharpe difference distinguishable from luck?")
    c1, c2, c3 = st.columns(3)
    sr_star = c1.number_input("Benchmark Sharpe SR* (annual) for PSR", -1.0, 2.0, 0.0, 0.05)
    n_trials = c2.number_input("Number of strategies tried (for the deflated Sharpe)", 1, 10000, 20)
    sd_trials = c3.number_input("Std dev of annual Sharpe across trials", 0.0, 2.0, 0.30, 0.05)
    rows = []
    for f in funds + ["benchmark"]:
        x = (returns[f] - returns["rf"]).to_numpy()
        sr = INF.sharpe(x)
        se = INF.sharpe_se(x)
        dsr, sr0 = INF.dsr(x, int(n_trials), (sd_trials / np.sqrt(PPY)) ** 2)
        a = RG.capm(returns, f) if f != "benchmark" else None
        rows.append({"fund": f, "Sharpe (ann.)": sr * np.sqrt(PPY), "std error (ann.)": se * np.sqrt(PPY),
                     "95 % CI low": (sr - 1.96 * se) * np.sqrt(PPY), "95 % CI high": (sr + 1.96 * se) * np.sqrt(PPY),
                     "PSR(SR*)": INF.psr(x, sr_star / np.sqrt(PPY)), "deflated Sharpe": dsr,
                     "min track record (years)": INF.min_track_record(x, sr_star / np.sqrt(PPY)) / PPY,
                     "alpha t (NW)": a.t[0] if a else np.nan})
    st_ = pd.DataFrame(rows).set_index("fund")
    st.dataframe(text(st_, {"Sharpe (ann.)": "{:.2f}", "std error (ann.)": "{:.2f}", "95 % CI low": "{:.2f}", "95 % CI high": "{:.2f}",
                            "PSR(SR*)": "{:.1%}", "deflated Sharpe": "{:.1%}", "min track record (years)": "{:.1f}", "alpha t (NW)": "{:.2f}"}),
                 width="stretch", height=h(st_))
    st.caption(f"Expected maximum annual Sharpe of {int(n_trials)} unskilled trials: "
               f"{INF.expected_max_sharpe(int(n_trials), (sd_trials / np.sqrt(PPY)) ** 2) * np.sqrt(PPY):.2f}. "
               "PSR = probability that the true Sharpe exceeds SR* given the sample length, skewness and kurtosis; the deflated "
               "Sharpe replaces SR* by that expected maximum (selection bias). Minimum track record = years needed for PSR ≥ 95 %.")
    st.subheader("Is the fund's Sharpe ratio different from another one?")
    c1, c2 = st.columns(2)
    fa = c1.selectbox("Fund", funds, key="t_a")
    fb = c2.selectbox("Against", ["benchmark"] + [f for f in funds if f != fa], key="t_b")
    xa, xb = (returns[fa] - returns["rf"]).to_numpy(), (returns[fb] - returns["rf"]).to_numpy()
    jk = INF.jobson_korkie(xa, xb)
    bb = INF.block_bootstrap_sharpe_diff(xa, xb, 2000, 6, 0, PPY)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric("Sharpe difference (ann.)", f"{jk['SR diff (per period)'] * np.sqrt(PPY):+.3f}")
        st.metric("Jobson-Korkie-Memmel p-value", f"{jk['p-value']:.3f}", f"z = {jk['z']:.2f}, ρ = {jk['correlation']:.2f}", delta_color="off")
        st.metric("Block bootstrap p-value", f"{bb['p-value']:.3f}", f"95 % CI [{bb['CI 2.5 %']:+.2f}, {bb['CI 97.5 %']:+.2f}]", delta_color="off")
        st.caption("The JKM test assumes i.i.d. normal returns; the circular block bootstrap (blocks of 6 months, 2,000 draws) "
                   "keeps autocorrelation, volatility clustering and fat tails.")
    with c2:
        fig = go.Figure(go.Histogram(x=bb["draws"], nbinsx=50, marker_color=BLUE, opacity=0.8))
        fig.add_vline(x=0, line_color="black")
        fig.add_vline(x=bb["SR diff (ann.)"], line_color=RED, annotation_text="observed")
        fig.update_layout(height=320, margin=dict(t=30, b=10), xaxis_title="bootstrap Sharpe difference (ann.)", title="Bootstrap distribution")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- attribution
with tabs[4]:
    if sectors is None:
        st.info("Import a sector file to run the attribution.")
    else:
        c1, c2, c3 = st.columns(3)
        fa = c1.selectbox("Fund", sorted(sectors["fund"].unique()), key="att_f")
        method = c2.radio("Multi-period linking", ["Carino", "Frongello", "Arithmetic (unlinked)"], horizontal=True)
        merge = c3.checkbox("Selection includes interaction (w_p in the selection term)", value=False)
        eff, tot = AT.by_period(sectors, fa, merge)
        L = AT.link(eff, tot, method)
        net = returns[fa].reindex(tot.index)
        k = st.columns(5)
        k[0].metric("Portfolio return (gross, compounded)", f"{L.attrs['Rp']:.1%}")
        k[1].metric("Benchmark return", f"{L.attrs['Rb']:.1%}")
        k[2].metric("Active return (gross)", f"{L.attrs['active (compounded)']:+.2%}")
        k[3].metric("Sum of linked effects", f"{L['total'].sum():+.2%}",
                    "exact" if abs(L["total"].sum() - L.attrs["active (compounded)"]) < 1e-9 else "residual (unlinked)", delta_color="off")
        k[4].metric("Fees (net − gross, compounded)", f"{np.prod(1 + net) - 1 - L.attrs['Rp']:+.2%}")
        c1, c2 = st.columns([3, 2])
        with c1:
            Ls = L.drop(columns="total").sort_values("allocation")
            fig = go.Figure()
            for e, col in zip(AT.EFFECTS, [BLUE, ORANGE, GREY]):
                fig.add_bar(y=Ls.index, x=Ls[e], name=e, orientation="h", marker_color=col)
            fig.update_layout(barmode="relative", height=480, margin=dict(t=30, b=10), xaxis_tickformat=".1%", legend=LEG,
                              title=f"Linked effects by sector ({method})")
            st.plotly_chart(fig, width="stretch")
        with c2:
            t = L.copy()
            t.loc["Total"] = t.sum()
            st.dataframe(text(t, {c: "{:+.2%}" for c in t.columns}), width="stretch", height=h(t))
        ce = AT.cumulative_effects(eff, tot)
        fig = go.Figure()
        for e, col in zip(AT.EFFECTS, [BLUE, ORANGE, GREY]):
            fig.add_scatter(x=ce.index, y=ce[e], name=e, line_color=col, stackgroup="one")
        fig.add_scatter(x=ce.index, y=ce["active (compounded)"], name="active return (compounded)", line=dict(color="black", width=2.5))
        fig.update_layout(height=360, margin=dict(t=30, b=10), yaxis_tickformat=".0%", legend=LEG,
                          title="Cumulative attribution through time (Frongello linking)")
        st.plotly_chart(fig, width="stretch")
        st.subheader("Single month")
        dsel = st.select_slider("Month", options=list(tot.index), value=tot.index[-1], format_func=lambda d: d.strftime("%b %Y"))
        sp = AT.single_period(sectors[(sectors["fund"] == fa) & (sectors["date"] == dsel)], merge)
        sp.loc["Total"] = [sp["w_port"].sum(), sp["w_bench"].sum(), sp["active weight"].sum(), sp.attrs["Rp"], sp.attrs["Rb"],
                           sp["allocation"].sum(), sp["selection"].sum(), sp["interaction"].sum(), sp["total"].sum()]
        st.dataframe(text(sp, {"w_port": "{:.1%}", "w_bench": "{:.1%}", "active weight": "{:+.1%}", "r_port": "{:+.2%}", "r_bench": "{:+.2%}",
                               "allocation": "{:+.3%}", "selection": "{:+.3%}", "interaction": "{:+.3%}", "total": "{:+.3%}"}), width="stretch", height=h(sp))
        st.caption("Brinson-Fachler: allocation is measured against the total benchmark return, so overweighting a sector "
                   "that beats the benchmark adds value even if the sector fell. Cash is a sector with zero benchmark weight.")

# ---------------------------------------------------------------- factor models
with tabs[5]:
    if not has_factors:
        st.info("The factor columns (MKT_RF, SMB, HML, RMW, CMA, MOM) are needed for this tab.")
    else:
        c1, c2, c3 = st.columns(3)
        ff = c1.selectbox("Fund", funds, key="ff_f")
        model = c2.selectbox("Model", list(RG.MODELS), index=4)
        active = c3.radio("Dependent variable", ["excess over risk-free", "active vs benchmark"], horizontal=True) == "active vs benchmark"
        res = RG.factor_model(returns, ff, model, active)
        k = st.columns(4)
        k[0].metric("Alpha (ann.)", f"{res.coef[0] * PPY:+.2%}", f"t = {res.t[0]:.2f} (NW)", delta_color="off")
        k[1].metric("R²", f"{res.r2:.1%}")
        k[2].metric("Adjusted R²", f"{res.r2_adj:.1%}")
        k[3].metric("Residual volatility (ann.)", f"{res.resid.std(ddof=len(res.coef)) * np.sqrt(PPY):.2%}")
        c1, c2 = st.columns(2)
        with c1:
            names = res.names[1:]
            b, se = res.coef[1:], res.se[1:]
            fig = go.Figure(go.Bar(x=names, y=b, error_y=dict(type="data", array=1.96 * se), marker_color=[GREEN if v > 0 else RED for v in b]))
            fig.update_layout(height=380, margin=dict(t=30, b=10), title="Factor loadings (95 % NW confidence intervals)")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fa_ = RG.factor_attribution(returns, ff, model, PPY, active)
            fig = go.Figure(go.Waterfall(x=list(fa_.index) + ["total"], y=list(fa_.values) + [0], measure=["relative"] * len(fa_) + ["total"],
                                         text=[f"{v:+.2%}" for v in fa_.values] + [f"{fa_.attrs['total']:+.2%}"],
                                         increasing_marker_color=GREEN, decreasing_marker_color=RED, totals_marker_color=BLUE))
            fig.update_layout(height=380, margin=dict(t=30, b=10), yaxis_tickformat=".1%",
                              title=("Active" if active else "Excess") + " return explained: α + Σ β·mean(factor), annualised")
            st.plotly_chart(fig, width="stretch")
        t = res.table(PPY)
        st.dataframe(text(t, {"coefficient": "{:.4f}", "NW std error": "{:.4f}", "t (NW)": "{:.2f}", "p-value": "{:.3f}", "t (OLS)": "{:.2f}", "annualised": "{:+.2%}"}),
                     width="stretch")
        st.subheader("Alpha across models")
        rows = {}
        for f in funds:
            rows[f] = {m: RG.factor_model(returns, f, m, active) for m in RG.MODELS}
        al = pd.DataFrame({f: {(m + " (MKT_RF factor)" if m == "CAPM" else m): f"{r.coef[0] * PPY:+.2%} (t {r.t[0]:.1f})" for m, r in d.items()}
                           for f, d in rows.items()})
        st.dataframe(al, width="stretch", height=h(al))
        st.caption("An alpha that disappears when style factors are added was a factor exposure (style beta), not skill. "
                   "The CAPM row uses the broad market factor MKT_RF, whereas the CAPM tab uses the fund's benchmark as the "
                   "market proxy: the alphas differ — Roll's critique in practice.")
        rb_ = RG.rolling_betas(returns, ff, model, window)
        fig = go.Figure()
        for i, c in enumerate(RG.MODELS[model]):
            fig.add_scatter(x=rb_.index, y=rb_[c], name=c, line_color=PAL[i % len(PAL)])
        fig.update_layout(height=360, margin=dict(t=30, b=10), legend=LEG, title=f"Rolling {window}-month factor exposures (excess-return regression)")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- frontier
with tabs[6]:
    if sector_ret is None:
        st.info("Sector index returns (from the sector file) are needed for the frontier.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        shrink = c1.slider("Covariance shrinkage towards the diagonal", 0.0, 1.0, 0.2, 0.05)
        wmax = c2.slider("Maximum weight per sector (long-only)", 0.1, 1.0, 1.0, 0.05)
        mu_src = c3.radio("Expected returns", ["historical means", "CAPM-implied (β × premium)"])
        show_unc = c4.checkbox("Show unconstrained frontier", value=True)
        S = PF.cov_matrix(sector_ret, shrink, PPY)
        rf_a = returns["rf"].mean() * PPY
        if mu_src == "historical means":
            mu = sector_ret.mean().to_numpy() * PPY
        else:
            smx = RG.sml(returns, sector_ret, PPY)
            mu = rf_a + smx["beta"].to_numpy() * smx.attrs["market premium"]
        lo = PF.long_only(mu, S.to_numpy(), rf_a, 30, wmax)
        an = PF.analytic(mu, S.to_numpy(), rf_a)
        fig = go.Figure()
        if show_unc:
            fr = an["frontier"]
            fig.add_scatter(x=fr["vol"], y=fr["return"], mode="lines", line=dict(color=GREY, dash="dot"), name="unconstrained frontier")
        fr = lo["frontier"]
        fig.add_scatter(x=fr["vol"], y=fr["return"], mode="lines", line=dict(color=BLUE, width=3), name="long-only frontier")
        tr, tv = PF.point(lo["tangency"], mu, S.to_numpy())
        xs = np.linspace(0, max(fr["vol"].max(), tv) * 1.15, 20)
        fig.add_scatter(x=xs, y=rf_a + (tr - rf_a) / tv * xs, mode="lines", line=dict(color=RED), name=f"CML (Sharpe {(tr - rf_a) / tv:.2f})")
        fig.add_scatter(x=[tv], y=[tr], mode="markers", marker=dict(color=RED, size=13, symbol="star"), name="tangency")
        gr, gv = PF.point(lo["gmv"], mu, S.to_numpy())
        fig.add_scatter(x=[gv], y=[gr], mode="markers", marker=dict(color=GREEN, size=12, symbol="diamond"), name="minimum variance")
        sv = np.sqrt(np.diag(S.to_numpy()))
        fig.add_scatter(x=sv, y=mu, mode="markers+text", text=list(sector_ret.columns), textposition="top center", textfont=dict(size=9),
                        marker=dict(color=GREY, size=8), name="sectors")
        wb_last = sectors[(sectors["fund"] == sectors["fund"].iloc[0]) & (sectors["date"] == sectors["date"].max()) & (sectors["sector"] != "Cash")] \
            .set_index("sector")["w_bench"].reindex(sector_ret.columns).to_numpy()
        br, bv = PF.point(wb_last / wb_last.sum(), mu, S.to_numpy())
        fig.add_scatter(x=[bv], y=[br], mode="markers", marker=dict(color="black", size=12), name="benchmark (current weights)")
        fig.update_layout(height=520, margin=dict(t=30, b=10), xaxis_tickformat=".0%", yaxis_tickformat=".1%", xaxis_title="volatility (ann.)",
                          yaxis_title="expected return (ann.)", legend=LEG, title="Mean-variance frontier of the sector indices",
                          xaxis_range=[0, max(sv.max(), fr["vol"].max()) * 1.1],
                          yaxis_range=[min(mu.min(), rf_a) - 0.01, max(mu.max(), fr["return"].max()) + 0.02])
        st.plotly_chart(fig, width="stretch")
        w = pd.DataFrame({"tangency": lo["tangency"], "minimum variance": lo["gmv"], "benchmark": wb_last / wb_last.sum(),
                          "expected return": mu, "volatility": sv}, index=sector_ret.columns)
        st.dataframe(text(w, {c: "{:.1%}" for c in w.columns}), width="stretch", height=h(w))
        st.caption("With CAPM-implied returns and a benchmark close to the market, the tangency portfolio moves towards the "
                   "benchmark weights: the market portfolio is mean-variance efficient by construction (two-fund separation). "
                   "Historical means produce concentrated, unstable portfolios — the classic error-maximisation problem.")

# ---------------------------------------------------------------- risk contributions
with tabs[7]:
    if sectors is None or sector_ret is None:
        st.info("A sector file is needed for risk contributions.")
    else:
        c1, c2, c3 = st.columns(3)
        fr_ = c1.selectbox("Fund", sorted(sectors["fund"].unique()), key="risk_f")
        cov_m = c2.radio("Covariance", ["sample", "EWMA λ = 0.97"], horizontal=True)
        q = c3.select_slider("VaR confidence", [0.95, 0.975, 0.99], value=0.99)
        ar = PF.active_risk(sectors, fr_, sector_ret, returns["rf"].reindex(sector_ret.index), PPY, 0.0, 0.97 if cov_m.startswith("EWMA") else None)
        realised_te = (returns[fr_] - returns["benchmark"]).std(ddof=1) * np.sqrt(PPY)
        k = st.columns(4)
        k[0].metric("Ex ante tracking error", f"{ar.attrs['TE']:.2%}")
        k[1].metric("of which allocation (sector bets)", f"{ar.attrs['allocation']:.2%}")
        k[2].metric("of which selection (within sectors)", f"{ar.attrs['selection']:.2%}")
        k[3].metric("Realised tracking error", f"{realised_te:.2%}")
        c1, c2 = st.columns([3, 2])
        with c1:
            a = ar.drop(index="Cash", errors="ignore").sort_values("total contribution")
            fig = go.Figure()
            fig.add_bar(y=a.index, x=a["allocation contribution"], name="allocation", orientation="h", marker_color=BLUE)
            fig.add_bar(y=a.index, x=a["selection contribution"], name="selection", orientation="h", marker_color=ORANGE)
            fig.update_layout(barmode="relative", height=460, margin=dict(t=30, b=10), xaxis_tickformat=".2%", legend=LEG,
                              title=f"Contributions to ex ante tracking error ({ar.attrs['date']} holdings)")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.dataframe(text(ar[["w_port", "w_bench", "active weight", "total contribution", "% of TE²"]],
                              {"w_port": "{:.1%}", "w_bench": "{:.1%}", "active weight": "{:+.1%}", "total contribution": "{:.3%}", "% of TE²": "{:.1%}"}),
                         width="stretch", height=h(ar))
            st.caption("Euler decomposition: contributions add up to the tracking error. A sector can contribute negatively "
                       "when its active weight hedges the other bets.")
        if has_factors:
            st.subheader("Active risk by factor")
            frk = PF.factor_risk(returns, fr_, RG.MODELS["Fama-French 5 + momentum"], True, PPY)
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = go.Figure(go.Bar(x=frk.index, y=frk.values, marker_color=[GREY if i == "specific" else BLUE for i in frk.index]))
                fig.update_layout(height=340, margin=dict(t=30, b=10), yaxis_tickformat=".2%",
                                  title=f"Contribution to active risk {frk.attrs['vol']:.2%} (FF5 + momentum on the active return)")
                st.plotly_chart(fig, width="stretch")
            with c2:
                sysv = frk.drop("specific").sum()
                st.metric("Systematic (factor) share of active variance", f"{sysv / frk.attrs['vol']:.0%}")
                st.metric("Specific share", f"{frk['specific'] / frk.attrs['vol']:.0%}")
                st.caption("A high factor share means the tracking error comes from style bets (size, value, quality, momentum) "
                           "rather than stock picking: those bets can be replicated cheaply with factor products.")
        st.subheader("Total risk: value at risk (1 month)")
        vc, vd = PF.var_contributions(sectors, fr_, sector_ret, returns["rf"], q)
        k = st.columns(4)
        k[0].metric(f"Parametric VaR {q:.1%}", f"{vd['parametric VaR']:.2%}")
        k[1].metric(f"Historical VaR {q:.1%}", f"{vd['historical VaR']:.2%}")
        k[2].metric(f"Historical ES {q:.1%}", f"{vd['historical ES']:.2%}")
        k[3].metric("Monthly volatility", f"{vd['vol']:.2%}")
        v = vc.sort_values("component VaR")
        fig = go.Figure(go.Bar(y=v.index, x=v["component VaR"], orientation="h", marker_color=PURPLE,
                               text=[f"{x:.0%}" for x in v["% of VaR"]], textposition="outside"))
        fig.update_layout(height=420, margin=dict(t=30, b=10), xaxis_tickformat=".2%", title="Component VaR by sector (parametric, Euler)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Historical VaR is computed on the history of the current portfolio (current weights applied to past sector "
                   "returns of the fund).")

# ---------------------------------------------------------------- data & import
with tabs[8]:
    st.markdown("**Returns file** (mandatory): one row per month — `date, rf, benchmark, MKT_RF, [SMB, HML, RMW, CMA, MOM], <fund 1>, <fund 2>, …` "
                "as monthly simple returns in decimals. Factor columns follow the Kenneth French data library convention (MKT_RF = market minus risk-free).")
    st.markdown("**Sector file** (optional, for attribution, frontier and risk contributions): `date, fund, sector, w_port, w_bench, r_port, r_bench` — "
                "beginning-of-month weights, monthly returns in decimals; weights sum to 1 per fund and date (use a `Cash` sector with w_bench = 0).")
    c1, c2 = st.columns(2)
    c1.download_button("Returns template (CSV)", IO.to_csv(returns_all.reset_index()), "returns_template.csv", "text/csv")
    if sectors_all is not None:
        c2.download_button("Sector template (CSV)", IO.to_csv(sectors_all), "sectors_template.csv", "text/csv")
    f1 = st.file_uploader("Returns CSV", type="csv")
    f2 = st.file_uploader("Sector CSV (optional)", type="csv")
    if f1 is not None and st.button("Load"):
        try:
            r_, s_ = IO.load(f1.getvalue(), f2.getvalue() if f2 is not None else None)
            st.session_state["imported"] = (r_, s_)
            st.session_state["imported_name"] = f1.name + (f" + {f2.name}" if f2 is not None else "")
            st.rerun()
        except Exception as err:  # noqa: BLE001
            st.error(f"Import failed: {err}")
    st.subheader("Returns")
    st.dataframe(returns_all, width="stretch", height=320)
    if sectors_all is None:
        st.caption("No sector file: attribution, frontier and risk contributions are disabled.")
    else:
        st.subheader("Sector weights and returns (first fund, last month)")
        last = sectors_all[(sectors_all["date"] == sectors_all["date"].max()) & (sectors_all["fund"] == sectors_all["fund"].iloc[0])]
        st.dataframe(last, width="stretch", hide_index=True)
    st.caption("Synthetic data: Student-t factor returns with a crisis episode and sample premia set to their design values; "
               "11 sector indices driven by the factors; four funds with planted characteristics — quality/growth style "
               "tilt, value/small-cap tilt with genuine stock selection, market timing through cash, and a closet indexer.")

# ---------------------------------------------------------------- theory
with tabs[9]:
    st.markdown(r"""
### 1. From mean-variance to the CAPM
An investor holding risky assets with expected returns $\mu$ and covariance $\Sigma$ solves
$\min_w \tfrac12 w^\top\Sigma w$ subject to $w^\top\mu = m$, $w^\top\mathbf 1 = 1$. With
$a=\mathbf 1^\top\Sigma^{-1}\mathbf 1$, $b=\mathbf 1^\top\Sigma^{-1}\mu$, $c=\mu^\top\Sigma^{-1}\mu$ the frontier is the hyperbola
$$\sigma^2(m)=\frac{a m^2-2bm+c}{ac-b^2}.$$
Adding a risk-free asset $r_f$, every investor holds a mix of $r_f$ and the **tangency portfolio**
$w_T=\Sigma^{-1}(\mu-r_f\mathbf 1)/\mathbf 1^\top\Sigma^{-1}(\mu-r_f\mathbf 1)$ (two-fund separation), along the capital market line
$E[r]=r_f+\frac{E[r_T]-r_f}{\sigma_T}\sigma$.

If all investors share the same beliefs and horizon, the tangency portfolio must be the **market portfolio** $M$ (markets clear).
The first-order condition of the tangency problem, $\mu - r_f\mathbf 1 = \lambda\,\Sigma w_M$, read asset by asset and
normalised by the market portfolio itself gives the **CAPM** (Sharpe 1964, Lintner 1965, Mossin 1966):
$$E[r_i]-r_f=\beta_i\,(E[r_M]-r_f),\qquad \beta_i=\frac{\operatorname{Cov}(r_i,r_M)}{\operatorname{Var}(r_M)}.$$
Only systematic risk is priced; idiosyncratic risk can be diversified away. The **security market line** plots $E[r_i]$ against $\beta_i$.

### 2. Testing the CAPM and measuring skill
Time-series regression $r_{i,t}-r_{f,t}=\alpha_i+\beta_i(r_{M,t}-r_{f,t})+\varepsilon_{i,t}$: the CAPM implies $\alpha_i=0$.
**Jensen's alpha** (1968) is the estimated $\alpha$; its significance is judged with HAC (Newey-West) standard errors.
Roll's critique: the true market portfolio is unobservable, so a test uses a proxy (here, the fund's benchmark) and
alpha is always relative to that proxy.

**Market timing.** Treynor-Mazuy (1966): $r-r_f=\alpha+\beta x+\gamma x^2$; Henriksson-Merton (1981):
$r-r_f=\alpha+\beta x+\gamma\max(x,0)$, an option-like payoff. $\gamma>0$ indicates timing ability.

### 3. Multifactor models
Empirical failures of the CAPM (size, value, momentum, profitability and investment effects) led to
Fama-French (1993) three factors, Carhart (1997) momentum, and Fama-French (2015) five factors:
$$r-r_f=\alpha+\beta_M\,\text{MKT}+s\,\text{SMB}+h\,\text{HML}+r\,\text{RMW}+c\,\text{CMA}\,(+m\,\text{MOM})+\varepsilon.$$
The alpha of a multifactor regression is the return not explained by cheap, replicable style exposures.
Factor attribution: $\overline{r-r_f}=\hat\alpha+\sum_k\hat\beta_k\,\bar f_k$ exactly (OLS residuals have zero mean).

### 4. Risk-adjusted ratios
Sharpe (1966) uses total risk and suits an investor holding only this fund; Treynor (1965) uses beta and suits a
well-diversified investor; the information ratio uses active risk and suits the choice of an active manager within an
asset class. The appraisal ratio $\alpha/\sigma_\varepsilon$ is the Treynor-Black (1973) criterion for sizing an active bet.
M² (Modigliani 1997) rescales the fund to the benchmark's volatility. Sortino and Omega use only the downside
(or the whole distribution around a threshold); the modified Sharpe uses a Cornish-Fisher VaR that accounts for skewness
and kurtosis. The Calmar ratio uses the maximum drawdown, a path-dependent measure.

**Limits.** All are sample estimates: a Sharpe ratio of 0.5 measured over 10 years has a standard error of about 0.3.
Ratios built on volatility penalise upside variance; ratios built on beta depend on the benchmark choice; ratios can be
gamed by option-like strategies (selling tail risk raises the Sharpe until the tail materialises).

### 5. Inference
For i.i.d. returns with skewness $\gamma_3$ and kurtosis $\gamma_4$ (Mertens 2002):
$\operatorname{Var}(\widehat{SR})\approx\frac1T\big(1+\tfrac12 SR^2-\gamma_3 SR+\tfrac{\gamma_4-3}{4}SR^2\big)$.
Probabilistic Sharpe ratio: $\text{PSR}(SR^*)=\Phi\!\Big(\frac{(\widehat{SR}-SR^*)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^2}}\Big)$.
After $N$ trials the best Sharpe is inflated; the deflated Sharpe ratio uses
$SR_0=\sqrt{V[\widehat{SR}_n]}\big((1-\gamma)\Phi^{-1}(1-\tfrac1N)+\gamma\Phi^{-1}(1-\tfrac1{Ne})\big)$ as the threshold
(Bailey and López de Prado 2014). Equality of two Sharpe ratios: Jobson-Korkie (1981) with Memmel's (2003) correction,
or a block bootstrap that is robust to autocorrelation and heavy tails (Ledoit and Wolf 2008).

### 6. Brinson attribution
Brinson, Hood and Beebower (1986) and Brinson-Fachler (1985) split the active return per segment $i$:
allocation $(w_{p,i}-w_{b,i})(r_{b,i}-R_b)$, selection $w_{b,i}(r_{p,i}-r_{b,i})$, interaction $(w_{p,i}-w_{b,i})(r_{p,i}-r_{b,i})$.
Single-period effects are arithmetic, but returns compound; linking makes the effects add up to
$\prod(1+R_{p,t})-\prod(1+R_{b,t})$. Carino (1999) scales each period with logarithmic coefficients; Frongello (2002)
reinvests past effects at the portfolio return and charges them the benchmark return.

### 7. Risk contributions
For a homogeneous risk measure $\rho(w)$, Euler's theorem gives $\rho(w)=\sum_i w_i\,\partial\rho/\partial w_i$.
For the tracking error $\sqrt{a^\top\Sigma a}$ the contribution of bet $i$ is $a_i(\Sigma a)_i/\text{TE}$; for the normal VaR,
$w_i(-\mu_i+z_q(\Sigma w)_i/\sigma_p)$. Contributions are additive and identify where the risk budget is spent.

### References
Sharpe (1964), *Journal of Finance*; Jensen (1968), *Journal of Finance*; Treynor & Mazuy (1966), *HBR*; Henriksson & Merton (1981),
*Journal of Business*; Brinson & Fachler (1985), *JPM*; Brinson, Hood & Beebower (1986), *FAJ*; Fama & French (1993, 2015), *JFE*;
Carhart (1997), *Journal of Finance*; Sortino & Price (1994), *Journal of Investing*; Modigliani & Modigliani (1997), *JPM*;
Carino (1999), *JPM*; Frongello (2002), *Journal of Performance Measurement*; Lo (2002), *FAJ*; Memmel (2003), *Finance Letters*;
Ledoit & Wolf (2008), *Journal of Empirical Finance*; Bailey & López de Prado (2012, 2014), *Journal of Risk*, *JPM*;
Newey & West (1987), *Econometrica*.
""")
