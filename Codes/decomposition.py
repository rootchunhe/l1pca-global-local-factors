import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from constants import (
    data_path, result_path, plot_path, ret_path,
    PCA_GLOBAL_FACTOR_FOLD, RG_SELECT_FOLD, RG_SELECT_SCALE,
    PORT_VWMKT, PORT_BFP_D1, PORT_BFP_D10, PORT_RG0, PORT_RG, PORT_MKTG, PORT_LNF,
)
from analysis_utils import (
    _next_ym, _get_prev_news_init, _ann_stats, _ann_stats_daily,
    _fmt_pct, _fmt_num, _fmt_num_plain, _fmt_triplet, _esc_tex,
    _fmt_pct_tex, _fmt_num_tex,
    _sig_stars_latex, SR_diff_test_indep,
    get_rBetas, get_robust_rBetas, get_port_monthly_ret, get_port_daily_ret,
)
from spanning import prepare_insample_spanning_df, compare_capm_spanning




def compute_lnf_exposure_proxy(Rg_hf_ret, hf_ym, shrink_coef=RG_SELECT_SCALE):
    """Per-stock LNF-exposure proxy: |γ̂(stock, L)|,
    where L = vwMKT30 − κ·φ·Rg_hf_ret,
    φ = robust projection of vwMKT30 on Rg_hf_ret, κ = shrink_coef ∈ (0, 1].
    κ < 1 shrinks because Rg_hf_ret is a noisy initial proxy for g.
    Returns DataFrame[permno, lnf_exposure_proxy]."""
    mkt = hf_ym["vwMKT30"]
    phi_hat = get_robust_rBetas(Rg_hf_ret, mkt)
    residual = mkt - shrink_coef * phi_hat * Rg_hf_ret
    dSTK = hf_ym["fRet30"].values
    present_permnos = [int(c) for c in hf_ym["fRet30"].columns]
    proxy = np.abs(get_rBetas(residual, dSTK))
    return (pd.DataFrame({"permno": present_permnos, "lnf_exposure_proxy": proxy})
              .dropna(subset=["lnf_exposure_proxy"]))



def decompose_market_returns(lnf_exposure_proxy, hf_ym,
                             update_lnf_exposure_proxy=False):
    """Decompose vwMKT into MKTg + LNF using a per-stock LNF-exposure proxy.
      1. Rg = VW bottom 1/RG_SELECT_FOLD of the universe by proxy.
      2. φ = robust projection of vwMKT30 on Rg (no shrinkage; Rg fixed).
      3. MKTg = φ·Rg, LNF = vwMKT − MKTg.
      4. update_lnf_exposure_proxy=True → replace saved proxy with |γ̂(stock, LNF)|.
         Default False: skipping the second-pass leaves the κ-shrunk proxy in place.
    Returns (Rg_wt, phi, lnf_exposure_proxy_out)."""
    n_universe = lnf_exposure_proxy.shape[0]
    if n_universe < 1000:
        print(f"WARNING: lnf_exposure_proxy universe has only {n_universe} stocks (< 1000)")
    nstks = n_universe // RG_SELECT_FOLD
    Rg_permnos = lnf_exposure_proxy.nsmallest(nstks, "lnf_exposure_proxy")["permno"].tolist()

    # vwMKT_wt from get_fRet30_all; renormalize within Rg subset
    wt = hf_ym["vwMKT_wt"].loc[Rg_permnos]
    wt = wt / wt.sum()
    cols = [str(p) for p in Rg_permnos]
    Rg_hf_ret = hf_ym["fRet30"][cols].values @ wt.values
    Rg_wt = pd.DataFrame({"permno": Rg_permnos, "Rg_wt": wt.values})

    phi = get_robust_rBetas(Rg_hf_ret, hf_ym["vwMKT30"])
    LNF_hf_ret = hf_ym["vwMKT30"] - phi * Rg_hf_ret

    if update_lnf_exposure_proxy:
        # Direct |β(stock, LNF)| on the post-decomposition residual; no further shrinkage.
        dSTK = hf_ym["fRet30"].values
        present_permnos = [int(c) for c in hf_ym["fRet30"].columns]
        lnf_exposure = np.abs(get_rBetas(LNF_hf_ret, dSTK))
        lnf_exposure_proxy_out = (
            pd.DataFrame({"permno": present_permnos, "lnf_exposure_proxy": lnf_exposure})
              .dropna(subset=["lnf_exposure_proxy"])
        )
    else:
        lnf_exposure_proxy_out = lnf_exposure_proxy

    return Rg_wt, phi, lnf_exposure_proxy_out



def align_oos_to_realization(port_ret_all):
    """Re-label `*_oos` columns from formation-month to realization-month.
    Builders write row(ym).ret_ex_oos = ym weights × (ym+1) returns;
    shifting down by 1 within each port puts it on row(ym+1) where it was realized.
    Assumes ym is contiguous within each port (true when builders share hf_data.keys())."""
    df = port_ret_all.sort_values(["port", "ym"]).reset_index(drop=True).copy()
    oos_cols = [c for c in df.columns if "_oos" in c]
    df[oos_cols] = df.groupby("port")[oos_cols].shift(1)
    return df



def build_benchmark_portfolios(hf_data, data):
    """Build per-month benchmark portfolios:
    vwMKT, BetaFP_D1 (low-β decile), BetaFP_D10 (high-β decile).
    Independent of Rg construction. All value-weighted.
    Shares the same monthly stock universe as Rg/MKTg/LNF
    (HF ∩ BetaFP non-NaN, set in get_fRet30_all).

    Returns (bench_wt, bench_ret):
      bench_wt : DataFrame[ym, permno, vwMKT_wt, BetaFP_D1_wt, BetaFP_D10_wt]
      bench_ret: long-format [ym, port, ret_ex_*, intra_ex_*, over_ex_*],
                 port ∈ {vwMKT, BetaFP_D1_vw, BetaFP_D10_vw}, pre-alignment.
    """
    wt_rows, ret_rows = [], []
    for ym in sorted(hf_data.keys()):
        hf_ym = hf_data[ym]
        ym_oos = _next_ym(ym)
        base = hf_ym["vwMKT_wt"].rename_axis("permno").rename("vwMKT_wt").reset_index()

        def _ret_pair(wt):
            return (get_port_monthly_ret(wt, ym, data, suffix="_is")
                    | get_port_monthly_ret(wt, ym_oos, data, suffix="_oos"))

        # vwMKT
        wt_vw = base.set_index("permno")["vwMKT_wt"]
        ret_rows.append({"ym": ym, "port": PORT_VWMKT} | _ret_pair(wt_vw))
        out_wt = base.copy()

        # BetaFP decile portfolios
        if "BetaFP" in data["crsp_taq"].columns:
            bfp = data["crsp_taq"].loc[data["crsp_taq"]["ym"] == ym, ["permno", "BetaFP"]].copy()
            bfp = bfp.dropna(subset=["permno", "BetaFP"])
            bfp = bfp[bfp["permno"].isin(base["permno"])]
            bfp["pct"] = bfp["BetaFP"].rank(pct=True, method="first")
            vw_map = base.set_index("permno")["vwMKT_wt"]

            perm_d1 = bfp.loc[bfp["pct"] <= 0.1, "permno"].tolist()
            wt_d1 = vw_map.reindex(perm_d1)
            wt_d1 = wt_d1 / float(wt_d1.sum())
            out_wt = out_wt.merge(
                wt_d1.rename("BetaFP_D1_wt").reset_index(),
                on="permno", how="left",
            )
            ret_rows.append({"ym": ym, "port": PORT_BFP_D1} | _ret_pair(wt_d1))

            perm_d10 = bfp.loc[bfp["pct"] >= 0.9, "permno"].tolist()
            wt_d10 = vw_map.reindex(perm_d10)
            wt_d10 = wt_d10 / float(wt_d10.sum())
            out_wt = out_wt.merge(
                wt_d10.rename("BetaFP_D10_wt").reset_index(),
                on="permno", how="left",
            )
            ret_rows.append({"ym": ym, "port": PORT_BFP_D10} | _ret_pair(wt_d10))

        out_wt.insert(0, "ym", ym)
        wt_rows.append(out_wt)

    bench_wt = pd.concat(wt_rows, ignore_index=True).fillna(0)
    bench_ret = pd.DataFrame(ret_rows)
    return bench_wt, bench_ret



def build_rg_factors(hf_data, pca_df, data,
                     shrink_coef=RG_SELECT_SCALE,
                     update_lnf_exposure_proxy=False,
                     verbose=True):
    """Build Rg / MKTg / LNF factors. Per month:
      1. news_proxy (monthly PCA of S&P 500; pca.run_monthly_analysis) →
         Rg0 = EW bottom 1/PCA_GLOBAL_FACTOR_FOLD of (S&P 500 ∩ HF universe) by news_proxy.
         Falls back to previous month's news_proxy if missing.
      2. lnf_exposure_proxy_i = |γ̂(stock_i, L)| over the HF universe,
         where L = vwMKT − κ·φ·Rg0 (κ = shrink_coef, φ = projection of vwMKT on Rg0).
         κ < 1 shrinks because Rg0 is a noisy initial proxy for g.
      3. Rg = VW bottom 1/RG_SELECT_FOLD of HF universe by proxy.
         φ = robust projection of vwMKT on Rg (HF, for SNR); φ is the Rg portfolio weight.
         MKTg = φ·Rg, LNF = vwMKT − MKTg.
      4. Monthly excess returns for {Rg0, Rg, MKTg, LNF}.

    `update_lnf_exposure_proxy` default False is more robust; toggle True for the alternative.

    Returns (rg_wt, rg_ret, lnf_exposure_proxies, phi_by_ym):
      rg_wt:                [ym, permno, Rg_wt_0, Rg_wt, mktg_wt]
      rg_ret:               long-format [ym, port, ret_ex_*, intra_ex_*, over_ex_*],
                            port ∈ {Rg0, Rg, MKTg, LNF}, pre-alignment.
      lnf_exposure_proxies: [ym, permno, lnf_exposure_proxy] from step 2.
      phi_by_ym:            dict[ym → φ_t]
    """
    wt_rows, ret_rows, proxy_rows = [], [], []
    phi_by_ym = {}

    for ym in sorted(hf_data.keys()):
        hf_ym = hf_data[ym]
        ym_oos = _next_ym(ym)
        news_ym = pca_df[pca_df["ym"] == ym][["permno", "news_proxy"]].dropna()
        if news_ym.empty:
            init_news = _get_prev_news_init(ym)
        else:
            init_news = news_ym.copy()
            init_news["permno"] = init_news["permno"].astype(int)
            init_news = init_news.drop_duplicates(["permno"], keep="last")

        def _ret_pair(wt):
            return (get_port_monthly_ret(wt, ym, data, suffix="_is")
                    | get_port_monthly_ret(wt, ym_oos, data, suffix="_oos"))

        # ── Rg0: equal-weighted bottom news-proxy quintile of S&P 500 (inline) ──
        nstks_Rg_0 = max(len(init_news) // PCA_GLOBAL_FACTOR_FOLD, 1)
        Rg0_permnos = list(init_news.nsmallest(nstks_Rg_0, "news_proxy")["permno"])
        fRet_ym = hf_ym["fRet30"]
        Rg0_cols = [str(p) for p in Rg0_permnos if str(p) in fRet_ym.columns]
        if not Rg0_cols:
            raise ValueError(f"{ym}: no valid Rg0 permnos in fRet30 panel.")
        w_eq = np.full(len(Rg0_cols), 1.0 / len(Rg0_cols))
        Rg0_hf_ret = (fRet_ym[Rg0_cols].values * w_eq).sum(axis=1)
        Rg0_wt = pd.DataFrame({"permno": [int(c) for c in Rg0_cols], "Rg_wt_0": w_eq})

        # ── shrunk LNF-exposure proxy from Rg0 ──
        proxy = compute_lnf_exposure_proxy(Rg0_hf_ret, hf_ym, shrink_coef=shrink_coef)

        # ── decompose vwMKT into Rg + MKTg + LNF ──
        Rg_wt_df, phi, proxy = decompose_market_returns(
            proxy, hf_ym,
            update_lnf_exposure_proxy=update_lnf_exposure_proxy,
        )
        phi_by_ym[ym] = phi

        # ── monthly excess returns ──
        r_rg0   = _ret_pair(Rg0_wt.set_index("permno")["Rg_wt_0"])
        r_rg    = _ret_pair(Rg_wt_df.set_index("permno")["Rg_wt"])
        r_vwmkt = _ret_pair(hf_ym["vwMKT_wt"])
        r_mktg = {k: phi * v for k, v in r_rg.items()}  # φ = Rg portfolio weight
        r_lnf  = {k: r_vwmkt[k] - r_mktg[k] for k in r_vwmkt}

        ret_rows.append({"ym": ym, "port": PORT_RG0}  | r_rg0)
        ret_rows.append({"ym": ym, "port": PORT_RG}   | r_rg)
        ret_rows.append({"ym": ym, "port": PORT_MKTG} | r_mktg)
        ret_rows.append({"ym": ym, "port": PORT_LNF}  | r_lnf)

        # ── weights ──
        out_wt = Rg0_wt.merge(Rg_wt_df, on="permno", how="outer")
        out_wt["mktg_wt"] = phi * out_wt["Rg_wt"].fillna(0)
        out_wt.insert(0, "ym", ym)
        wt_rows.append(out_wt)

        # ── exposure proxy ──
        proxy_with_ym = proxy.copy()
        proxy_with_ym.insert(0, "ym", ym)
        proxy_rows.append(proxy_with_ym)

        if verbose:
            # overwrite progress line
            msg = (f"  {ym}  vwMKT-RF {r_vwmkt['ret_ex_is']*100:+.2f}%, "
                   f"n_Rg={Rg_wt_df.shape[0]}, "
                   f"Rg-RF is={r_rg['ret_ex_is']*100:+.2f}% oos={r_rg['ret_ex_oos']*100:+.2f}%")
            print("\r" + " " * 200, end="", flush=True)
            print("\r" + msg, end="", flush=True)
    if verbose:
        print()  # finish progress line

    rg_wt = pd.concat(wt_rows, ignore_index=True).fillna(0)
    rg_ret = pd.DataFrame(ret_rows)
    lnf_exposure_proxies = pd.concat(proxy_rows, ignore_index=True)
    return rg_wt, rg_ret, lnf_exposure_proxies, phi_by_ym



def _coerce_int_keys(df, cols=("ym", "permno")):
    """Coerce key columns to int, dropping rows where any key is non-numeric."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(cols))
    for c in cols:
        df[c] = df[c].astype(int)
    return df


def rg_stocks_summary(mktg_port_all):
    char_df = pd.read_csv(Path(data_path) / "monthly_lagged_betas.csv", low_memory=False)
    char_df = char_df.rename(columns={"yyyymm": "ym"})

    cols = ["ym", "permno", "size_wrds", "Beta60m", "BetaFP", "beta_252d"]
    ch = _coerce_int_keys(char_df[cols].copy())
    ch["Size_exp"] = np.exp(pd.to_numeric(ch["size_wrds"], errors="coerce"))
    ch["size_pct"] = ch.groupby("ym")["Size_exp"].rank(pct=True)
    ch["Beta60m"] = pd.to_numeric(ch["Beta60m"], errors="coerce")
    ch["BetaFP"] = pd.to_numeric(ch["BetaFP"], errors="coerce")
    ch["Beta252d"] = pd.to_numeric(ch["beta_252d"], errors="coerce")
    beta_cols = ["Beta60m", "BetaFP", "Beta252d"]

    port = _coerce_int_keys(mktg_port_all.copy())

    def summarize(weight_col):
        if weight_col not in port.columns:
            raise KeyError(f"{weight_col} not found in mktg_port_all")
        if "BetaFP_D1_wt" not in port.columns:
            raise KeyError("BetaFP_D1_wt not found in mktg_port_all")
        hold = port.loc[port[weight_col].fillna(0) > 0, ["ym", "permno", weight_col]].copy()
        hold[weight_col] = pd.to_numeric(hold[weight_col], errors="coerce")
        merged = hold.merge(
            ch[["ym", "permno", "Size_exp", "size_pct"] + beta_cols],
            on=["ym", "permno"],
            how="left",
        )
        out = merged.groupby("ym", as_index=False).agg(
            n_stocks=("permno", "nunique"),
            avg_size=("Size_exp", "mean"),
            avg_size_quantile=("size_pct", "mean"),
        )
        for b in beta_cols:
            tmp = merged.loc[merged[b].notna(), ["ym", weight_col, b]].copy()
            vw = (tmp[weight_col] * tmp[b]).groupby(tmp["ym"]).sum() / tmp[weight_col].groupby(tmp["ym"]).sum()
            out = out.merge(vw.rename(f"vw_{b}").reset_index(), on="ym", how="left")

        # Simple overlap ratio vs BetaFP_D1_wt base: |A∩B| / |A|, where A={weight_col>0}, B={BetaFP_D1_wt>0}.
        base = port.loc[port["BetaFP_D1_wt"].fillna(0) > 0, ["ym", "permno"]].drop_duplicates()
        a = hold[["ym", "permno"]].drop_duplicates()
        inter = a.merge(base, on=["ym", "permno"], how="inner")
        a_n = a.groupby("ym")["permno"].nunique().rename("a_n")
        inter_n = inter.groupby("ym")["permno"].nunique().rename("inter_n")
        ov = pd.concat([a_n, inter_n], axis=1).fillna(0)
        ov["overlap_vs_bfp_d1"] = np.where(ov["a_n"] > 0, ov["inter_n"] / ov["a_n"], np.nan)
        out = out.merge(ov["overlap_vs_bfp_d1"].reset_index(), on="ym", how="left")
        return out.sort_values("ym").reset_index(drop=True)

    return summarize("Rg_wt")



def summarize_and_plot_rg_lnf(mktg_ret_all):
    ret_cols = [c for c in mktg_ret_all.columns if c.startswith(("ret_ex", "intra_ex", "over_ex"))]
    vwmkt = mktg_ret_all[mktg_ret_all["port"] == PORT_VWMKT].set_index("ym")[ret_cols]
    highbeta = mktg_ret_all[mktg_ret_all["port"] == PORT_BFP_D10].set_index("ym")[ret_cols]
    # MKTg / LNF rows are already in the panel (built by build_rg_factors).
    ret_ext = mktg_ret_all.copy()

    summary_ports = [PORT_VWMKT, PORT_RG0, PORT_RG, PORT_MKTG, PORT_LNF, PORT_BFP_D1, PORT_BFP_D10]
    rets = [("ret_ex", "Full return"), ("intra_ex", "Intraday return"), ("over_ex", "Overnight return")]

    sub = ret_ext[ret_ext["port"].isin(summary_ports)].copy()
    sub["ym"] = sub["ym"].astype(int)

    stats_map = {}
    for port in summary_ports:
        row = sub[sub["port"] == port].sort_values("ym")
        stats_map[port] = {}
        for rtype, _ in rets:
            r_is = row[f"{rtype}_is"].dropna()
            r_oos = row[f"{rtype}_oos"].dropna()
            st_is = _ann_stats(r_is)
            ann_is = st_is["AnnRet(%)"]
            vol_is = st_is["AnnVol(%)"]
            sr_is = st_is["SR"]
            st_oos = _ann_stats(r_oos)
            ann_oos = st_oos["AnnRet(%)"]
            vol_oos = st_oos["AnnVol(%)"]
            sr_oos = st_oos["SR"]
            stats_map[port][rtype] = {
                "is": (ann_is, vol_is, sr_is),
                "oos": (ann_oos, vol_oos, sr_oos),
            }

    def _print_section(title, rows, sample_key):
        print(f"\n{title}")
        print(" " * 26 + "Full" + " " * 21 + "Intraday" + " " * 19 + "Overnight")
        print(" " * 20 + "Ret     Vol    SR" + " " * 9 + "Ret     Vol    SR" + " " * 9 + "Ret     Vol    SR")
        print("-" * 94)
        for port, label in rows:
            f = _fmt_triplet(stats_map[port]["ret_ex"][sample_key])
            i = _fmt_triplet(stats_map[port]["intra_ex"][sample_key])
            o = _fmt_triplet(stats_map[port]["over_ex"][sample_key])
            print(f"{label:18}  {f}  {i}  {o}")

    # Traditional factors are formed using lagged characteristics, so the
    # "_is" return series is already economically out-of-sample.
    _print_section(
        "Traditional Factors",
        [(PORT_VWMKT, "Mkt-RF"), (PORT_BFP_D1, "LowBeta-RF"), (PORT_BFP_D10, "HighBeta-RF")],
        "is",
    )
    _print_section(
        "In-sample Decomposition",
        [(PORT_RG0, "Rg0-RF (from S&P 500)"),
         (PORT_RG, "Rg-RF"),
         (PORT_MKTG, "MKTg-RF"), (PORT_LNF, PORT_LNF)],
        "is",
    )
    _print_section(
        "Decomposition Out-of-sample",
        [(PORT_RG0, "Rg0-RF (from S&P 500)"),
         (PORT_RG, "Rg-RF"),
         (PORT_MKTG, "MKTg-RF"), (PORT_LNF, PORT_LNF)],
        "oos",
    )

    # Cum-return plots, IS & OOS panels. Benchmarks pinned to _is (lagged inputs, no look-ahead).
    colors_plot = {
        PORT_VWMKT: "#111111",
        PORT_RG: "#1f77b4",
        PORT_LNF: "#2ca02c",
        "BetaFP_D10_vm": "#d62728",
        PORT_BFP_D1: "#9467bd",
    }
    highbeta_vm = highbeta.copy()
    for col in ret_cols:
        hb_vol = pd.to_numeric(highbeta[col], errors="coerce").std()
        mkt_vol = pd.to_numeric(vwmkt[col], errors="coerce").std()
        scale = mkt_vol / hb_vol if pd.notna(hb_vol) and hb_vol > 0 else np.nan
        highbeta_vm[col] = highbeta[col] * scale
    highbeta_vm = highbeta_vm.reset_index().assign(port="BetaFP_D10_vm")
    sub_plot = pd.concat([sub, highbeta_vm], ignore_index=True)
    series_specs = [
        (PORT_VWMKT, "MKT-RF", colors_plot[PORT_VWMKT]),
        (PORT_RG, "Rg-RF", colors_plot[PORT_RG]),
        (PORT_LNF, "LNF", colors_plot[PORT_LNF]),
        ("BetaFP_D10_vm", "HighBeta-RF (vol-matched)", colors_plot["BetaFP_D10_vm"]),
        (PORT_BFP_D1, "LowBeta-RF", colors_plot[PORT_BFP_D1]),
    ]
    # Only Rg / LNF switch between IS and OOS; the rest stay on _is.
    oos_ports = {PORT_RG, PORT_LNF}

    def _plot_panel(sample, save_name):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (rtype, rname) in zip(axes, rets):
            for port, label, c in series_specs:
                s = sub_plot[sub_plot["port"] == port].sort_values("ym")
                col = f"{rtype}_{sample}" if port in oos_ports else f"{rtype}_is"
                dt = pd.to_datetime(s["ym"].astype(str), format="%Y%m")
                r = pd.to_numeric(s[col], errors="coerce")
                if r.isna().all():
                    continue
                r = r.fillna(0.0)
                ax.plot(dt.values, (1 + r).cumprod().values, color=c, lw=1.0, label=label)
            ax.set_title(rname)
            ax.set_xlabel("")
            ax.set_ylabel("cumulative return")
            ax.xaxis.set_major_locator(mdates.YearLocator(4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(Path(plot_path) / save_name, bbox_inches="tight", dpi=300)
        plt.show()

    _plot_panel("is",  "is_decomp.pdf")
    _plot_panel("oos", "oos_decomp.pdf")
    return ret_ext, sub



def build_rg_lnf_perf_stats(
    sub,
    ports=None,
    labels=None,
    rets=None,
    include_mktg=True,
):
    if ports is None:
        ports = [PORT_VWMKT, PORT_RG0, PORT_RG, PORT_LNF, PORT_BFP_D1, PORT_BFP_D10]
        if include_mktg:
            ports = [PORT_VWMKT, PORT_RG0, PORT_RG, PORT_MKTG, PORT_LNF, PORT_BFP_D1, PORT_BFP_D10]
    if labels is None:
        labels = {
            PORT_VWMKT: "Mkt-RF",
            PORT_RG0: "Rg0-RF (from S&P 500)",
            PORT_RG: "Rg-RF",
            PORT_MKTG: "MKTg-RF",
            PORT_LNF: PORT_LNF,
            PORT_BFP_D1: "LowBeta-RF",
            PORT_BFP_D10: "HighBeta-RF",
        }
    if rets is None:
        rets = [("ret_ex", "Full (excess)"), ("intra_ex", "Intraday (excess)"), ("over_ex", "Overnight (excess)")]

    rows = []
    for rtype, rname in rets:
        for port in ports:
            row = sub[sub["port"] == port].sort_values("ym")

            st_is = _ann_stats(row[f"{rtype}_is"])
            st_oos = _ann_stats(row[f"{rtype}_oos"])
            ia, iv, isr = st_is["AnnRet(%)"], st_is["AnnVol(%)"], st_is["SR"]
            oa, ov, osr = st_oos["AnnRet(%)"], st_oos["AnnVol(%)"], st_oos["SR"]
            rows.append({
                "panel": rname,
                "asset": labels[port],
                "IS AnnRet": ia,
                "IS Vol": iv,
                "IS SR": isr,
                "OOS AnnRet": oa,
                "OOS Vol": ov,
                "OOS SR": osr,
            })
    return pd.DataFrame(rows)



def build_conditional_stats(
    ret_ext,
    group_ym,
    save_filename=None,
    ret_col="ret_ex_is",
    latex_filename=None,
):
    """Unconditional + conditional annualized stats. group_ym provides 'Group' labels
    or a single numeric column (median split → High/Low)."""
    portfolios = [PORT_VWMKT, PORT_BFP_D1, PORT_BFP_D10, PORT_RG, PORT_LNF]
    labels = {
        PORT_VWMKT: "vwMKT-RF",
        PORT_BFP_D1: "LowBeta-RF",
        PORT_BFP_D10: "HighBeta-RF",
        PORT_RG: "Rg-RF",
        PORT_LNF: PORT_LNF,
    }

    grp = group_ym.copy()
    grp["ym"] = pd.to_numeric(grp["ym"], errors="coerce")
    grp = grp.dropna(subset=["ym"]).copy()
    grp["ym"] = grp["ym"].astype(int)

    if "Group" not in grp.columns:
        candidates = [c for c in grp.columns if c != "ym"]
        if len(candidates) != 1:
            raise ValueError("group_ym must provide 'Group' or exactly one grouping column besides 'ym'.")
        x = pd.to_numeric(grp[candidates[0]], errors="coerce")
        med = x.median()
        grp["Group"] = np.where(x >= med, "High group", "Low group")

    grp = grp[["ym", "Group"]].dropna(subset=["Group"]).drop_duplicates("ym", keep="last")

    # Group order: low first, then high, else alphabetical. SR diff = SR(low) - SR(high).
    groups = sorted(
        grp["Group"].unique().tolist(),
        key=lambda x: (
            0 if str(x).lower().startswith("low") else 1 if str(x).lower().startswith("high") else 2,
            str(x),
        ),
    )

    sr_diff_by_port = {}  # label -> (sr_diff, sr_diff_t)
    rows = []
    for port in portfolios:
        label = labels.get(port, port)
        s = ret_ext[ret_ext["port"] == port][["ym", ret_col]].copy().rename(columns={ret_col: "ret_ex"})
        s["ym"] = pd.to_numeric(s["ym"], errors="coerce")
        s = s.dropna(subset=["ym"]).copy()
        s["ym"] = s["ym"].astype(int)

        m = s.merge(grp, on="ym", how="left").dropna(subset=["ret_ex"]).copy()
        st_all = _ann_stats(m["ret_ex"])
        rows.append({"Portfolio": label, "Group": "Unconditional", **st_all})

        group_returns = {}
        for g, gg in m.dropna(subset=["Group"]).groupby("Group"):
            st = _ann_stats(gg["ret_ex"])
            rows.append({"Portfolio": label, "Group": g, **st})
            group_returns[g] = gg["ret_ex"].values

        # SR(group[0]) − SR(group[1]) + NW two-independent-sample t-stat (lags=10).
        if len(groups) == 2 and all(g in group_returns for g in groups):
            sr0 = _ann_stats(pd.Series(group_returns[groups[0]]))["SR"]
            sr1 = _ann_stats(pd.Series(group_returns[groups[1]]))["SR"]
            t_stat = SR_diff_test_indep(group_returns[groups[0]], group_returns[groups[1]], lags=10)
            sr_diff_by_port[label] = (sr0 - sr1, float(t_stat) if pd.notna(t_stat) else np.nan)
        else:
            sr_diff_by_port[label] = (np.nan, np.nan)

    stats_df = pd.DataFrame(rows)
    # Attach diff / t-stat to the Unconditional row for each portfolio (CSV-visible).
    stats_df["SR_diff"] = np.nan
    stats_df["SR_diff_t"] = np.nan
    for label, (d, t) in sr_diff_by_port.items():
        mask = (stats_df["Portfolio"] == label) & (stats_df["Group"] == "Unconditional")
        stats_df.loc[mask, "SR_diff"] = d
        stats_df.loc[mask, "SR_diff_t"] = t

    if save_filename is not None:
        stats_df.to_csv(Path(result_path) / save_filename, index=False)

    order = [labels.get(p, p) for p in portfolios]

    # Console print in the same structure as LaTeX table.
    w_port = 16
    w_num = 9
    print("\nConditional Stats (Full Excess Return, annualized)")
    print(f"  Months partitioned into {{{', '.join(map(str, groups))}}}. "
          f"Each row reports that portfolio's annualized stats within the month subsample.")
    show_diff = len(groups) == 2
    diff_lbl = f"SR diff ({groups[0]}-{groups[1]})" if show_diff else ""
    w_diff = 14
    hdr1 = f"{'Portfolio':<{w_port}} | {'Unconditional':<{3 * w_num + 2}}"
    for g in groups:
        hdr1 += f" | {str(g):<{3 * w_num + 2}}"
    if show_diff:
        hdr1 += f" | {diff_lbl:<{w_diff + 1 + w_num}}"
    print(hdr1)
    hdr2 = f"{'':<{w_port}} | {'mean(%)':>{w_num}} {'vol(%)':>{w_num}} {'SR':>{w_num}}"
    for _ in groups:
        hdr2 += f" | {'mean(%)':>{w_num}} {'vol(%)':>{w_num}} {'SR':>{w_num}}"
    if show_diff:
        hdr2 += f" | {'diff':>{w_diff}} {'t-stat':>{w_num}}"
    print(hdr2)
    print("-" * len(hdr2))
    for p in order:
        rr_u = stats_df[(stats_df["Portfolio"] == p) & (stats_df["Group"] == "Unconditional")]
        if rr_u.empty:
            continue
        ru = rr_u.iloc[0]
        line = (
            f"{p:<{w_port}} | "
            f"{ru['AnnRet(%)'] if pd.notna(ru['AnnRet(%)']) else np.nan:>{w_num}.2f} "
            f"{ru['AnnVol(%)'] if pd.notna(ru['AnnVol(%)']) else np.nan:>{w_num}.2f} "
            f"{ru['SR'] if pd.notna(ru['SR']) else np.nan:>{w_num}.2f}"
        )
        for g in groups:
            rr_g = stats_df[(stats_df["Portfolio"] == p) & (stats_df["Group"] == g)]
            if rr_g.empty:
                line += f" | {'':>{w_num}} {'':>{w_num}} {'':>{w_num}}"
            else:
                rg = rr_g.iloc[0]
                line += (
                    f" | "
                    f"{rg['AnnRet(%)'] if pd.notna(rg['AnnRet(%)']) else np.nan:>{w_num}.2f} "
                    f"{rg['AnnVol(%)'] if pd.notna(rg['AnnVol(%)']) else np.nan:>{w_num}.2f} "
                    f"{rg['SR'] if pd.notna(rg['SR']) else np.nan:>{w_num}.2f}"
                )
        if show_diff:
            d, t = sr_diff_by_port.get(p, (np.nan, np.nan))
            diff_str = _fmt_num_plain(d) + _sig_stars_latex(t)
            t_str = _fmt_num_plain(t)
            line += f" | {diff_str:>{w_diff}} {t_str:>{w_num}}"
        print(line)

    # Default LaTeX filename: aligned with save_filename if provided.
    if latex_filename is None:
        if save_filename is not None:
            latex_filename = Path(save_filename).with_suffix(".tex").name
        else:
            latex_filename = "conditional_stats_is.tex"

    show_diff_tex = len(groups) == 2
    n_main_cols = 3 * (1 + len(groups))
    n_extra_cols = 2 if show_diff_tex else 0

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(
        rf"\begin{{tabularx}}{{0.95\textwidth}}{{l *{{{n_main_cols + n_extra_cols}}}{{>{{\centering\arraybackslash}}X}}}}"
    )
    lines.append(r"\toprule")

    hdr_top = r"& \multicolumn{3}{c}{Unconditional}"
    for g in groups:
        hdr_top += rf" & \multicolumn{{3}}{{c}}{{{_esc_tex(g)}}}"
    if show_diff_tex:
        diff_caption = rf"SR diff ({_esc_tex(groups[0])} $-$ {_esc_tex(groups[1])})"
        hdr_top += rf" & \multicolumn{{2}}{{c}}{{{diff_caption}}}"
    hdr_top += r" \\"
    lines.append(hdr_top)
    hdr_bottom = r"Portfolio & mean (\%) & vol (\%) & SR"
    for _ in groups:
        hdr_bottom += r" & mean (\%) & vol (\%) & SR"
    if show_diff_tex:
        hdr_bottom += r" & diff & $t$-stat"
    hdr_bottom += r" \\"
    lines.append(hdr_bottom)
    lines.append(r"\midrule")

    for p in order:
        rr_u = stats_df[(stats_df["Portfolio"] == p) & (stats_df["Group"] == "Unconditional")]
        if rr_u.empty:
            continue
        ru = rr_u.iloc[0]
        row = (
            f"{_esc_tex(p)} & "
            f"{_fmt_pct_tex(ru['AnnRet(%)'])} & "
            f"{_fmt_pct_tex(ru['AnnVol(%)'], signed=False)} & "
            f"{_fmt_num_tex(ru['SR'])}"
        )
        for g in groups:
            rr_g = stats_df[(stats_df["Portfolio"] == p) & (stats_df["Group"] == g)]
            if rr_g.empty:
                row += " &  &  & "
            else:
                rg = rr_g.iloc[0]
                row += (
                    f" & {_fmt_pct_tex(rg['AnnRet(%)'])}"
                    f" & {_fmt_pct_tex(rg['AnnVol(%)'], signed=False)}"
                    f" & {_fmt_num_tex(rg['SR'])}"
                )
        if show_diff_tex:
            d, t = sr_diff_by_port.get(p, (np.nan, np.nan))
            row += f" & {_fmt_num_plain(d)}{_sig_stars_latex(t)} & {_fmt_num_plain(t)}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")

    latex_file = Path(result_path) / latex_filename
    with open(latex_file, "w") as f:
        f.write("\n".join(lines))
    return stats_df



def print_latex_rg_lnf_perf_table(perf_df, latex_filename=None):
    panel_map = {
        "Full": "Full (excess)",
        "Intraday": "Intraday (excess)",
        "Overnight": "Overnight (excess)",
    }
    sections = [
        # Traditional factors are built on lagged characteristics, so the
        # reported "_is" performance is already economically out-of-sample.
        ("Traditional Factors", [("Mkt-RF", "IS"), ("LowBeta-RF", "IS"), ("HighBeta-RF", "IS")]),
        ("In-sample Decomposition",
         [("Rg0-RF (from S&P 500)", "IS"), ("Rg-RF", "IS"), ("MKTg-RF", "IS"), (PORT_LNF, "IS")]),
        ("Decomposition Out-of-sample",
         [("Rg0-RF (from S&P 500)", "OOS"), ("Rg-RF", "OOS"), ("MKTg-RF", "OOS"), (PORT_LNF, "OOS")]),
    ]

    def _lookup(asset, panel, sample):
        row = perf_df[(perf_df["asset"] == asset) & (perf_df["panel"] == panel)]
        if row.empty:
            return np.nan, np.nan, np.nan
        row = row.iloc[0]
        return row[f"{sample} AnnRet"], row[f"{sample} Vol"], row[f"{sample} SR"]

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabularx}{0.95\textwidth}{l *{9}{>{\centering\arraybackslash}X}}")
    lines.append(r"\toprule")
    for sec_idx, (title, rows) in enumerate(sections):
        if sec_idx > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textbf{{{_esc_tex(title)}}}}} \\")
        lines.append(r"& \multicolumn{3}{c}{Full} & \multicolumn{3}{c}{Intraday} & \multicolumn{3}{c}{Overnight} \\")
        lines.append(r"Asset & Ret & Vol & SR & Ret & Vol & SR & Ret & Vol & SR \\")
        lines.append(r"\midrule")
        for asset, sample in rows:
            full = _lookup(asset, panel_map["Full"], sample)
            intra = _lookup(asset, panel_map["Intraday"], sample)
            over = _lookup(asset, panel_map["Overnight"], sample)
            lines.append(
                f"{_esc_tex(asset)} & "
                f"{_fmt_pct_tex(full[0])} & {_fmt_pct_tex(full[1], signed=False)} & {_fmt_num_tex(full[2])} & "
                f"{_fmt_pct_tex(intra[0])} & {_fmt_pct_tex(intra[1], signed=False)} & {_fmt_num_tex(intra[2])} & "
                f"{_fmt_pct_tex(over[0])} & {_fmt_pct_tex(over[1], signed=False)} & {_fmt_num_tex(over[2])} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)
    if latex_filename is not None:
        latex_file = Path(result_path) / latex_filename
        with open(latex_file, "w") as f:
            f.write(tex)



def summarize_daily_rg_lnf_subsample_stats(
    days_df,
    ff6_daily,
    latex_filename=None,
):
    rg_port_wt_all = pd.read_csv(Path(result_path) / "Rg_port_wt_all.csv")
    rg_port_wt_all["ym"] = pd.to_numeric(rg_port_wt_all["ym"], errors="coerce").astype(int)
    rg_port_wt_all["permno"] = pd.to_numeric(rg_port_wt_all["permno"], errors="coerce").astype(int)
    rg_port_wt_all = rg_port_wt_all.drop_duplicates(["ym", "permno"], keep="last")

    ret_daily = pd.read_csv(Path(ret_path) / "ret_daily_data.csv").drop(columns=["SPY"], errors="ignore")
    date_num = pd.to_numeric(ret_daily["date"], errors="coerce").astype(int)

    rf_daily = ff6_daily[["date", "RF"]].copy()
    rf_daily["date"] = pd.to_numeric(rf_daily["date"], errors="coerce").astype(int)
    rf_daily = rf_daily.drop_duplicates("date", keep="last")

    day_flags = days_df.copy()
    day_flags["date"] = pd.to_numeric(day_flags["date"], errors="coerce").astype(int)
    if "macro_day" not in day_flags.columns:
        day_flags["macro_day"] = day_flags[["cpi_day", "ppi_day", "unrate_day"]].max(axis=1).astype(int)
    day_flags["special_day"] = day_flags[["lead_day", "macro_day", "fomc_day"]].sum(axis=1).gt(0).astype(int)
    day_flags = day_flags[["date", "special_day"]].drop_duplicates("date", keep="last")

    ret_meta = (
        pd.DataFrame({"date": date_num})
        .merge(rf_daily, on="date", how="left")
        .merge(day_flags, on="date", how="left")
    )
    ret_meta["ym"] = ret_meta["date"] // 100
    ret_meta["RF"] = pd.to_numeric(ret_meta["RF"], errors="coerce").fillna(0.0)
    ret_meta["special_day"] = ret_meta["special_day"].fillna(0).astype(int)
    ret_meta["non_special_day"] = 1 - ret_meta["special_day"]

    ret_daily = ret_daily.copy()
    ret_daily["date"] = date_num
    ret_daily = ret_daily.drop(
        columns=["ym", "RF", "special_day", "non_special_day"], errors="ignore"
    ).merge(
        ret_meta[["date", "ym", "RF", "special_day", "non_special_day"]],
        on="date", how="left",
    )

    def _build_daily_port_series(wt_col, suffix=""):
        parts = []
        for ym, g in rg_port_wt_all.groupby("ym", sort=True):
            wt = g.set_index("permno")[wt_col].fillna(0.0)
            parts.append(get_port_daily_ret(wt, int(ym), ret_daily, suffix=suffix))
        out = pd.concat(parts, ignore_index=True)
        return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    vw_daily = _build_daily_port_series("vwMKT_wt", suffix="_vw")
    rg_daily = _build_daily_port_series("Rg_wt", suffix="_rg")
    mktg_daily = _build_daily_port_series("mktg_wt", suffix="_mktg")

    daily_ports = (
        vw_daily.rename(columns={"ret_ex_vw": PORT_VWMKT})
        .merge(rg_daily[["date", "ret_ex_rg"]].rename(columns={"ret_ex_rg": PORT_RG}), on="date", how="inner")
        .merge(mktg_daily[["date", "ret_ex_mktg"]].rename(columns={"ret_ex_mktg": PORT_MKTG}), on="date", how="inner")
    )
    daily_ports[PORT_LNF] = daily_ports[PORT_VWMKT] - daily_ports[PORT_MKTG]

    port_map = {
        "Mkt-RF": PORT_VWMKT,
        "Rg-RF": PORT_RG,
        "MKTg-RF": PORT_MKTG,
        PORT_LNF: PORT_LNF,
    }
    group_map = {
        "Full": slice(None),
        "Special days": daily_ports["special_day"].eq(1),
        "Non-special days": daily_ports["non_special_day"].eq(1),
    }

    rows = []
    for label, col in port_map.items():
        row = {"Portfolio": label}
        sub_series = {}
        for group, mask in group_map.items():
            series = daily_ports[col] if isinstance(mask, slice) else daily_ports.loc[mask, col]
            st = _ann_stats_daily(series)
            row[f"{group} Ret"] = st["AnnRet(%)"]
            row[f"{group} Vol"] = st["AnnVol(%)"]
            row[f"{group} SR"] = st["SR"]
            sub_series[group] = series.dropna().values
        # SR(Special) − SR(Non-special) + NW two-independent-sample t-stat (lags=10, daily).
        sp = sub_series["Special days"]
        ns = sub_series["Non-special days"]
        sr_sp = _ann_stats_daily(pd.Series(sp))["SR"]
        sr_ns = _ann_stats_daily(pd.Series(ns))["SR"]
        t_stat = SR_diff_test_indep(sp, ns, lags=10)
        row["SR_diff"] = (sr_sp - sr_ns) if (pd.notna(sr_sp) and pd.notna(sr_ns)) else np.nan
        row["SR_diff_t"] = float(t_stat) if pd.notna(t_stat) else np.nan
        rows.append(row)
    stats_df = pd.DataFrame(rows)

    print("\nDaily In-sample Decomposition Stats")
    print(" " * 26 + "Full" + " " * 21 + "Special days" + " " * 12 + "Non-special days"
          + " " * 6 + "SR diff (Sp-NS)")
    print(" " * 20 + "Ret     Vol    SR" + " " * 9 + "Ret     Vol    SR"
          + " " * 9 + "Ret     Vol    SR"
          + " " * 6 + "diff      t-stat")
    print("-" * 120)
    for _, r in stats_df.iterrows():
        full = _fmt_triplet((r["Full Ret"], r["Full Vol"], r["Full SR"]))
        special = _fmt_triplet((r["Special days Ret"], r["Special days Vol"], r["Special days SR"]))
        nonspecial = _fmt_triplet((r["Non-special days Ret"], r["Non-special days Vol"], r["Non-special days SR"]))
        diff_str = _fmt_num_plain(r["SR_diff"]) + _sig_stars_latex(r["SR_diff_t"])
        t_str = _fmt_num_plain(r["SR_diff_t"])
        print(f"{r['Portfolio']:18}  {full}  {special}  {nonspecial}  {diff_str:>12} {t_str:>8}")

    if latex_filename is not None:
        lines = []
        lines.append(r"\begin{table}[h!]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabularx}{0.95\textwidth}{l *{11}{>{\centering\arraybackslash}X}}")
        lines.append(r"\toprule")
        lines.append(
            r"& \multicolumn{3}{c}{Full} & \multicolumn{3}{c}{Special days} & \multicolumn{3}{c}{Non-special days}"
            r" & \multicolumn{2}{c}{SR diff (Sp $-$ NS)} \\"
        )
        lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-12}")
        lines.append(
            r"& Ret(\%) & Vol(\%) & SR & Ret(\%) & Vol(\%) & SR & Ret(\%) & Vol(\%) & SR"
            r" & diff & $t$-stat \\"
        )
        lines.append(r"\midrule")
        for _, r in stats_df.iterrows():
            diff_tex = _fmt_num_plain(r["SR_diff"]) + _sig_stars_latex(r["SR_diff_t"])
            t_tex = _fmt_num_plain(r["SR_diff_t"])
            lines.append(
                f"{_esc_tex(r['Portfolio'])} & "
                f"{_fmt_pct(r['Full Ret'], signed=False).strip().replace('%', '')} & "
                f"{_fmt_pct(r['Full Vol'], signed=False).strip().replace('%', '')} & "
                f"{_fmt_num(r['Full SR']).strip()} & "
                f"{_fmt_pct(r['Special days Ret'], signed=False).strip().replace('%', '')} & "
                f"{_fmt_pct(r['Special days Vol'], signed=False).strip().replace('%', '')} & "
                f"{_fmt_num(r['Special days SR']).strip()} & "
                f"{_fmt_pct(r['Non-special days Ret'], signed=False).strip().replace('%', '')} & "
                f"{_fmt_pct(r['Non-special days Vol'], signed=False).strip().replace('%', '')} & "
                f"{_fmt_num(r['Non-special days SR']).strip()} & "
                f"{diff_tex} & {t_tex} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabularx}")
        lines.append(r"\end{table}")
        latex_file = Path(result_path) / latex_filename
        with open(latex_file, "w") as f:
            f.write("\n".join(lines))

    return daily_ports, stats_df



def compute_robustness_at_kappa(hf_data, pca_df, data, bench_wt, bench_ret,
                                shrink_coef=RG_SELECT_SCALE,
                                update_lnf_exposure_proxy=False,
                                verbose=False):
    """Build Rg/MKTg/LNF at one `shrink_coef`; collect robustness metrics.
    Returns (metrics, rg_ts):
      metrics: dict with kappa, Rg/LNF SR (full/intra/over IS + full OOS), avg
               Rg ∩ BetaFP_D1 overlap, conditional SR by overlap (median split),
               low_risk α + t for CAPM and Rg+LNF decomp.
      rg_ts:   DataFrame[ym, ret_ex_is, ret_ex_oos] of monthly Rg returns, OOS-aligned.
    Caller passes pre-built `bench_wt`/`bench_ret` (κ-independent)."""
    rg_wt, rg_ret, _proxy, _phi = build_rg_factors(
        hf_data, pca_df, data,
        shrink_coef=shrink_coef,
        update_lnf_exposure_proxy=update_lnf_exposure_proxy,
        verbose=verbose,
    )
    port_ret = pd.concat([bench_ret, rg_ret], ignore_index=True)
    port_wt  = bench_wt.merge(rg_wt, on=["ym", "permno"], how="outer").fillna(0)
    port_ret = align_oos_to_realization(port_ret)

    metrics = {"kappa": shrink_coef}

    measures = [("full", "ret_ex_is"), ("intra", "intra_ex_is"), ("over", "over_ex_is")]
    for port in [PORT_RG, PORT_LNF]:
        sub = port_ret[port_ret["port"] == port]
        for tag, col in measures:
            metrics[f"{port}_{tag}_sr"] = _ann_stats(sub[col])["SR"]
        # Full-month out-of-sample Sharpe (OOS column already shifted by align_oos_to_realization).
        metrics[f"{port}_full_oos_sr"] = _ann_stats(sub["ret_ex_oos"])["SR"]

    # Average month-by-month overlap rate between Rg holdings and BetaFP D1 (low-β decile).
    rg_summary = rg_stocks_summary(port_wt)
    overlap_ym = rg_summary[["ym", "overlap_vs_bfp_d1"]].copy()
    overlap_ym["ym"] = overlap_ym["ym"].astype(int)
    metrics["Rg_avg_overlap_lowbeta"] = float(overlap_ym["overlap_vs_bfp_d1"].mean())

    # Median split on overlap; SR(Low) − SR(High), full IS — matches build_conditional_stats.
    med = overlap_ym["overlap_vs_bfp_d1"].median()
    overlap_ym["Group"] = np.where(overlap_ym["overlap_vs_bfp_d1"] >= med, "High", "Low")
    overlap_map = overlap_ym.set_index("ym")["Group"]
    for port in [PORT_RG, PORT_LNF]:
        sub = port_ret[port_ret["port"] == port][["ym", "ret_ex_is"]].copy()
        sub["ym"] = sub["ym"].astype(int)
        sub["Group"] = sub["ym"].map(overlap_map)
        r_hi = sub.loc[sub["Group"] == "High", "ret_ex_is"].dropna().values
        r_lo = sub.loc[sub["Group"] == "Low",  "ret_ex_is"].dropna().values
        metrics[f"{port}_high_sr"] = _ann_stats(pd.Series(r_hi))["SR"]
        metrics[f"{port}_low_sr"]  = _ann_stats(pd.Series(r_lo))["SR"]
        sr_hi, sr_lo = metrics[f"{port}_high_sr"], metrics[f"{port}_low_sr"]
        t_stat = SR_diff_test_indep(r_lo, r_hi, lags=10)
        metrics[f"{port}_sr_diff"]   = (sr_lo - sr_hi) if (pd.notna(sr_hi) and pd.notna(sr_lo)) else np.nan
        metrics[f"{port}_sr_diff_t"] = float(t_stat) if pd.notna(t_stat) else np.nan

    reg_df = prepare_insample_spanning_df(port_ret, data["ff6"], g_port=PORT_RG, ret_col="ret_ex")
    lr_models = compare_capm_spanning(reg_df, span_cols=["low_risk"], pair_cols=("g_rf", "lnf"))
    # [0] low_risk ~ MKT_RF ;  [1] low_risk ~ g_rf + lnf
    metrics["lowrisk_capm_alpha"]   = float(lr_models[0].params["Intercept"])
    metrics["lowrisk_capm_t"]       = float(lr_models[0].tvalues["Intercept"])
    metrics["lowrisk_decomp_alpha"] = float(lr_models[1].params["Intercept"])
    metrics["lowrisk_decomp_t"]     = float(lr_models[1].tvalues["Intercept"])

    rg_ts = (port_ret.loc[port_ret["port"] == PORT_RG, ["ym", "ret_ex_is", "ret_ex_oos"]]
             .assign(ym=lambda d: d["ym"].astype(int))
             .sort_values("ym").reset_index(drop=True))
    print(f"  κ={metrics['kappa']:.2f}: "
          f"Rg SR={metrics['Rg_full_sr']:+.2f} (oos {metrics['Rg_full_oos_sr']:+.2f})  "
          f"LNF SR={metrics['LNF_full_sr']:+.2f} (oos {metrics['LNF_full_oos_sr']:+.2f})  "
          f"Rg+LNF α={metrics['lowrisk_decomp_alpha']:+.2f} (t={metrics['lowrisk_decomp_t']:+.2f})")
    return metrics, rg_ts



def print_latex_robustness_table(metrics_df, latex_filename="robustness_shrink_coef.tex"):
    """LaTeX κ-sweep table: κ, IS SR Rg/LNF (full/intra/over), avg overlap,
    Rg cond SR (Hi/Lo), low_risk α & t, OOS SR (Rg/LNF). 2 decimals."""
    def _fnp(v):
        return _fmt_num_plain(v) if pd.notna(v) else ""

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabularx}{\textwidth}{l *{13}{>{\centering\arraybackslash}X}}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{3}{c}{IS SR -- Rg}"
        r" & \multicolumn{3}{c}{IS SR -- LNF}"
        r" & "
        r" & \multicolumn{2}{c}{Rg cond. (overlap)}"
        r" & \multicolumn{2}{c}{low\_risk $\alpha$ (Rg+LNF)}"
        r" & \multicolumn{2}{c}{OOS SR (full)} \\"
    )
    lines.append(
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} "
        r"\cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14}"
    )
    lines.append(
        r"$\kappa$ & full & intra & over & full & intra & over"
        r" & overlap (avg) & Hi & Lo & $\alpha$ & $t$ & Rg & LNF \\"
    )
    lines.append(r"\midrule")

    for kappa, r in metrics_df.iterrows():
        cells = [
            f"{kappa:.2f}",
            _fnp(r["Rg_full_sr"]),  _fnp(r["Rg_intra_sr"]),  _fnp(r["Rg_over_sr"]),
            _fnp(r["LNF_full_sr"]), _fnp(r["LNF_intra_sr"]), _fnp(r["LNF_over_sr"]),
            _fnp(r["Rg_avg_overlap_lowbeta"]),
            _fnp(r["Rg_high_sr"]),  _fnp(r["Rg_low_sr"]),
            _fnp(r["lowrisk_decomp_alpha"]), _fnp(r["lowrisk_decomp_t"]),
            _fnp(r["Rg_full_oos_sr"]),  _fnp(r["LNF_full_oos_sr"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")

    out = Path(result_path) / latex_filename
    out.write_text("\n".join(lines))
