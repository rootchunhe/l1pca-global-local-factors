import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from constants import (
    data_path, result_path, case_path, ret_path, spy_path,
    spy_5min_path, fret30min_ym_path, N_BARS_PER_DAY, N_BARS_30,
    PCA_GLOBAL_FACTOR_FOLD, CASE_SPEC,
)
from pca import get_monthly_data
from analysis_utils import _esc_tex, _fmt_num_plain, get_rBetas

def _clustered_corr(C):
    """Reorder correlation by hierarchical clustering."""
    if C.shape[0] < 2:
        return C.copy()
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
    C = C.copy()
    np.fill_diagonal(C, 1)
    C = np.nan_to_num(C, nan=0)
    D = squareform(1 - C, checks=False)
    Z = optimal_leaf_ordering(linkage(D, "average"), D)
    perm = leaves_list(Z)
    return C[perm][:, perm]


def _plot_resid_corr(dSTK_r, top_ix, bot_ix, event_ix, ym, case_name, split_panels=False):
    """Plot residual correlation heatmaps (4 panels). Returns rCorr_event (event x event)."""
    rCorr = np.nan_to_num(np.corrcoef(dSTK_r.T), nan=0)
    rCorr_event = rCorr[np.ix_(event_ix, event_ix)].copy()
    out = Path(case_path) / case_name
    clim = (rCorr.min(), rCorr.max())
    panels = [
        _clustered_corr(rCorr),
        _clustered_corr(rCorr_event),
        _clustered_corr(rCorr[np.ix_(top_ix, top_ix)]),
        _clustered_corr(rCorr[np.ix_(bot_ix, bot_ix)]),
    ]
    titles = ["All", f"Event (n={len(event_ix)})", "Top quintile", "Bot quintile"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    im = None
    for a, C, t in zip(ax.flat, panels, titles):
        im = a.imshow(C, aspect="auto", clim=clim, cmap="turbo")
        a.set_title(t)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out / f"{case_name}_{ym}_res_rCorr.pdf", bbox_inches="tight")
    plt.close()
    if split_panels:
        split_specs = [
            ((0, 1), f"{case_name}_{ym}_sub0.pdf"),
            ((2, 3), f"{case_name}_{ym}_sub1.pdf"),
        ]
        for idxs, fname in split_specs:
            fig_sub, ax_sub = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
            im_sub = None
            for a, idx in zip(ax_sub.flat, idxs):
                im_sub = a.imshow(panels[idx], aspect="auto", clim=clim, cmap="turbo")
                a.set_title(titles[idx])
            fig_sub.colorbar(im_sub, ax=ax_sub, shrink=0.85)
            fig_sub.savefig(out / fname, bbox_inches="tight")
            plt.close(fig_sub)
    return rCorr_event


def _build_crsp_sp500(crsp):
    """Pre-merge crsp ⨝ sp500 membership; built once per run_case_study (vs per month)."""
    sp500_ym = pd.read_csv(Path(spy_path) / "sp500_list_ym.csv", low_memory=False)
    sp500_ym["ym"] = pd.to_numeric(sp500_ym["ym"], errors="coerce").astype(int)
    sp500_ym["permno"] = pd.to_numeric(sp500_ym["permno"], errors="coerce").astype(int)
    sp500_pairs = sp500_ym[["ym", "permno"]].drop_duplicates()

    out = crsp[["ym", "permno", "lag_mktcap"]].copy()
    out["ym"] = pd.to_numeric(out["ym"], errors="coerce").astype(int)
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype(int)
    out["lag_mktcap"] = pd.to_numeric(out["lag_mktcap"], errors="coerce")
    return out.merge(sp500_pairs, on=["ym", "permno"], how="inner")


def get_sp500_subset_weight(crsp_sp500, ym, stock_list):
    """Total start-of-month SP500 weight for stock_list using lag_mktcap.
    crsp_sp500 must be pre-merged via _build_crsp_sp500."""
    month_df = crsp_sp500[crsp_sp500["ym"] == int(ym)].drop_duplicates("permno", keep="last").copy()
    month_df["cluster_wt"] = month_df["lag_mktcap"] / month_df["lag_mktcap"].sum()
    return float(month_df.set_index("permno")["cluster_wt"].reindex(pd.Index(stock_list, dtype=int)).sum())


def run_case_study(case_name):
    """Run case study for t-1, t, t+1. Returns (summary_tab, stocks_df).
    Event permnos: NAICS-matched stocks present in cRet data in all three months."""
    if case_name not in CASE_SPEC:
        raise ValueError(f"Unknown case: {case_name}")
    spec = CASE_SPEC[case_name]
    out = Path(case_path) / case_name
    if out.exists():
        print(f"  removing existing output dir: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    pca  = pd.read_csv(Path(result_path) / "monthly_pcas.csv")
    crsp = pd.read_csv(Path(data_path) / "crsp_taq_monthly_info.csv", low_memory=False)
    crsp_sp500 = _build_crsp_sp500(crsp)
    daily_ret = pd.read_csv(Path(ret_path) / "ret_daily_data.csv")
    daily_ret["date"] = daily_ret["date"].astype(int)

    ed = pd.Timestamp(spec["event_date"] + "-01")
    months_ts = [ed + pd.DateOffset(months=k) for k in (-1, 0, 1)]
    suffixes  = ["t_1", "t", "t1"]

    # SPY 30min (right-labeled, intraday bars only, resampled from 5min log returns)
    spy5_raw = pd.read_csv(spy_5min_path)
    spy5_raw["datetime"] = pd.to_datetime(spy5_raw["datetime"])
    spy30_series = (spy5_raw.set_index("datetime")["SPY"]
                    .resample("30min", closed="right", label="right").sum())
    spy30_series = spy30_series[spy30_series.index.hour * 60 + spy30_series.index.minute >= 630]

    # Pass 1: load monthly data, identify event permnos per month
    monthly = []
    for dt in months_ts:
        ym      = dt.strftime("%Y-%m")
        yyyymm  = int(ym.replace("-", ""))
        pca_ym  = pca[pca["YM"] == yyyymm]
        crsp_ym = crsp[crsp["ym"] == yyyymm].drop_duplicates("permno", keep="last")

        dSTK, dSPY, stock_info = get_monthly_data(ym)

        naics_map = crsp_ym.set_index("permno")["naics"].fillna("").astype(str).to_dict()
        stock_info["naics"] = stock_info["permno"].map(naics_map).fillna("")
        event_permnos = (set(stock_info.loc[stock_info["naics"].str.startswith(str(spec["naics"])), "permno"].astype(int))
                         - set(spec["remove"]))

        q = 1.0 / PCA_GLOBAL_FACTOR_FOLD
        stk_news    = pca_ym["news_proxy"].values
        top_permnos = set(pca_ym.loc[stk_news > np.nanquantile(stk_news, 1 - q), "permno"].astype(int))
        bot_permnos = set(pca_ym.loc[stk_news < np.nanquantile(stk_news, q),     "permno"].astype(int))

        monthly.append(dict(ym=ym, dt=dt, dSTK=dSTK, dSPY=dSPY, stock_info=stock_info,
                            event_permnos=event_permnos,
                            top_permnos=top_permnos, bot_permnos=bot_permnos))

    # Common event permnos: present in cRet data in all 3 months
    common_permnos = set.intersection(*[m["event_permnos"] for m in monthly])
    all_event      = set.union(*[m["event_permnos"] for m in monthly])
    removed_no_3m  = sorted(all_event - common_permnos)
    ticker_map: dict[int, str] = {}
    for m in monthly:
        si = m["stock_info"].dropna(subset=["ticker"])
        ticker_map.update(zip(si["permno"].astype(int), si["ticker"].astype(str).str.strip()))
    def _ticker(p):
        t = ticker_map.get(int(p), "")
        return t if t else str(p)

    remove_list = set(spec["remove"])
    if remove_list or removed_no_3m:
        print(f"\n{case_name} — kept: NAICS match AND present in cRet for all 3 months")
    if remove_list:
        print(f"  Excluded (manual): {[_ticker(p) for p in sorted(remove_list)]}")
    if removed_no_3m:
        print(f"  Excluded (missing ≥1 month): {[_ticker(p) for p in removed_no_3m]}")

    target_yyyymm = [int(dt.strftime("%Y%m")) for dt in months_ts]
    betas = pd.read_csv(
        Path(data_path) / "monthly_lagged_betas.csv",
        usecols=["permno", "yyyymm", "beta_252d"],
        low_memory=False,
    )
    betas["permno"] = betas["permno"].astype(int)
    betas["yyyymm"] = betas["yyyymm"].astype(int)
    betas = betas.loc[betas["permno"].isin(common_permnos)].sort_values(["permno", "yyyymm"])
    betas["beta_252d_t"] = betas.groupby("permno")["beta_252d"].shift(-1)  # shift(-1) undoes upstream lag -> contemporaneous
    betas = betas[betas["yyyymm"].isin(target_yyyymm)].copy()
    beta_map = betas.set_index(["yyyymm", "permno"])["beta_252d_t"].to_dict()

    # Pass 2: compute stats for common_permnos
    all_data  = {}
    agg_rows  = []
    for m, suffix in zip(monthly, suffixes):
        ym, dt      = m["ym"], m["dt"]
        yyyymm      = int(ym.replace("-", ""))
        dSTK, dSPY  = m["dSTK"], m["dSPY"]
        stock_info  = m["stock_info"]

        event_ix = np.where(stock_info["permno"].isin(common_permnos))[0]
        event_permnos = stock_info.iloc[event_ix]["permno"].astype(int).tolist()
        top_ix   = np.where(stock_info["permno"].isin(m["top_permnos"]))[0]
        bot_ix   = np.where(stock_info["permno"].isin(m["bot_permnos"]))[0]
        event_overall_weight = get_sp500_subset_weight(crsp_sp500, yyyymm, event_permnos)

        # Residual correlation plot (lagged cBeta, full stock universe).
        cbeta0_full = stock_info["cBeta0"].values
        dSTK_r    = dSTK - dSPY[:, None] * cbeta0_full
        rCorr_evt = _plot_resid_corr(
            dSTK_r, top_ix, bot_ix, event_ix, ym, case_name, split_panels=(suffix == "t")
        )

        # c-stats (5min cRet, intraday, first bar per day dropped; N_BARS_PER_DAY+1 = full-day total)
        cBetas = get_rBetas(dSPY, dSTK[:, event_ix])
        cVols  = np.sqrt((dSTK[:, event_ix] ** 2).sum(axis=0) / dSTK.shape[0] * 252 * (N_BARS_PER_DAY + 1))
        if len(event_ix) > 1:
            i, j = np.tril_indices(len(event_ix), k=-1)
            cCorr = float(np.nanmean(rCorr_evt[i, j]))
        else:
            cCorr = np.nan

        daily_cols = [str(p) for p in event_permnos if str(p) in daily_ret.columns]
        dVols = np.full(len(event_permnos), np.nan)
        dVol_mean = np.nan
        if daily_cols:
            daily_ym = daily_ret.loc[(daily_ret["date"] // 100) == yyyymm, daily_cols]
            if not daily_ym.empty:
                dVol_map = np.sqrt(252.0 / daily_ym.shape[0] * (daily_ym.astype(float) ** 2).sum(axis=0))
                dVols = np.array([dVol_map.get(str(p), np.nan) for p in event_permnos], dtype=float)
                dVol_mean = float(np.nanmean(dVols))

        # r-stats (30min fRet, intraday, first bar dropped; N_BARS_30+1 = full-day total)
        r_by_permno = {}
        rBeta_mean = rVol_mean = rCorr_mean = np.nan
        f30_path = Path(fret30min_ym_path) / f"fret30min_{ym}.csv"
        if f30_path.exists():
            fret30    = pd.read_csv(f30_path, index_col="datetime", parse_dates=True)
            spy30_ym  = spy30_series[(spy30_series.index.year == dt.year) & (spy30_series.index.month == dt.month)]
            common_dt = fret30.index.intersection(spy30_ym.index)
            sub_cols  = [str(p) for p in event_permnos if str(p) in fret30.columns]
            if len(common_dt) > 0 and sub_cols:
                dSTK_30  = fret30.loc[common_dt, sub_cols].values.astype(float)
                dSPY_30  = spy30_ym.loc[common_dt].values.astype(float)
                rb = get_rBetas(dSPY_30, dSTK_30)
                rv = np.sqrt((dSTK_30 ** 2).sum(axis=0) / dSTK_30.shape[0] * 252 * (N_BARS_30 + 1))
                # Residual via LAGGED cBeta0 (5-min CAPM beta), same convention as the 5-min path.
                cb_lookup = stock_info.set_index("permno")["cBeta0"]
                sub_cbeta0 = cb_lookup.reindex([int(c) for c in sub_cols]).fillna(0).values
                dSTK_30_r = dSTK_30 - dSPY_30[:, None] * sub_cbeta0
                if dSTK_30_r.shape[1] > 1:
                    rcorr30 = np.corrcoef(dSTK_30_r.T)
                    ii, jj = np.tril_indices(dSTK_30_r.shape[1], k=-1)
                    rCorr_mean = float(np.nanmean(rcorr30[ii, jj]))
                rBeta_mean = float(np.nanmean(rb))
                rVol_mean  = float(np.nanmean(rv))
                r_by_permno = {col: (rb[j], rv[j]) for j, col in enumerate(sub_cols)}

        agg_rows.append(dict(month=suffix,
                             cBeta=cBetas.mean(), cVol=cVols.mean(), cCorr=cCorr,
                             rBeta=rBeta_mean,    rVol=rVol_mean,    rCorr=rCorr_mean,
                             dVol=dVol_mean,
                             beta_252d=np.nanmean([beta_map.get((yyyymm, int(p)), np.nan)
                                                  for p in event_permnos]),
                             cluster_wt=event_overall_weight,
                             nstks=len(event_ix)))

        for k, (_, row) in enumerate(stock_info.iloc[event_ix].iterrows()):
            p = int(row["permno"])
            if p not in all_data:
                all_data[p] = {"permno": p, "ticker": row.get("ticker", "")}
            all_data[p][f"cBeta_{suffix}"] = cBetas[k]
            all_data[p][f"cVol_{suffix}"]  = cVols[k]
            pstr = str(p)
            if pstr in r_by_permno:
                all_data[p][f"rBeta_{suffix}"] = r_by_permno[pstr][0]
                all_data[p][f"rVol_{suffix}"]  = r_by_permno[pstr][1]
            all_data[p][f"dVol_{suffix}"] = dVols[k]
            all_data[p][f"beta_252d_{suffix}"] = beta_map.get((yyyymm, p), np.nan)

    df = pd.DataFrame(all_data.values())
    ccols = [f"cBeta_{s}" for s in suffixes] + [f"cVol_{s}" for s in suffixes]
    rcols = [f"rBeta_{s}" for s in suffixes] + [f"rVol_{s}" for s in suffixes]
    dcols = [f"dVol_{s}" for s in suffixes]
    bcols = [f"beta_252d_{s}" for s in suffixes]
    df = df.reindex(columns=["permno", "ticker"] + ccols + rcols + dcols + bcols)
    df.to_csv(out / f"{case_name}_stocks.csv", index=False)

    summary_tab = pd.DataFrame(agg_rows).set_index("month")
    summary_tab.to_csv(out / f"{case_name}_stats.csv")

    included_tickers = [_ticker(p) for p in sorted(common_permnos)]
    excluded_manual_tickers = [_ticker(p) for p in sorted(remove_list)]
    excluded_missing_tickers = [_ticker(p) for p in removed_no_3m]
    summary_cols = ["cBeta", "cVol", "cCorr", "rBeta", "rVol", "rCorr", "dVol", "beta_252d", "cluster_wt", "nstks"]
    summary_txt = summary_tab[summary_cols].to_string()

    row_name_map = {"t_1": "prev. month", "t": "event month", "t1": "next month"}
    row_order = suffixes
    case_title_map = {"oil_shock": "Oil shock", "svb": "SVB", "deepseek": "DeepSeek shock"}
    event_month_label = pd.Timestamp(spec["event_date"] + "-01").strftime("%B %Y")
    n_case_stocks = len(common_permnos)
    naics_label = str(spec["naics"]) + ("*" if case_name in {"oil_shock", "svb"} else "")
    panel_desc = (
        f"{_esc_tex(case_name)}: {_esc_tex(case_title_map.get(case_name, case_name))} "
        f"({event_month_label}; {n_case_stocks} stocks; NAICS {naics_label}"
    )
    if excluded_manual_tickers:
        panel_desc += f", excluding {', '.join(_esc_tex(t) for t in excluded_manual_tickers)}"
    panel_desc += ")"
    caption_parts = [
        r"\textbf{Case studies: residual correlation, volatility, and market beta.} "
        + f"This table reports summary statistics for the {case_title_map.get(case_name, case_name).lower()} case study "
        + f"centered on {event_month_label}.",
        f"The sample consists of S\\&P 500 firms satisfying the corresponding NAICS filter ({naics_label}).",
        r"Superscripts $c$ and $f$ denote statistics based on 5-minute continuous returns and 30-minute intraday full returns, respectively.",
        r"We report average annualized volatilities, realized betas with respect to SPY, and average pairwise residual correlations, together with annualized realized volatility based on daily returns and the contemporaneous rolling CAPM beta estimated from 252 daily returns ending within the event month.",
        r"Residual returns are obtained by removing market exposure using lagged realized betas estimated from 5-minute continuous returns.",
        r"Firms are required to remain in the 5-minute continuous-return sample over the month before, the event month, and the month after the event.",
    ]
    if excluded_manual_tickers:
        if case_name == "oil_shock":
            caption_parts.append(
                r"Excluded tickers are "
                + ", ".join(_esc_tex(t) for t in excluded_manual_tickers)
                + r" because they are not upstream oil and gas producers."
            )
        elif case_name == "svb":
            caption_parts.append(
                r"Excluded tickers are "
                + ", ".join(_esc_tex(t) for t in excluded_manual_tickers)
                + r" because SIVB and SBNY do not remain in the three-month event window and SYF, AXP, COF, and DFS are more naturally classified as consumer-finance firms than deposit-run exposure banks."
            )
        elif case_name == "deepseek":
            caption_parts.append(
                r"Excluded tickers are "
                + ", ".join(_esc_tex(t) for t in excluded_manual_tickers)
                + r" because they are not semiconductor firms despite sharing the same broad NAICS classification."
            )
        else:
            caption_parts.append(
                r"Excluded tickers are " + ", ".join(_esc_tex(t) for t in excluded_manual_tickers) + "."
            )
    if excluded_missing_tickers:
        caption_parts.append(
            r"Additional NAICS-matched tickers excluded due to missing data in at least one of the three event-window months are "
            + ", ".join(_esc_tex(t) for t in excluded_missing_tickers)
            + "."
        )
    caption_parts.append(r"The event month is highlighted in boldface.")
    latex_caption = " ".join(caption_parts)

    tab_lines = []
    tab_lines.append(r"\begin{table}[h!]")
    tab_lines.append(r"\centering")
    tab_lines.append(r"\caption{" + latex_caption + r"}")
    tab_lines.append(r"\begin{tabularx}{\textwidth}{l *{8}{>{\centering\arraybackslash}X}}")
    tab_lines.append(r"\toprule")
    tab_lines.append(
        r" & \multicolumn{3}{c}{Intraday 5-min (continuous)} & "
        r"\multicolumn{3}{c}{Intraday 30-min (full)} & Daily RV & Daily \\"
    )
    tab_lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-8} \cmidrule(lr){9-9}")
    tab_lines.append(
        r" & $\sigma^{c}$ & $\beta^{c}$ & $\overline{\rho}^{c}$ & "
        r"$\sigma^{f}$ & $\beta^{f}$ & $\overline{\rho}^{f}$ & $\sigma^{d}$ & $\beta^{252d}$ \\"
    )
    tab_lines.append(r"\midrule")
    tab_lines.append(rf"\multicolumn{{9}}{{l}}{{{panel_desc}}} \\")
    tab_lines.append(r"\midrule")
    for rk in row_order:
        rr = summary_tab.loc[rk]
        vals = [
            row_name_map.get(rk, rk),
            _fmt_num_plain(rr.get('cVol', np.nan)),
            _fmt_num_plain(rr.get('cBeta', np.nan)),
            _fmt_num_plain(rr.get('cCorr', np.nan)),
            _fmt_num_plain(rr.get('rVol', np.nan)),
            _fmt_num_plain(rr.get('rBeta', np.nan)),
            _fmt_num_plain(rr.get('rCorr', np.nan)),
            _fmt_num_plain(rr.get('dVol', np.nan)),
            _fmt_num_plain(rr.get('beta_252d', np.nan)),
        ]
        if rk == "t":
            vals = [rf"\textbf{{{v}}}" for v in vals]
        tab_lines.append(" & ".join(vals) + r" \\")
    tab_lines.append(r"\bottomrule")
    tab_lines.append(r"\end{tabularx}")
    tab_lines.append(r"\end{table}")
    latex_table = "\n".join(tab_lines)

    tex_lines = []
    tex_lines.append("% Please add the following required packages to your document preamble:")
    tex_lines.append("% \\usepackage{booktabs}")
    tex_lines.append("% \\usepackage{tabularx}")
    tex_lines.append("% \\usepackage{array}")
    tex_lines.append(f"% {case_name} - kept: NAICS match AND present in cRet for all 3 months")
    tex_lines.append(f"% Excluded (manual): {excluded_manual_tickers}")
    if excluded_missing_tickers:
        tex_lines.append(f"% Excluded (missing >=1 month): {excluded_missing_tickers}")
    tex_lines.append(f"% Included tickers: {included_tickers}")
    tex_lines.append("")
    tex_lines.append(latex_table)
    (out / f"{case_name}_summary.tex").write_text("\n".join(tex_lines))

    sep = "=" * 60
    print(f"\n{sep}")
    print(f" case study of {case_name} completed: event month ({spec['event_date']})")
    print("  c: 5-min continuous returns  |  r: 30-min full returns  |  d: daily")
    print(f"{sep}\n")
    print(summary_txt)
    print(f"{sep}\n")

    return summary_tab, df
