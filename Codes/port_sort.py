import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from constants import (
    data_path, ret_path, result_path, plot_path,
)
from analysis_utils import (
    _esc_tex, _fmt_pct_from_dec, _fmt_num_plain, _sig_stars_latex,
    _get_stats, _ols_intercept_slope_r2,
    _ann_stats,
    _weighted_mean, _cluster_mean_stats, _nw_mean_ci,
    _build_ym_permno_lookup, _vw_excess_return, SR_diff_test_HC1,
)




def load_signal_sorting_data(lnf_col="lnf_exposure_proxy"):
    """Load monthly data for signal decile sorting. chars[ym] = end-of-ym value."""
    lnf = pd.read_csv(Path(result_path) / "lnf_exposure_proxies.csv", usecols=["ym", "permno", lnf_col])
    chars = pd.read_csv(Path(data_path) / "monthly_more_jkp_chars_lagged.csv").rename(columns={"yyyymm": "ym"})
    crsp_taq = pd.read_csv(
        Path(data_path) / "crsp_taq_monthly_info.csv",
        usecols=["ym", "permno", "ret", "lag_mktcap"],
        low_memory=False,
    )
    ff6 = pd.read_csv(Path(ret_path) / "ff6_monthly_returns.csv", usecols=["yyyymm", "RF"])

    lnf["ym"] = pd.to_numeric(lnf["ym"], errors="coerce").astype("Int64")
    lnf["permno"] = pd.to_numeric(lnf["permno"], errors="coerce").astype("Int64")
    chars["ym"] = pd.to_numeric(chars["ym"], errors="coerce").astype("Int64")
    chars["permno"] = pd.to_numeric(chars["permno"], errors="coerce").astype("Int64")
    crsp_taq["ym"] = pd.to_numeric(crsp_taq["ym"], errors="coerce").astype("Int64")
    crsp_taq["permno"] = pd.to_numeric(crsp_taq["permno"], errors="coerce").astype("Int64")
    ff6["yyyymm"] = pd.to_numeric(ff6["yyyymm"], errors="coerce").astype("Int64")

    lnf = lnf.dropna(subset=["ym", "permno"]).copy()
    chars = chars.dropna(subset=["ym", "permno"]).copy()
    crsp_taq = crsp_taq.dropna(subset=["ym", "permno"]).copy()
    ff6 = ff6.dropna(subset=["yyyymm"]).copy()

    lnf["ym"] = lnf["ym"].astype(int)
    lnf["permno"] = lnf["permno"].astype(int)
    chars["ym"] = chars["ym"].astype(int)
    chars["permno"] = chars["permno"].astype(int)
    crsp_taq["ym"] = crsp_taq["ym"].astype(int)
    crsp_taq["permno"] = crsp_taq["permno"].astype(int)
    ff6["yyyymm"] = ff6["yyyymm"].astype(int)

    lnf = lnf.drop_duplicates(["ym", "permno"], keep="last")
    chars = chars.drop_duplicates(["ym", "permno"], keep="last")
    crsp_taq = crsp_taq.drop_duplicates(["ym", "permno"], keep="last")
    ff6 = ff6.drop_duplicates(["yyyymm"], keep="last")

    sample_end_ym = int(chars["ym"].max())
    lnf = lnf[lnf["ym"] <= sample_end_ym].copy()
    chars = chars[chars["ym"] <= sample_end_ym].copy()
    crsp_taq = crsp_taq[crsp_taq["ym"] <= sample_end_ym].copy()
    ff6 = ff6[ff6["yyyymm"] <= sample_end_ym].copy()

    chars = chars.sort_values(["permno", "ym"]).copy()
    char_cols = [c for c in chars.columns if c not in ["ym", "permno"]]
    # undo JKP lag → chars[ym] = end-of-ym (rank at month-end, hold ym+1).
    chars[char_cols] = chars.groupby("permno")[char_cols].shift(-1)
    # ffill cap=2: bridges one missing release month for quarterly signals without arbitrarily stale carry
    chars[char_cols] = chars.groupby("permno")[char_cols].ffill(limit=2)

    lnf_by_ym = _build_ym_permno_lookup(lnf, lnf_col)
    cap_by_ym = _build_ym_permno_lookup(crsp_taq, "lag_mktcap")
    ret_by_ym = _build_ym_permno_lookup(crsp_taq, "ret")
    rf_map = ff6.set_index("yyyymm")["RF"].to_dict()

    # rg_set_by_ym[ym]: in-sample Rg constituents from Rg_port_wt_all.csv (built by build_rg_factors).
    # Avoids re-deriving with a hardcoded fraction that could drift from decomposition.RG_SELECT_FOLD.
    rg_wt = pd.read_csv(
        Path(result_path) / "Rg_port_wt_all.csv",
        usecols=["ym", "permno", "Rg_wt"],
    )
    rg_wt["ym"] = pd.to_numeric(rg_wt["ym"], errors="coerce").astype("Int64")
    rg_wt["permno"] = pd.to_numeric(rg_wt["permno"], errors="coerce").astype("Int64")
    rg_wt = rg_wt.dropna(subset=["ym", "permno"]).copy()
    rg_wt["ym"] = rg_wt["ym"].astype(int)
    rg_wt["permno"] = rg_wt["permno"].astype(int)
    rg_wt = rg_wt[(rg_wt["ym"] <= sample_end_ym) & (rg_wt["Rg_wt"] > 0)]
    rg_set_by_ym = {
        int(ym): pd.Series(1.0, index=g["permno"].values)
        for ym, g in rg_wt.groupby("ym", sort=True)
    }

    return {
        "lnf_col": lnf_col,
        "sample_end_ym": sample_end_ym,
        "chars": chars,
        "char_cols": char_cols,
        "lnf_by_ym": lnf_by_ym,
        "cap_by_ym": cap_by_ym,
        "ret_by_ym": ret_by_ym,
        "rf_map": rf_map,
        "rg_set_by_ym": rg_set_by_ym,
    }



def sorting_decile_from_signal(data, signal_by_ym, signal_name, side="bottom", bottom_decile=0.1, top_decile=0.9):
    """Decile-sort `signal_by_ym` at ym, hold ym+1 VW by cap; return = decile_VW − RF.

    Two side portfolios from the same decile, partitioned by Rg membership at ym+1:
        overlap     = VW of (decile ∩ Rg_{ym+1}), weights renormalized in-subset
        non_overlap = VW of (decile \\ Rg_{ym+1}), weights renormalized in-subset
    Each fully invested (not additive); tests if anomaly premium concentrates in overlap.
    """
    if side not in {"bottom", "top"}:
        raise ValueError("side must be 'bottom' or 'top'")

    lnf_by_ym = data["lnf_by_ym"]
    cap_by_ym = data["cap_by_ym"]
    ret_by_ym = data["ret_by_ym"]
    rf_map = data["rf_map"]
    rg_set_by_ym = data["rg_set_by_ym"]

    rows = []
    yms = sorted(set(signal_by_ym.keys()) & set(lnf_by_ym.keys()) & set(cap_by_ym.keys()) & set(ret_by_ym.keys()))

    for i in range(len(yms) - 1):
        ym = yms[i]
        next_ym = yms[i + 1]

        signal_ym = signal_by_ym[ym]
        universe_ym = lnf_by_ym[ym]
        cap_ym = cap_by_ym[ym]
        ret_next_ym = ret_by_ym[next_ym]
        rg_set_next_ym = rg_set_by_ym.get(next_ym)
        rf_next_ym = float(rf_map.get(int(next_ym), 0.0))

        universe_idx = pd.to_numeric(universe_ym, errors="coerce").dropna().index

        s = pd.to_numeric(signal_ym.reindex(universe_idx), errors="coerce")
        m = pd.to_numeric(cap_ym.reindex(universe_idx), errors="coerce")
        valid = s.notna() & m.notna() & (m > 0)
        s = s[valid]
        m = m[valid]

        pct = s.rank(pct=True, method="first")
        sel = (pct <= bottom_decile) if side == "bottom" else (pct >= top_decile)
        cap_sel = m[sel]

        cap_sum = cap_sel.sum()
        w = cap_sel / cap_sum if cap_sum > 0 else cap_sel.astype(float)
        ret_ex = _vw_excess_return(ret_next_ym, cap_sel, rf_next_ym)

        if rg_set_next_ym is None:
            overlap = np.nan
            ret_overlap = np.nan
            ret_non_overlap = np.nan
        else:
            rg_indicator = rg_set_next_ym.reindex(w.index).fillna(0.0)
            in_mask = rg_indicator > 0.5
            out_mask = ~in_mask
            overlap = float((w * rg_indicator).sum())
            ret_overlap = _vw_excess_return(ret_next_ym, cap_sel[in_mask], rf_next_ym)
            ret_non_overlap = _vw_excess_return(ret_next_ym, cap_sel[out_mask], rf_next_ym)

        rows.append({
            "signal": signal_name,
            "side": side,
            "ym": ym,
            "next_ym": next_ym,
            "ret_ex_oos": ret_ex,
            "ret_ex_overlap_oos": ret_overlap,
            "ret_ex_non_overlap_oos": ret_non_overlap,
            "weighted_overlap": overlap,
            "n_selected": int(len(cap_sel)),
        })
    return pd.DataFrame(rows)



def summarize_signal_sorting(detail):
    rows = []
    for (side, signal), g in detail.groupby(["side", "signal"], sort=True):
        s_all = _ann_stats(g["ret_ex_oos"])
        s_overlap = _ann_stats(g["ret_ex_overlap_oos"])
        s_non_overlap = _ann_stats(g["ret_ex_non_overlap_oos"])
        pair = g[["ret_ex_overlap_oos", "ret_ex_non_overlap_oos"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 2:
            sr_diff_t = SR_diff_test_HC1(
                pair["ret_ex_overlap_oos"].values,
                pair["ret_ex_non_overlap_oos"].values,
            )
        else:
            sr_diff_t = np.nan
        rows.append({
            "side": side,
            "signal": signal,
            "ret": s_all["AnnRet(%)"] / 100 if pd.notna(s_all["AnnRet(%)"]) else np.nan,
            "std": s_all["AnnVol(%)"] / 100 if pd.notna(s_all["AnnVol(%)"]) else np.nan,
            "sr": s_all["SR"],
            "ret_overlap": s_overlap["AnnRet(%)"] / 100 if pd.notna(s_overlap["AnnRet(%)"]) else np.nan,
            "std_overlap": s_overlap["AnnVol(%)"] / 100 if pd.notna(s_overlap["AnnVol(%)"]) else np.nan,
            "sr_overlap": s_overlap["SR"],
            "ret_non_overlap": s_non_overlap["AnnRet(%)"] / 100 if pd.notna(s_non_overlap["AnnRet(%)"]) else np.nan,
            "std_non_overlap": s_non_overlap["AnnVol(%)"] / 100 if pd.notna(s_non_overlap["AnnVol(%)"]) else np.nan,
            "sr_non_overlap": s_non_overlap["SR"],
            "sr_diff": (
                s_overlap["SR"] - s_non_overlap["SR"]
                if (pd.notna(s_overlap["SR"]) and pd.notna(s_non_overlap["SR"]))
                else np.nan
            ),
            "sr_diff_t": sr_diff_t,
            "ovlp": float(pd.to_numeric(g["weighted_overlap"], errors="coerce").mean()),
            "months": int(pd.to_numeric(g["ret_ex_oos"], errors="coerce").notna().sum()),
        })
    out = pd.DataFrame(rows).sort_values(["side", "signal"]).reset_index(drop=True)
    num_cols = [c for c in out.columns if c not in ["side", "signal", "months"]]
    out[num_cols] = out[num_cols].round(4)
    return out



def run_signal_sorting_pack(data, long_bottom_decile, long_top_decile,
                            bottom_decile=0.1, top_decile=0.9):
    """Run decile sorting for a pack of signals and return (detail, summary, meta)."""
    chars = data["chars"]
    available_bottom = [s for s in long_bottom_decile if s in chars.columns]
    available_top = [s for s in long_top_decile if s in chars.columns]
    missing_bottom = [s for s in long_bottom_decile if s not in chars.columns]
    missing_top = [s for s in long_top_decile if s not in chars.columns]

    all_available = list(dict.fromkeys(available_bottom + available_top))
    signal_by_ym_map = {s: _build_ym_permno_lookup(chars[["ym", "permno", s]], s) for s in all_available}
    signal_jobs = [("lnf_exposure_proxy", data["lnf_by_ym"], "bottom")]
    signal_jobs += [(s, signal_by_ym_map[s], "bottom") for s in available_bottom]
    signal_jobs += [(s, signal_by_ym_map[s], "top") for s in available_top]

    detail_parts = []
    for signal_name, sdict, side in signal_jobs:
        detail_parts.append(
            sorting_decile_from_signal(
                data=data,
                signal_by_ym=sdict,
                signal_name=signal_name,
                side=side,
                bottom_decile=bottom_decile,
                top_decile=top_decile,
            )
        )

    detail = pd.concat(detail_parts, ignore_index=True).sort_values(["side", "signal", "ym"]).reset_index(drop=True)
    summary = summarize_signal_sorting(detail)
    meta = {
        "sample_end_ym": data["sample_end_ym"],
        "available_bottom": available_bottom,
        "available_top": available_top,
        "missing_bottom": missing_bottom,
        "missing_top": missing_top,
    }
    return detail, summary, meta



def print_latex_signal_sorting_table(summary_df, chars, side="bottom", latex_filename=None):
    """LaTeX table for OOS signal sorting summary. If latex_filename, save to Results/<latex_filename>."""
    if side not in {"bottom", "top"}:
        raise ValueError("side must be 'bottom' or 'top'")

    sub = summary_df.copy()
    sub = sub[sub["side"] == side].copy()
    sub = sub[sub["signal"].isin(chars)].copy()
    sub = sub.sort_values(["ovlp", "signal"], ascending=[False, True]).reset_index(drop=True)

    pretty = pd.DataFrame({
        "Signal": sub["signal"],
        "Ret": sub["ret"].map(_fmt_pct_from_dec),
        "Vol": sub["std"].map(_fmt_pct_from_dec),
        "SR": sub["sr"].map(_fmt_num_plain),
        "Ret ovlp": sub["ret_overlap"].map(_fmt_pct_from_dec),
        "Vol ovlp": sub["std_overlap"].map(_fmt_pct_from_dec),
        "SR ovlp": sub["sr_overlap"].map(_fmt_num_plain),
        "Ret non": sub["ret_non_overlap"].map(_fmt_pct_from_dec),
        "Vol non": sub["std_non_overlap"].map(_fmt_pct_from_dec),
        "SR non": sub["sr_non_overlap"].map(_fmt_num_plain),
        "Wgt ovlp": sub["ovlp"].map(_fmt_pct_from_dec),
        "SR diff": [
            _fmt_num_plain(v) + _sig_stars_latex(t)
            for v, t in zip(sub["sr_diff"], sub["sr_diff_t"])
        ],
        "t-stat": sub["sr_diff_t"].map(_fmt_num_plain),
    })
    print(f"\nSignal sorting summary ({side} decile)")
    print(pretty.to_string(index=False))

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabularx}{1\textwidth}{l *{9}{>{\centering\arraybackslash}p{0.05\textwidth}} c c c}")
    lines.append(r"\toprule")
    lines.append(
        rf"\multirow{{2}}{{*}}{{Signal}} & "
        rf"\multicolumn{{3}}{{c}}{{{side} decile}} & "
        r"\multicolumn{3}{c}{Subset: overlap with $R_g$} & "
        r"\multicolumn{3}{c}{Subset: non-overlap} & "
        r"\multirow{2}{*}{\begin{tabular}[c]{@{}c@{}}Weighted\\ overlap (\%)\end{tabular}} & "
        r"\multicolumn{2}{c}{SR} \\"
    )
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){12-13}")
    lines.append(
        r" & mean (\%) & vol (\%) & SR & mean (\%) & vol (\%) & SR & mean (\%) & vol (\%) & SR & & diff & $t$-stat \\"
    )
    lines.append(r"\midrule")

    for _, r in sub.iterrows():
        sr_diff_txt = _fmt_num_plain(r.get("sr_diff")) + _sig_stars_latex(r.get("sr_diff_t"))
        sr_diff_t_txt = _fmt_num_plain(r.get("sr_diff_t"))
        lines.append(
            f"{_esc_tex(r['signal'])} & "
            f"{_fmt_pct_from_dec(r.get('ret'))} & {_fmt_pct_from_dec(r.get('std'))} & {_fmt_num_plain(r.get('sr'))} & "
            f"{_fmt_pct_from_dec(r.get('ret_overlap'))} & {_fmt_pct_from_dec(r.get('std_overlap'))} & {_fmt_num_plain(r.get('sr_overlap'))} & "
            f"{_fmt_pct_from_dec(r.get('ret_non_overlap'))} & {_fmt_pct_from_dec(r.get('std_non_overlap'))} & {_fmt_num_plain(r.get('sr_non_overlap'))} & "
            f"{_fmt_pct_from_dec(r.get('ovlp'))} & {sr_diff_txt} & {sr_diff_t_txt} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if latex_filename is not None:
        (Path(result_path) / latex_filename).write_text(tex)



def build_hlr_panel(beta_col="Beta252d"):
    intraday = pd.read_csv(Path(ret_path) / "ret_intraday_data.csv").drop(columns=["SPY"])
    overnight = pd.read_csv(Path(ret_path) / "ret_overnight_data.csv").drop(columns=["SPY"])

    intraday = pd.melt(intraday, id_vars=["date"], var_name="permno", value_name="intraday_ret")
    overnight = pd.melt(overnight, id_vars=["date"], var_name="permno", value_name="overnight_ret")
    intraday["permno"] = intraday["permno"].astype(int)
    overnight["permno"] = overnight["permno"].astype(int)

    panel = intraday.merge(overnight, on=["date", "permno"], how="inner")
    panel["permno"] = panel["permno"].astype(int)
    panel["date"] = panel["date"].astype(int)
    panel["yyyymm"] = panel["date"] // 100

    char = pd.read_csv(Path(data_path) / "monthly_lagged_betas.csv", low_memory=False)
    char["permno"] = char["permno"].astype(int)
    char["yyyymm"] = char["yyyymm"].astype(int)
    char["size_exp"] = np.exp(pd.to_numeric(char["size_wrds"], errors="coerce"))
    char = char.rename(columns={"beta_252d": "Beta252d"})
    # beta_sort kept JKP-lagged: HLR/SML sort at month-start on prior-month β, evaluate intra-month.
    char_sub = char[["yyyymm", "permno", "size_exp", beta_col]].copy()
    char_sub = char_sub.rename(columns={beta_col: "beta_sort"})

    # Inherits get_fRet30_all universe filters after merging with lnf exposures.
    lnf = pd.read_csv(Path(result_path) / "lnf_exposure_proxies.csv")
    lnf["permno"] = lnf["permno"].astype(int)
    lnf["yyyymm"] = lnf["ym"].astype(int)
    lnf = lnf[["yyyymm", "permno", "lnf_exposure_proxy"]]

    panel = panel.merge(char_sub, on=["yyyymm", "permno"], how="inner")
    panel = panel.merge(lnf, on=["yyyymm", "permno"], how="inner")
    panel = panel.dropna(subset=["beta_sort", "lnf_exposure_proxy", "intraday_ret", "overnight_ret"])
    # lnf_exposure_proxy is already |.|-valued from compute_lnf_exposure_proxy; rank directly.
    panel["news_exposure_quantile"] = panel.groupby("date")["lnf_exposure_proxy"].transform(lambda x: x.rank(pct=True))
    return panel.reset_index(drop=True)



def _build_beta_decile_panel(panel, ff6_daily, beta_col="beta_sort", k_groups=10, eqw=True, include_mkt=False):
    """Cross-sectional β-decile sort; vw/ew aggregation per (date, beta_group); RF split
    half-half into day_ex/night_ex (HLR convention)."""
    df = panel.copy()
    df["_w"] = 1.0 if eqw else df["size_exp"]
    df["beta_group"] = df.groupby("date")[beta_col].transform(
        lambda x: pd.qcut(x, q=k_groups, labels=False, duplicates="drop") + 1
    )
    agg = (
        df.assign(
            _w_beta=df[beta_col] * df["_w"],
            _w_day=df["intraday_ret"] * df["_w"],
            _w_night=df["overnight_ret"] * df["_w"],
        )
        .groupby(["date", "beta_group"], as_index=False)
        .agg(
            beta_temp=("_w_beta", "sum"),
            day_temp=("_w_day", "sum"),
            night_temp=("_w_night", "sum"),
            denom=("_w", "sum"),
        )
    )
    agg["beta"] = agg["beta_temp"] / agg["denom"]
    agg["day"] = agg["day_temp"] / agg["denom"]
    agg["night"] = agg["night_temp"] / agg["denom"]
    agg = agg.drop(columns=["beta_temp", "day_temp", "night_temp", "denom"])

    ff_cols = ["date", "RF"] + (["MKT_RF"] if include_mkt else [])
    ff = ff6_daily[ff_cols].copy()
    ff["date"] = pd.to_numeric(ff["date"], errors="coerce").astype(int)
    ff = ff.drop_duplicates("date", keep="last")
    agg = agg.merge(ff, on="date", how="left")
    agg["day_ex"] = agg["day"] - agg["RF"] / 2
    agg["night_ex"] = agg["night"] - agg["RF"] / 2
    return agg


def get_hlr_sort_result(panel, ff6_daily, k_groups=10, eqw=True):
    res = _build_beta_decile_panel(panel, ff6_daily, beta_col="beta_sort",
                                    k_groups=k_groups, eqw=eqw, include_mkt=True)

    post_betas = {}
    for g in sorted(res["beta_group"].dropna().unique()):
        tmp = res.loc[res["beta_group"] == g, ["day", "night", "RF", "MKT_RF"]].copy()
        tmp["RET"] = (1 + tmp["day"]) * (1 + tmp["night"]) - 1
        tmp["RET_RF"] = tmp["RET"] - tmp["RF"]
        v = tmp["MKT_RF"].var()
        post_betas[g] = np.nan if (pd.isna(v) or v == 0) else tmp[["RET_RF", "MKT_RF"]].cov().iloc[0, 1] / v

    out = res.groupby("beta_group")[["day", "night", "day_ex", "night_ex", "beta"]].mean().reset_index()
    out["post_beta"] = out["beta_group"].map(post_betas)

    x = out["post_beta"].values
    y_day = out["day_ex"].values * 100
    y_night = out["night_ex"].values * 100
    day_coef = np.polyfit(x, y_day, 1) if len(out) >= 2 else [np.nan, np.nan]
    night_coef = np.polyfit(x, y_night, 1) if len(out) >= 2 else [np.nan, np.nan]
    return out, day_coef, night_coef



def plot_hlr_replication(panel, ff6_daily, eqw=True, k_groups=10, save_filename=None):
    res, day_coef, night_coef = get_hlr_sort_result(panel, ff6_daily, k_groups=k_groups, eqw=eqw)
    betas = res["post_beta"].values
    day = res["day_ex"].values * 100
    night = res["night_ex"].values * 100
    line_day = np.poly1d(day_coef)
    line_night = np.poly1d(night_coef)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(betas, day, color="tab:blue", label="Open-to-close (intraday)")
    ax.scatter(betas, night, color="tab:orange", marker="^", label="Close-to-open (overnight)")
    ax.plot(betas, line_day(betas), color="tab:blue")
    ax.plot(betas, line_night(betas), color="tab:orange")
    ax.set_xlabel("post-sorting beta")
    ax.set_ylabel(f"excess return (%, daily, {'equally weighted' if eqw else 'value weighted'})")
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(Path(plot_path) / save_filename, bbox_inches="tight", dpi=300)
    plt.show()
    return res



def plot_hlr_news_quintiles(panel, ff6_daily, eqw=True, k_groups=10, low=0.2, high=0.8, save_filename=None):
    top_panel = panel[panel["news_exposure_quantile"] >= high].copy()
    mid_panel = panel[(panel["news_exposure_quantile"] > low) & (panel["news_exposure_quantile"] < high)].copy()
    bot_panel = panel[panel["news_exposure_quantile"] <= low].copy()

    groups = [("top", top_panel), ("middle", mid_panel), ("bottom", bot_panel)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    out = {}
    for i, (news_group, panel_g) in enumerate(groups):
        res_g, day_coef, night_coef = get_hlr_sort_result(panel_g, ff6_daily, k_groups=k_groups, eqw=eqw)
        out[news_group] = res_g
        betas = res_g["post_beta"].values
        day = res_g["day_ex"].values * 100
        night = res_g["night_ex"].values * 100
        line_day = np.poly1d(day_coef)
        line_night = np.poly1d(night_coef)

        axes[i].scatter(betas, day, color="tab:blue", label="Open-to-close (intraday)")
        axes[i].scatter(betas, night, color="tab:orange", marker="^", label="Close-to-open (overnight)")
        axes[i].plot(betas, line_day(betas), color="tab:blue")
        axes[i].plot(betas, line_night(betas), color="tab:orange")
        axes[i].set_xlabel("post-sorting beta")
        axes[i].grid(False)
        if i == 0:
            axes[i].set_ylabel(f"excess return (%, daily, {'equally weighted' if eqw else 'value weighted'})")
            axes[i].legend(loc="upper left")
            axes[i].set_title("Top quintile")
        elif i == 1:
            axes[i].set_title("Middle quintiles")
        else:
            axes[i].set_title("Bottom quintile")

    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(Path(plot_path) / save_filename, bbox_inches="tight", dpi=300)
    plt.show()
    return out



def fm_reg(HLR_res, y, lags=10):
    d = HLR_res[["date", y, "beta"]].dropna().copy()
    d[y] = d[y] * 100

    def _fit(g):
        inter, slp, r2 = _ols_intercept_slope_r2(g[y].values, g["beta"].values)
        return pd.Series({"Intercept": inter, "beta": slp, "adj_R2": r2})

    fm = d.groupby("date", group_keys=False).apply(_fit).reset_index()
    se0 = smf.ols("Intercept ~ 1", fm).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).bse.iloc[0]
    se1 = smf.ols("beta ~ 1", fm).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).bse.iloc[0]

    m0, m1 = fm["Intercept"].mean(), fm["beta"].mean()
    t0, t1 = m0 / se0, m1 / se1
    r2 = fm["adj_R2"].mean()

    print(f"[{y}]  Mean(Intercept) = {m0:.3f}, t(NW) = {t0:.2f}")
    print(f"[{y}]  Mean(beta)       = {m1:.3f}, t(NW) = {t1:.2f}")
    print(f"[{y}]  Avg adj-R^2     = {r2:.4f}")
    return fm



def get_fm_results(panel, ff6_daily=None, eqw=True, beta_col="beta_sort", k_groups=10, lags=10):
    if beta_col not in panel.columns and "beta_sort" in panel.columns:
        beta_col = "beta_sort"
    if ff6_daily is None:
        ff6_daily = pd.read_csv(Path(ret_path) / "ff6_daily_returns.csv")

    HLR_res = _build_beta_decile_panel(panel, ff6_daily, beta_col=beta_col,
                                        k_groups=k_groups, eqw=eqw, include_mkt=False)
    fm_night = fm_reg(HLR_res, y="night_ex", lags=lags)
    fm_day = fm_reg(HLR_res, y="day_ex", lags=lags)
    return fm_night, fm_day, HLR_res



def run_hlr_fm_by_news_quintiles(panel, ff6_daily=None, eqw=True, beta_col="beta_sort", k_groups=10, lags=10, low=0.2, high=0.8):
    mode = "equal-weight" if eqw else "value-weight"
    top_panel = panel[panel["news_exposure_quantile"] >= high].copy()
    mid_panel = panel[(panel["news_exposure_quantile"] > low) & (panel["news_exposure_quantile"] < high)].copy()
    bot_panel = panel[panel["news_exposure_quantile"] <= low].copy()

    print("*" * 20 + f"full-sample: {mode}" + "*" * 20)
    fm_night_full, fm_day_full, _ = get_fm_results(panel, ff6_daily=ff6_daily, eqw=eqw, beta_col=beta_col, k_groups=k_groups, lags=lags)
    print("*" * 20 + f"top-news quintile: {mode}" + "*" * 20)
    fm_night_top, fm_day_top, _ = get_fm_results(top_panel, ff6_daily=ff6_daily, eqw=eqw, beta_col=beta_col, k_groups=k_groups, lags=lags)
    print("*" * 20 + f"middle-news quintiles: {mode}" + "*" * 20)
    fm_night_mid, fm_day_mid, _ = get_fm_results(mid_panel, ff6_daily=ff6_daily, eqw=eqw, beta_col=beta_col, k_groups=k_groups, lags=lags)
    print("*" * 20 + f"bottom-news quintile: {mode}" + "*" * 20)
    fm_night_bot, fm_day_bot, _ = get_fm_results(bot_panel, ff6_daily=ff6_daily, eqw=eqw, beta_col=beta_col, k_groups=k_groups, lags=lags)

    return {
        "full": {"night": fm_night_full, "day": fm_day_full},
        "top": {"night": fm_night_top, "day": fm_day_top},
        "middle": {"night": fm_night_mid, "day": fm_day_mid},
        "bottom": {"night": fm_night_bot, "day": fm_day_bot},
    }



def print_latex_hlr_fm_table(fm_ew, fm_vw, lags=10, latex_filename="hlr_fm_table.tex"):
    """LaTeX table for HLR Fama-MacBeth (EW vs VW). fm_ew/fm_vw: outputs of
    run_hlr_fm_by_news_quintiles. lags: NW lag for note line."""
    block_order = ["full", "top", "middle", "bottom"]
    block_labels = {
        "full": "Unconditional",
        "top": r"Top $|\hat{\gamma}_{i,t}|$ quintile",
        "middle": r"Middle $|\hat{\gamma}_{i,t}|$ quintiles",
        "bottom": r"Bottom $|\hat{\gamma}_{i,t}|$ quintile",
    }
    row_order = ["night", "day"]
    row_labels = {"night": "Night", "day": "Day"}

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabularx}{0.95\textwidth}{l *{6}{>{\centering\arraybackslash}X}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{Equally-weighted} & \multicolumn{3}{c}{Value-weighted} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"& Intercept & Beta & adj.\ $R^2$ & Intercept & Beta & adj.\ $R^2$ \\")
    lines.append(r"\midrule")

    for bi, bucket in enumerate(block_order):
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{block_labels[bucket]}}}}} \\")
        for part in row_order:
            se = _get_stats(fm_ew, bucket, lags=lags, part=part)
            sv = _get_stats(fm_vw, bucket, lags=lags, part=part)
            lines.append(
                f"{row_labels[part]} & "
                f"{_fmt_num_plain(se['intercept'])}{_sig_stars_latex(se['t_intercept'])} & "
                f"{_fmt_num_plain(se['beta'])}{_sig_stars_latex(se['t_beta'])} & "
                f"{_fmt_num_plain(se['r2'])} & "
                f"{_fmt_num_plain(sv['intercept'])}{_sig_stars_latex(sv['t_intercept'])} & "
                f"{_fmt_num_plain(sv['beta'])}{_sig_stars_latex(sv['t_beta'])} & "
                f"{_fmt_num_plain(sv['r2'])} \\\\"
            )
            lines.append(
                f"& ({_fmt_num_plain(se['t_intercept'])}) & ({_fmt_num_plain(se['t_beta'])}) & "
                f"& ({_fmt_num_plain(sv['t_intercept'])}) & ({_fmt_num_plain(sv['t_beta'])}) & \\\\"
            )
        if bi < len(block_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(rf"\multicolumn{{7}}{{l}}{{\footnotesize Fama-MacBeth regression: t-statistics (Newey-West corrections, lags = {lags}) in parentheses.}} \\")
    lines.append(r"\multicolumn{7}{r}{\footnotesize $^*p<0.1;\ ^{**}p<0.05;\ ^{***}p<0.01$}")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if latex_filename is not None:
        (Path(result_path) / latex_filename).write_text(tex)



def build_sml_panel(beta_col="Beta252d"):
    data = pd.read_csv(Path(ret_path) / "ret_daily_data.csv").drop(columns=["SPY"])
    data = pd.melt(data, id_vars=["date"], var_name="permno", value_name="ret")
    data["permno"] = data["permno"].astype(int)
    data["date"] = data["date"].astype(int)
    data["yyyymm"] = data["date"] // 100

    char = pd.read_csv(Path(data_path) / "monthly_lagged_betas.csv", low_memory=False)
    char["permno"] = char["permno"].astype(int)
    char["yyyymm"] = char["yyyymm"].astype(int)
    char["size_exp"] = np.exp(pd.to_numeric(char["size_wrds"], errors="coerce"))
    char = char.rename(columns={"beta_252d": "Beta252d"})
    # beta_sort kept JKP-lagged: HLR/SML sort at month-start on prior-month β, evaluate intra-month.
    char_sub = char[["yyyymm", "permno", "size_exp", beta_col]].copy()
    char_sub = char_sub.rename(columns={beta_col: "beta_sort"})

    # Inherits get_fRet30_all universe filters after merging with lnf exposures.
    lnf = pd.read_csv(Path(result_path) / "lnf_exposure_proxies.csv")
    lnf["permno"] = lnf["permno"].astype(int)
    lnf["yyyymm"] = lnf["ym"].astype(int)
    lnf = lnf[["yyyymm", "permno", "lnf_exposure_proxy"]]

    data = data.merge(char_sub, on=["yyyymm", "permno"], how="inner")
    data = data.merge(lnf, on=["yyyymm", "permno"], how="inner")
    data = data.dropna(subset=["ret", "beta_sort", "lnf_exposure_proxy"])
    # lnf_exposure_proxy is already |.|-valued from compute_lnf_exposure_proxy; rank directly.
    data["news_exposure_quantile"] = data.groupby("date")["lnf_exposure_proxy"].transform(lambda x: x.rank(pct=True))
    return data.reset_index(drop=True)



def get_sml_sort_result(panel, ff6_daily, eqw=True, k_groups=10, beta_col="beta_sort"):
    df = panel.copy()
    df["weight"] = 1.0 if eqw else df["size_exp"]
    df["beta_groups"] = df.groupby("date")[beta_col].transform(
        lambda x: pd.qcut(x, q=k_groups, labels=False, duplicates="drop") + 1
    )

    port = (
        df.assign(w_beta=df[beta_col] * df["weight"], w_ret=df["ret"] * df["weight"])
        .groupby(["date", "beta_groups"], as_index=False)
        .agg(beta_temp=("w_beta", "sum"), ret_temp=("w_ret", "sum"), denom=("weight", "sum"))
    )
    port["beta"] = port["beta_temp"] / port["denom"]
    port["ret"] = port["ret_temp"] / port["denom"]
    port = port.drop(columns=["beta_temp", "ret_temp", "denom"])

    ff = ff6_daily[["date", "RF", "MKT_RF"]].copy()
    ff["date"] = ff["date"].astype(int)
    port = port.merge(ff, on="date", how="left")
    port["ret_ex"] = port["ret"] - port["RF"]
    port_res = port.groupby("beta_groups")[["beta", "ret", "ret_ex"]].mean().reset_index()
    post_betas = {}
    for g in sorted(port_res["beta_groups"].dropna().unique()):
        tmp = port[port["beta_groups"] == g]
        v = tmp["MKT_RF"].var()
        post_betas[g] = np.nan if (pd.isna(v) or v == 0) else tmp[["ret_ex", "MKT_RF"]].cov().iloc[0, 1] / v
    port_res["post_beta"] = port_res["beta_groups"].map(post_betas)

    betas = port_res["post_beta"].values
    rets = port_res["ret_ex"].values * 100
    sml = np.poly1d(np.polyfit(betas, rets, 1)) if len(port_res) >= 2 else np.poly1d([np.nan, np.nan])
    return port_res, betas, rets, sml, port



def plot_sml_conditional(panel, ff6_daily, eqw=True, k_groups=10, beta_col="beta_sort",
                         low=0.2, high=0.8, quantile_col="news_exposure_quantile",
                         condition_label=None, save_filename=None):
    """Plot SML unconditional + conditional on top/middle/bottom of `quantile_col`.
    `quantile_col` default conditions on news exposure; pass e.g. "char_quantile" for
    a JKP-characteristic ranking. `condition_label` is appended to legend entries."""
    _, betas, rets, sml, _ = get_sml_sort_result(panel, ff6_daily, eqw=eqw, k_groups=k_groups, beta_col=beta_col)
    top = panel[panel[quantile_col] >= high].copy()
    mid = panel[(panel[quantile_col] > low) & (panel[quantile_col] < high)].copy()
    bot = panel[panel[quantile_col] <= low].copy()
    label_suffix = "" if condition_label is None else f" ({condition_label})"

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(betas, rets, color="tab:blue", marker="^", label="Unconditional")
    ax.plot(betas, sml(betas), color="tab:blue")

    colors = ["tab:orange", "tab:green", "tab:red"]
    groups = [("top", top), ("middle", mid), ("bottom", bot)]
    for i, (gname, gdf) in enumerate(groups):
        _, b, r, s, _ = get_sml_sort_result(gdf, ff6_daily, eqw=eqw, k_groups=k_groups, beta_col=beta_col)
        suffix = "s" if gname == "middle" else ""
        ax.scatter(b, r, color=colors[i], label=gname.capitalize() + " quintile" + suffix + label_suffix)
        ax.plot(b, s(b), color=colors[i])

    ax.legend(loc="best")
    ax.set_xlabel("post-sorting beta")
    ax.set_ylabel(f"excess return (%, daily, {'equally weighted' if eqw else 'value weighted'})")
    ax.grid(False)
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(Path(plot_path) / save_filename, bbox_inches="tight", dpi=300)
    plt.show()



def plot_sml_days(panel, days_df, ff6_daily, eqw=True, k_groups=10, beta_col="beta_sort", save_filename=None):
    panel = panel.merge(days_df, on="date", how="left")
    day_cols = [c for c in days_df.columns if c != "date"]
    panel[day_cols] = panel[day_cols].fillna(0).astype(int)

    label_map = {
        "lead_day": "LEAD",
        "fomc_day": "FOMC",
        "macro_day": "Macro",
    }
    colors = {
        "LEAD": "tab:orange",
        "FOMC": "tab:red",
        "Macro": "tab:green",
        "Other": "tab:blue",
    }
    plot_order = ["lead_day", "fomc_day", "macro_day"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for col in plot_order:
        label = label_map[col]
        sub = panel.loc[panel[col].eq(1)].copy()
        if sub.empty:
            continue
        _, betas, rets, sml, _ = get_sml_sort_result(
            sub,
            ff6_daily,
            eqw=eqw,
            k_groups=k_groups,
            beta_col=beta_col,
        )
        ax.scatter(betas, rets, color=colors[label], label=label)
        ax.plot(betas, sml(betas), color=colors[label])

    other = panel.loc[panel[day_cols].sum(axis=1).eq(0)].copy()
    if not other.empty:
        _, betas, rets, sml, _ = get_sml_sort_result(
            other,
            ff6_daily,
            eqw=eqw,
            k_groups=k_groups,
            beta_col=beta_col,
        )
        ax.scatter(betas, rets, color=colors["Other"], label="Other")
        ax.plot(betas, sml(betas), color=colors["Other"])

    ax.legend(loc="upper left")
    ax.set_xlabel("post-sorting beta")
    ax.set_ylabel(f"excess return (%, daily, {'equally weighted' if eqw else 'value weighted'})")
    ax.grid(False)
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(Path(plot_path) / save_filename, bbox_inches="tight", dpi=300)
    plt.show()






def get_sml_fm_results(panel, ff6_daily=None, eqw=True, beta_col="beta_sort", k_groups=10, lags=10):
    df = panel.copy()
    if ff6_daily is None:
        ff6_daily = pd.read_csv(Path(ret_path) / "ff6_daily_returns.csv")
    df["weight"] = 1.0 if eqw else df["size_exp"]
    df["beta_groups"] = df.groupby("date")[beta_col].transform(
        lambda x: pd.qcut(x, q=k_groups, labels=False, duplicates="drop") + 1
    )
    sml_res = (
        df.assign(w_beta=df[beta_col] * df["weight"], w_ret=df["ret"] * df["weight"])
        .groupby(["date", "beta_groups"], as_index=False)
        .agg(beta_temp=("w_beta", "sum"), ret_temp=("w_ret", "sum"), denom=("weight", "sum"))
    )
    sml_res["beta"] = sml_res["beta_temp"] / sml_res["denom"]
    sml_res["ret"] = sml_res["ret_temp"] / sml_res["denom"]
    sml_res = sml_res.drop(columns=["beta_temp", "ret_temp", "denom"])
    rf_daily = ff6_daily[["date", "RF"]].copy()
    rf_daily["date"] = pd.to_numeric(rf_daily["date"], errors="coerce").astype(int)
    rf_daily = rf_daily.drop_duplicates("date", keep="last")
    sml_res = sml_res.merge(rf_daily, on="date", how="left")
    sml_res["ret_ex"] = sml_res["ret"] - sml_res["RF"]
    fm = fm_reg(sml_res, y="ret_ex", lags=lags)
    return fm, sml_res



def run_sml_fm_by_quintiles(panel, ff6_daily=None, eqw=True, beta_col="beta_sort",
                            k_groups=10, lags=10, low=0.2, high=0.8,
                            quantile_col="news_exposure_quantile", condition_label="news"):
    """Run SML Fama-MacBeth on full sample + top/middle/bottom buckets of `quantile_col`.
    Default conditions on news exposure (LNF replication); pass `quantile_col`/`condition_label`
    to condition on any other quantile column (e.g. a JKP characteristic ranking)."""
    mode = "equal-weight" if eqw else "value-weight"
    top = panel[panel[quantile_col] >= high].copy()
    mid = panel[(panel[quantile_col] > low) & (panel[quantile_col] < high)].copy()
    bot = panel[panel[quantile_col] <= low].copy()

    print("*" * 20 + f"full-sample: {mode}" + "*" * 20)
    fm_full, _ = get_sml_fm_results(panel, ff6_daily=ff6_daily, eqw=eqw, beta_col=beta_col, k_groups=k_groups, lags=lags)
    print("*" * 20 + f"top {condition_label} quintile: {mode}" + "*" * 20)
    fm_top, _ = get_sml_fm_results(top, ff6_daily=ff6_daily, eqw=eqw, beta_col=beta_col, k_groups=k_groups, lags=lags)
    print("*" * 20 + f"middle {condition_label} quintiles: {mode}" + "*" * 20)
    fm_mid, _ = get_sml_fm_results(mid, ff6_daily=ff6_daily, eqw=eqw, beta_col=beta_col, k_groups=k_groups, lags=lags)
    print("*" * 20 + f"bottom {condition_label} quintile: {mode}" + "*" * 20)
    fm_bot, _ = get_sml_fm_results(bot, ff6_daily=ff6_daily, eqw=eqw, beta_col=beta_col, k_groups=k_groups, lags=lags)

    return {"full": fm_full, "top": fm_top, "middle": fm_mid, "bottom": fm_bot}



def print_latex_sml_fm_table(
    fm_sml_ew,
    fm_sml_vw,
    lags=10,
    latex_filename="sml_fm_ew_vw.tex",
):
    """Build and save LaTeX table for SML Fama-MacBeth results (EW vs VW)."""
    block_order = ["full", "top", "middle", "bottom"]
    block_labels = {
        "full": "Unconditional",
        "top": r"Top $|\hat{\gamma}_{i,t}|$ quintile",
        "middle": r"Middle $|\hat{\gamma}_{i,t}|$ quintiles",
        "bottom": r"Bottom $|\hat{\gamma}_{i,t}|$ quintile",
    }

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\caption{\textbf{Fama-MacBeth regression:}}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabularx}{0.95\textwidth}{l *{6}{>{\centering\arraybackslash}X}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{Equally-weighted} & \multicolumn{3}{c}{Value-weighted} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"Conditional & Intercept & Beta & adj.\ $R^2$ & Intercept & Beta & adj.\ $R^2$ \\")
    lines.append(r"\midrule")

    for bi, bucket in enumerate(block_order):
        se = _get_stats(fm_sml_ew, bucket, lags=lags)
        sv = _get_stats(fm_sml_vw, bucket, lags=lags)
        lines.append(
            f"{block_labels[bucket]} & {_fmt_num_plain(se['intercept'])}{_sig_stars_latex(se['t_intercept'])} & {_fmt_num_plain(se['beta'])}{_sig_stars_latex(se['t_beta'])} & {_fmt_num_plain(se['r2'])} & "
            f"{_fmt_num_plain(sv['intercept'])}{_sig_stars_latex(sv['t_intercept'])} & {_fmt_num_plain(sv['beta'])}{_sig_stars_latex(sv['t_beta'])} & {_fmt_num_plain(sv['r2'])} \\\\"
        )
        lines.append(
            f"& ({_fmt_num_plain(se['t_intercept'])}) & ({_fmt_num_plain(se['t_beta'])}) & "
            f"& ({_fmt_num_plain(sv['t_intercept'])}) & ({_fmt_num_plain(sv['t_beta'])}) & \\\\"
        )
        if bi < len(block_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(rf"\multicolumn{{7}}{{l}}{{\footnotesize Fama-MacBeth regression: t-statistics (Newey-West corrections, lags = {lags}) in parentheses.}} \\")
    lines.append(r"\multicolumn{7}{r}{\footnotesize $^*p<0.1;\ ^{**}p<0.05;\ ^{***}p<0.01$}")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    (Path(result_path) / latex_filename).write_text("\n".join(lines))



def get_beta_shock_panel():
    # Inherits get_fRet30_all universe filters after merging with lnf exposures.
    lnf = pd.read_csv(Path(result_path) / "lnf_exposure_proxies.csv")
    char = pd.read_csv(Path(data_path) / "monthly_lagged_betas.csv", low_memory=False)
    crsp_taq = pd.read_csv(Path(data_path) / "crsp_taq_monthly_info.csv", low_memory=False)

    lnf = lnf[["ym", "permno", "lnf_exposure_proxy"]].copy()
    lnf["ym"] = lnf["ym"].astype(int)
    lnf["permno"] = lnf["permno"].astype(int)
    lnf = lnf.rename(columns={"ym": "yyyymm"})

    char["permno"] = char["permno"].astype(int)
    char["yyyymm"] = char["yyyymm"].astype(int)
    char["size_exp"] = np.exp(pd.to_numeric(char["size_wrds"], errors="coerce"))
    char["Beta252d_lag"] = pd.to_numeric(char["beta_252d"], errors="coerce")
    char = char.sort_values(["permno", "yyyymm"]).copy()
    # undo JKP lag → current-month β; need both β_t and β_{t-1} to form shock = β_t − β_{t-1}.
    char["Beta252d_t"] = char.groupby("permno")["Beta252d_lag"].shift(-1)
    char["Beta252d_shock"] = char["Beta252d_t"] - char["Beta252d_lag"]

    crsp_taq["ym"] = pd.to_numeric(crsp_taq["ym"], errors="coerce")
    crsp_taq["permno"] = pd.to_numeric(crsp_taq["permno"], errors="coerce")
    crsp_taq = crsp_taq.dropna(subset=["ym", "permno"]).copy()
    crsp_taq["ym"] = crsp_taq["ym"].astype(int)
    crsp_taq["permno"] = crsp_taq["permno"].astype(int)
    if "naics" in crsp_taq.columns:
        crsp_taq["naics3"] = (
            crsp_taq["naics"].astype(str).str.replace(r"\D", "", regex=True).str[:3]
        )
        crsp_taq.loc[crsp_taq["naics3"].str.len() < 3, "naics3"] = np.nan
    else:
        crsp_taq["naics3"] = np.nan
    crsp_taq = crsp_taq[["ym", "permno", "naics3"]].drop_duplicates(["ym", "permno"], keep="last")
    crsp_taq = crsp_taq.rename(columns={"ym": "yyyymm"})

    panel = lnf.merge(
        char[["yyyymm", "permno", "size_exp", "Beta252d_lag", "Beta252d_t", "Beta252d_shock"]],
        on=["yyyymm", "permno"],
        how="inner",
    )
    panel = panel.merge(crsp_taq, on=["yyyymm", "permno"], how="left")
    panel = panel.dropna(subset=["lnf_exposure_proxy", "size_exp", "Beta252d_shock"]).copy()
    panel = panel[panel["size_exp"] > 0].copy()
    # lnf_exposure_proxy is already |.|-valued from compute_lnf_exposure_proxy; alias for downstream clarity.
    panel["abs_lnf"] = panel["lnf_exposure_proxy"]
    return panel.reset_index(drop=True)



def plot_beta_shock_monthly_groups(panel, k_group=10, value_weighted=False):
    q_hi = 1.0 - 1.0 / float(k_group)
    q_lo = 1.0 / float(k_group)

    tmp = panel[["yyyymm", "abs_lnf", "Beta252d_shock", "size_exp", "naics3"]].copy()
    tmp["q_hi"] = tmp.groupby("yyyymm")["abs_lnf"].transform(lambda x: x.quantile(q_hi))
    tmp["q_lo"] = tmp.groupby("yyyymm")["abs_lnf"].transform(lambda x: x.quantile(q_lo))
    tmp["grp"] = "other"
    tmp.loc[tmp["abs_lnf"] >= tmp["q_hi"], "grp"] = "high"
    tmp.loc[tmp["abs_lnf"] <= tmp["q_lo"], "grp"] = "low"

    if value_weighted:
        def _w_stats(g):
            return _cluster_mean_stats(g["Beta252d_shock"], g["naics3"], w=g["size_exp"])

        ts_stat = tmp.groupby(["yyyymm", "grp"]).apply(_w_stats, include_groups=False).reset_index()
    else:
        def _uw_stats(g):
            return _cluster_mean_stats(g["Beta252d_shock"], g["naics3"], w=None)

        ts_stat = tmp.groupby(["yyyymm", "grp"]).apply(_uw_stats, include_groups=False).reset_index()

    ts_mean = ts_stat.pivot(index="yyyymm", columns="grp", values="mean").sort_index()
    ts_ci_lo = ts_stat.pivot(index="yyyymm", columns="grp", values="ci_lo").sort_index()
    ts_ci_hi = ts_stat.pivot(index="yyyymm", columns="grp", values="ci_hi").sort_index()
    ts_plot = ts_mean.reset_index()
    ts_plot["dt"] = pd.to_datetime(ts_plot["yyyymm"].astype(str), format="%Y%m")

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    style = {"high": "tab:blue", "low": "tab:orange", "other": "tab:green"}
    labels = {"high": "High", "low": "Low", "other": "other"}
    for col in ["high", "low", "other"]:
        if col not in ts_mean.columns:
            continue
        x = ts_plot["dt"].values
        y = ts_mean[col].values.astype(float)
        ci_lo_vals = ts_ci_lo[col].values.astype(float)
        ci_hi_vals = ts_ci_hi[col].values.astype(float)
        ax.plot(x, y, lw=1.4, color=style[col], label=labels[col])
        ax.fill_between(x, ci_lo_vals, ci_hi_vals, color=style[col], alpha=0.25)
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    wt_tag = "vw" if value_weighted else "ew"
    plt.savefig(Path(plot_path) / f"beta_shock_monthly_{wt_tag}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    return ts_stat



def plot_beta_shock_event_decay(panel, k_group=10, value_weighted=False, hac_lags=None):
    q_hi = 1.0 - 1.0 / float(k_group)
    q_lo = 1.0 / float(k_group)

    base = panel[["permno", "yyyymm", "abs_lnf", "size_exp"]].drop_duplicates().copy()
    base["q_hi"] = base.groupby("yyyymm")["abs_lnf"].transform(lambda x: x.quantile(q_hi))
    base["q_lo"] = base.groupby("yyyymm")["abs_lnf"].transform(lambda x: x.quantile(q_lo))
    base["grp"] = "other"
    base.loc[base["abs_lnf"] >= base["q_hi"], "grp"] = "high"
    base.loc[base["abs_lnf"] <= base["q_lo"], "grp"] = "low"
    base["ym_period"] = pd.PeriodIndex(base["yyyymm"].astype(str), freq="M")

    shock = panel[["permno", "yyyymm", "Beta252d_shock"]].drop_duplicates().copy()
    shock["ym_period"] = pd.PeriodIndex(shock["yyyymm"].astype(str), freq="M")
    shock = shock[["permno", "ym_period", "Beta252d_shock"]].rename(
        columns={"ym_period": "ym_fut", "Beta252d_shock": "Beta252d_shock_fut"}
    )

    rows = []
    for h in range(13):
        tmp = base.copy()
        tmp["ym_fut"] = tmp["ym_period"] + h
        m = tmp.merge(shock, on=["permno", "ym_fut"], how="left")
        use_lags = h if hac_lags is None else int(hac_lags)
        for g in ["high", "low", "other"]:
            dg = m[m["grp"] == g].copy()
            if value_weighted:
                cohort = (
                    dg.groupby("yyyymm")
                    .apply(
                        lambda z: _weighted_mean(
                            z["Beta252d_shock_fut"].to_numpy(dtype=float),
                            z["size_exp"].to_numpy(dtype=float),
                        ),
                        include_groups=False,
                    )
                    .rename("cohort_mean")
                    .reset_index()
                )
            else:
                cohort = (
                    dg.groupby("yyyymm")["Beta252d_shock_fut"]
                    .mean()
                    .rename("cohort_mean")
                    .reset_index()
                )
            mean, se, ci_lo, ci_hi, n_months = _nw_mean_ci(cohort["cohort_mean"], lags=use_lags)
            rows.append({
                "h": h,
                "grp": g,
                "mean": mean,
                "se": se,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_months": n_months,
                "hac_lags": use_lags,
            })

    decay = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    labels = {"high": "High", "low": "Low", "other": "other"}
    colors = {"high": "tab:blue", "low": "tab:orange", "other": "tab:green"}
    for g in ["high", "low", "other"]:
        dg = decay[decay["grp"] == g].sort_values("h")
        x = dg["h"].values.astype(float)
        y = dg["mean"].values.astype(float)
        ci_lo_vals = dg["ci_lo"].values.astype(float)
        ci_hi_vals = dg["ci_hi"].values.astype(float)
        ax.plot(x, y, marker="o", lw=1.4, color=colors[g], label=labels[g])
        ax.fill_between(x, ci_lo_vals, ci_hi_vals, color=colors[g], alpha=0.25)
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax.set_xlabel("Months after signal (h)")
    ax.set_ylabel("")
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    wt_tag = "vw" if value_weighted else "ew"
    plt.savefig(Path(plot_path) / f"beta_shock_decay_{wt_tag}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    return decay

_SML_EXTRA_CHAR_PATH = Path(data_path) / "monthly_more_jkp_chars_lagged.csv"



def _filter_sml_panel_non_special_days(sml_panel):
    announcement_dates = pd.read_csv(Path(data_path) / "announcement_dates.csv")
    announcement_dates["date"] = pd.to_datetime(announcement_dates["date"]).dt.strftime("%Y%m%d").astype(int)
    announcement_dates["macro_day"] = announcement_dates[["cpi_day", "ppi_day", "unrate_day"]].max(axis=1).astype(int)
    # fomc_day is scheduled-only upstream (data.get_announcement_dates).
    days_df = announcement_dates[["date", "lead_day", "macro_day", "fomc_day"]].copy()
    days_df[["lead_day", "macro_day", "fomc_day"]] = days_df[["lead_day", "macro_day", "fomc_day"]].astype(int)

    # Defensive: assert announcement_dates.csv covers every date in sml_panel.
    # An inner merge would silently drop uncovered dates (treating them as "special"
    # by exclusion); fail loudly instead so stale announcement files are surfaced.
    panel_dates = set(sml_panel["date"].unique())
    covered_dates = set(days_df["date"].unique())
    missing = panel_dates - covered_dates
    if missing:
        raise ValueError(
            f"announcement_dates.csv missing {len(missing)} dates present in sml_panel "
            f"(e.g., {sorted(missing)[:5]}); regenerate the file via "
            f"data.get_announcement_dates."
        )

    non_special_days = days_df.loc[days_df[["lead_day", "macro_day", "fomc_day"]].sum(axis=1).eq(0), ["date"]]
    return sml_panel.merge(non_special_days, on="date", how="inner")



def get_sml_panel_extra_char(sml_panel, char_col=None, non_special_only=False, rank_by_abs=False):
    """Prepare the SML panel with a JKP-characteristic-based daily ranking (`char_quantile`).
    Use as input to plot_sml_conditional / run_sml_fm_by_quintiles with quantile_col='char_quantile'."""
    print(
        f"Original build_sml_panel sample | months: {sml_panel['yyyymm'].nunique():,}, "
        f"permnos: {sml_panel['permno'].nunique():,}, rows: {len(sml_panel):,}"
    )

    if non_special_only:
        sml_panel_use = _filter_sml_panel_non_special_days(sml_panel)
    else:
        sml_panel_use = sml_panel.copy()

    if char_col is None:
        print(
            f"LNF replication sample | months: {sml_panel_use['yyyymm'].nunique():,}, "
            f"permnos: {sml_panel_use['permno'].nunique():,}, rows: {len(sml_panel_use):,}"
        )
        return sml_panel_use.reset_index(drop=True)

    char_names = pd.read_csv(_SML_EXTRA_CHAR_PATH, nrows=0).columns.tolist()
    char_names = [c for c in char_names if c not in {"permno", "yyyymm"}]
    if char_col not in char_names:
        raise ValueError(
            f"Unknown JKP characteristic: {char_col}. "
            + "Available chars: "
            + ", ".join(sorted(char_names))
        )

    extra_char = pd.read_csv(_SML_EXTRA_CHAR_PATH, usecols=["permno", "yyyymm", char_col])
    extra_char["permno"] = extra_char["permno"].astype(int)
    extra_char["yyyymm"] = extra_char["yyyymm"].astype(int)
    extra_char = extra_char.sort_values(["permno", "yyyymm"]).drop_duplicates(["permno", "yyyymm"], keep="last")
    raw_char_min_yyyymm = int(extra_char["yyyymm"].min())
    raw_char_max_yyyymm = int(extra_char["yyyymm"].max())
    # undo JKP lag → char[ym] = end-of-ym (ranking variable for forward-looking sort).
    extra_char[char_col] = extra_char.groupby("permno")[char_col].shift(-1)
    current_col = f"{char_col}_current"
    extra_char = extra_char.rename(columns={char_col: current_col})
    current_months = extra_char.loc[extra_char[current_col].notna(), "yyyymm"]
    current_char_min_yyyymm = int(current_months.min())
    current_char_max_yyyymm = int(current_months.max())

    sml_panel_use = sml_panel_use[sml_panel_use["yyyymm"].between(current_char_min_yyyymm, current_char_max_yyyymm)].copy()

    sml_panel_extra_char = sml_panel_use.merge(
        extra_char,
        on=["permno", "yyyymm"],
        how="left",
        validate="many_to_one",
    )

    missing_before = sml_panel_extra_char[current_col].isna()
    monthly_char = (
        sml_panel_extra_char[["permno", "yyyymm", current_col]]
        .drop_duplicates(["permno", "yyyymm"], keep="last")
        .sort_values(["permno", "yyyymm"])
    )
    monthly_char[current_col] = monthly_char.groupby("permno")[current_col].ffill(limit=2)
    sml_panel_extra_char = sml_panel_extra_char.drop(columns=[current_col]).merge(
        monthly_char,
        on=["permno", "yyyymm"],
        how="left",
        validate="many_to_one",
    )

    missing_after = sml_panel_extra_char[current_col].isna()

    rank_values = sml_panel_extra_char[current_col].abs() if rank_by_abs else sml_panel_extra_char[current_col]
    sml_panel_extra_char["char_quantile"] = rank_values.groupby(sml_panel_extra_char["date"]).transform(
        lambda x: x.rank(pct=True)
    )
    # Keep unresolved missing values in the middle group.
    sml_panel_extra_char["char_quantile"] = sml_panel_extra_char["char_quantile"].fillna(0.5)

    summary_row = {
        "conditioning": "char",
        "char_col": char_col,
        "current_col": current_col,
        "quantile_col": "char_quantile",
        "non_special_only": non_special_only,
        "rank_by_abs": rank_by_abs,
        "raw_char_min_yyyymm": raw_char_min_yyyymm,
        "raw_char_max_yyyymm": raw_char_max_yyyymm,
        "current_char_min_yyyymm": current_char_min_yyyymm,
        "current_char_max_yyyymm": current_char_max_yyyymm,
        "n_months": int(sml_panel_extra_char["yyyymm"].nunique()),
        "n_permnos": int(sml_panel_extra_char["permno"].nunique()),
        "n_rows": len(sml_panel_extra_char),
        "missing_before_fill": int(missing_before.sum()),
        "missing_share_before_fill": float(missing_before.mean()),
        "missing_after_fill": int(missing_after.sum()),
        "missing_share_after_fill": float(missing_after.mean()),
        "usable_sample_share": float((~missing_after).mean()),
    }
    print(
        f"Extra-char sample | months: {summary_row['n_months']:,}, "
        f"permnos: {summary_row['n_permnos']:,}, rows: {summary_row['n_rows']:,}"
    )
    print(
        f"JKP lagged-char window | months: {summary_row['raw_char_min_yyyymm']} to "
        f"{summary_row['raw_char_max_yyyymm']}"
    )
    print(
        f"Current-char analysis window | months: {summary_row['current_char_min_yyyymm']} to "
        f"{summary_row['current_char_max_yyyymm']}"
    )
    print(
        f"Missing before impute: {summary_row['missing_before_fill']:,} "
        f"({summary_row['missing_share_before_fill']:.2%})"
    )
    print(
        f"Missing after impute:  {summary_row['missing_after_fill']:,} "
        f"({summary_row['missing_share_after_fill']:.2%})"
    )
    return sml_panel_extra_char.reset_index(drop=True)
