import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

from constants import (
    data_path, fret30min_ym_path, ret_path, result_path,
    MIN_MKTCAP_M, MIN_CLOSE_PRICE, N_BARS_30,
)


def _ann_stats(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return {"NMonths": 0, "AnnRet(%)": np.nan, "AnnVol(%)": np.nan, "SR": np.nan}
    ann = x.mean() * 12 * 100
    vol = x.std() * (12 ** 0.5) * 100
    sr = ann / vol if (pd.notna(vol) and vol > 0) else np.nan
    return {"NMonths": n, "AnnRet(%)": ann, "AnnVol(%)": vol, "SR": sr}


def _ann_stats_daily(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return {"NDays": 0, "AnnRet(%)": np.nan, "AnnVol(%)": np.nan, "SR": np.nan}
    ann = x.mean() * 252 * 100
    vol = x.std() * np.sqrt(252) * 100
    sr = ann / vol if (pd.notna(vol) and vol > 0) else np.nan
    return {"NDays": n, "AnnRet(%)": ann, "AnnVol(%)": vol, "SR": sr}


def _fmt_pct(v, signed=True):
    if pd.isna(v):
        return "   nan"
    return f"{v:+.2f}%" if signed else f"{v:.2f}%"


def _fmt_num(v):
    if pd.isna(v):
        return "  nan"
    return f"{v:.2f}"


def _fmt_pct_tex(v, signed=True):
    """LaTeX-cell variant of _fmt_pct: strips trailing % and leading + for table cells."""
    return _fmt_pct(v, signed=signed).strip().replace("%", "").lstrip("+")


def _fmt_num_tex(v):
    """LaTeX-cell variant of _fmt_num: strips whitespace for table cells."""
    return _fmt_num(v).strip()


def _sig_stars_latex(t):
    t = pd.to_numeric(t, errors="coerce")
    if pd.isna(t):
        return ""
    at = abs(float(t))
    if at >= 2.576:
        return r"$^{***}$"
    if at >= 1.96:
        return r"$^{**}$"
    if at >= 1.645:
        return r"$^{*}$"
    return ""


def _esc_tex(s):
    s = str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _next_ym(ym):
    """Next YYYYMM int."""
    y, m = ym // 100, ym % 100
    return (y * 100 + m + 1) if m < 12 else ((y + 1) * 100 + 1)


def _get_prev_news_init(ym):
    """Use the latest earlier news_proxy cross-section when this month has no local factors."""
    pca = pd.read_csv(Path(result_path) / "monthly_pcas.csv", usecols=["YM", "permno", "news_proxy"])
    pca = pca[(pca["YM"] < int(ym)) & pca["news_proxy"].notna()]
    if pca.empty:
        raise ValueError(f"{ym}: this month has no local factors, and no earlier month is available for initialization.")

    prev_ym = int(pca["YM"].max())
    prev_news = pca.loc[pca["YM"] == prev_ym, ["permno", "news_proxy"]].copy()
    prev_news["permno"] = prev_news["permno"].astype(int)
    return prev_news.drop_duplicates(["permno"], keep="last")


def _fmt_pct_from_dec(v):
    v = pd.to_numeric(v, errors="coerce")
    if pd.isna(v):
        return ""
    return _fmt_pct(float(v) * 100, signed=False).replace("%", "").strip()


def _fmt_num_plain(v):
    v = pd.to_numeric(v, errors="coerce")
    return "" if pd.isna(v) else _fmt_num(float(v)).strip()


def _nw_mean_t(s, lags):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan
    md = smf.ols("x ~ 1", data=pd.DataFrame({"x": s})).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    m = float(s.mean())
    se = float(md.bse.iloc[0])
    t = np.nan if (pd.isna(se) or se == 0) else m / se
    return m, t


def _get_stats(fm_pack, bucket, lags, part=None):
    d = fm_pack[bucket][part] if part is not None else fm_pack[bucket]
    m_int, t_int = _nw_mean_t(d["Intercept"], lags=lags)
    m_beta, t_beta = _nw_mean_t(d["beta"], lags=lags)
    r2 = pd.to_numeric(d["adj_R2"], errors="coerce").mean()
    return {"intercept": m_int, "t_intercept": t_int, "beta": m_beta, "t_beta": t_beta, "r2": r2}


def _fmt_triplet(v):
    a, b, c = v
    fa = f"{a:.2f}%" if pd.notna(a) else "nan"
    fb = f"{b:.2f}%" if pd.notna(b) else "nan"
    fc = f"{c:.2f}" if pd.notna(c) else "nan"
    return f"{fa:>8}  {fb:>7} {fc:>5}"


def get_fRet30_all(crsp_taq, start_year=2006, end_year=2025, max_zero_ratio=0.1):
    """Load monthly 30-min intraday returns.

    Returns ym_data: dict {ym: {'fRet30', 'vwMKT30', 'vwMKT_wt'}}.
    Quality filters: missing-count <= N_BARS_30, zero-ratio <= max_zero_ratio.
    Liquidity filters (crsp_taq): lag_mktcap > MIN_MKTCAP_M, lag_close > MIN_CLOSE_PRICE.
    Universe filter: BetaFP non-NaN. Enforced here (not just in the BetaFP-decile
    benchmark) so vwMKT, Rg/MKTg/LNF, and BetaFP_D1/D10 share the same monthly
    stock universe; cross-portfolio comparisons (overlap, spanning regressions,
    conditional stats) require this.
    """
    ym_data = {}
    crsp_sub = crsp_taq[["ym", "permno", "lag_mktcap", "lag_close", "BetaFP"]].copy()
    crsp_sub["ym"] = crsp_sub["ym"].astype(int)
    crsp_sub["permno"] = crsp_sub["permno"].astype(int)
    # No dropna on lag_mktcap: the `> MIN_MKTCAP_M` filter below excludes NaN automatically.

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            ym = year * 100 + month
            ym_str = f"{year:04d}-{month:02d}"
            fret_path = Path(fret30min_ym_path) / f"fret30min_{ym_str}.csv"
            if not fret_path.exists():
                continue
            raw = pd.read_csv(fret_path)
            dt_col = raw.columns[0]
            raw[dt_col] = pd.to_datetime(raw[dt_col])
            raw_cols = [c for c in raw.columns[1:] if str(c).isdigit()
                        and raw[c].isna().sum() <= N_BARS_30]
            # zero-ratio counts NaN (gap) and 0 (no-trade) bars equivalently as illiquid.
            zero_ratio = ((raw[raw_cols] == 0) | raw[raw_cols].isna()).mean()
            perm_cols = [c for c in raw_cols if zero_ratio[c] <= max_zero_ratio]
            raw[perm_cols] = raw[perm_cols].fillna(0)

            crsp_ym = crsp_sub[
                (crsp_sub["ym"] == ym)
                & (crsp_sub["lag_mktcap"] > MIN_MKTCAP_M)
                & (crsp_sub["lag_close"] > MIN_CLOSE_PRICE)
                & crsp_sub["BetaFP"].notna()
            ]
            crsp_permnos = set(crsp_ym["permno"].tolist())
            avail_permnos = [int(c) for c in perm_cols if int(c) in crsp_permnos]
            mktcap = crsp_ym.set_index("permno")["lag_mktcap"]
            if not avail_permnos:
                raise ValueError(f"{ym}: no available permnos after intersection/quality filters")
            avail_cols = [str(p) for p in avail_permnos]
            fRet_ym = raw[[dt_col] + avail_cols].copy()
            fRet_ym["date"] = fRet_ym[dt_col].dt.strftime("%Y%m%d").astype(int)
            fRet_ym["minute_index"] = fRet_ym[dt_col].dt.hour * 60 + fRet_ym[dt_col].dt.minute
            fRet_ym = fRet_ym.drop(columns=[dt_col]).set_index(["date", "minute_index"]).fillna(0)

            wt = mktcap.reindex(avail_permnos).dropna()
            wt_sum = float(wt.sum())
            if wt_sum <= 0:
                raise ValueError(f"{ym}: non-positive vwMKT weight sum ({wt_sum})")
            wt = wt / wt_sum
            # Log-return portfolio aggregated as Σ wᵢ rᵢ (true is log(Σ wᵢ exp(rᵢ))).
            # Per-bar bias O(σ²/2) ≈ 5e-6, immaterial for β regressions; matches HF
            # realized-var convention used throughout pca.py/decomposition.py.
            vwMKT30_arr = fRet_ym[avail_cols].fillna(0).values @ wt.values

            ym_data[ym] = {
                "fRet30":   fRet_ym,
                "vwMKT30":  vwMKT30_arr,
                "vwMKT_wt": wt,
            }

    return ym_data


def get_rBetas(dMKT, dSTK):
    """Per-stock realized beta of dSTK columns on the series dMKT, full sample.
    Denominator masks bars where the stock is untraded (dSTK=0). Frequency-agnostic:
    the caller supplies the return series (5-min in the PCA path, 30-min elsewhere)."""
    dMKT = np.atleast_1d(dMKT).ravel().astype(float)
    mask = (dSTK != 0).astype(float)
    numer = (dMKT[:, None] * dSTK).sum(axis=0)
    denom = ((dMKT[:, None] ** 2) * mask).sum(axis=0)
    return np.where(denom != 0, numer / denom, np.nan)


def _medrv(x):
    """Median realized variance of x (jump-robust variance estimator)."""
    r = np.abs(np.atleast_1d(x).ravel().astype(float))
    n = r.size
    med = np.median(np.vstack([r[:-2], r[1:-1], r[2:]]), axis=0)
    c = (np.pi / (6 - 4 * np.sqrt(3) + np.pi)) * (n / (n - 2))
    return c * np.sum(med ** 2)


def get_robust_rBetas(x, y):   # x = regressor (denominator var), y = dependent (numerator)
    """Jump-robust β of y on x: rBPCov(x, y) / MedRV(x). x is the projection direction.
    y may be 1-D (one asset → scalar) or 2-D (T × N → N exposures); an all-zero y column
    returns NaN. rBPCov / MedRV downweight jumps (LNF carries more jumps than g)."""
    x = np.atleast_1d(x).ravel().astype(float)
    y_arr = np.asarray(y, dtype=float)
    one_d = y_arr.ndim == 1
    if one_d:
        y_arr = y_arr[:, None]
    x_col = x[:, None]
    plus  = np.abs(x_col[1:] + y_arr[1:]) * np.abs(x_col[:-1] + y_arr[:-1])
    minus = np.abs(x_col[1:] - y_arr[1:]) * np.abs(x_col[:-1] - y_arr[:-1])
    rbpcov = (np.pi / 8) * (plus - minus).sum(axis=0)
    medrv = _medrv(x)
    traded = (y_arr != 0).any(axis=0)
    beta = np.where(traded & (medrv != 0), rbpcov / medrv, np.nan)
    return float(beta[0]) if one_d else beta


def load_port_analysis_data():
    """Load all data needed for portfolio analysis into a single dict."""
    ret_dir = Path(ret_path)

    crsp_taq = pd.read_csv(Path(data_path) / "crsp_taq_monthly_info.csv", low_memory=False)
    crsp_taq["ym"]     = crsp_taq["ym"].astype(int)
    crsp_taq["permno"] = crsp_taq["permno"].astype(int)
    # Canonicalize to one row per (ym, permno) at the data-loading boundary.
    crsp_taq = crsp_taq.drop_duplicates(["ym", "permno"], keep="last")
    # ffill lag_mktcap, capped at 2 months to avoid stale mcap across delisting/relisting gaps.
    crsp_taq = crsp_taq.sort_values(["permno", "ym"]).copy()
    crsp_taq["lag_mktcap"] = crsp_taq.groupby("permno")["lag_mktcap"].ffill(limit=2)
    crsp_taq = crsp_taq.dropna(subset=["lag_mktcap"]).copy()

    beta_fp = pd.read_csv(
        Path(data_path) / "monthly_lagged_betas.csv",
        usecols=["yyyymm", "permno", "BetaFP"],
    ).rename(columns={"yyyymm": "ym"})
    beta_fp["ym"] = beta_fp["ym"].astype(int)
    beta_fp["permno"] = beta_fp["permno"].astype(int)
    beta_fp = beta_fp.drop_duplicates(["ym", "permno"], keep="last")
    beta_fp = beta_fp.sort_values(["permno", "ym"]).copy()
    beta_fp["BetaFP"] = beta_fp.groupby("permno")["BetaFP"].ffill()
    # Defensive: drop pre-existing BetaFP to avoid a _x/_y suffix collision on merge.
    crsp_taq = crsp_taq.drop(columns=["BetaFP"], errors="ignore")
    crsp_taq = crsp_taq.merge(beta_fp, on=["ym", "permno"], how="left")

    crsp_ret = crsp_taq[["ym", "permno", "ret"]].drop_duplicates(["ym", "permno"])
    crsp_taq = crsp_taq[["ym", "permno", "lag_mktcap", "lag_close", "BetaFP"]].copy()

    intra = pd.read_csv(ret_dir / "ret_intraday_data.csv")
    over  = pd.read_csv(ret_dir / "ret_overnight_data.csv")
    perm_cols = [c for c in intra.columns if c != "date"]
    # compound daily simple returns: prod(1+r)-1
    intra_monthly = (
        intra[perm_cols]
        .add(1.0)
        .groupby(intra["date"] // 100)
        .prod()
        .sub(1.0)
    )
    over_monthly = (
        over[perm_cols]
        .add(1.0)
        .groupby(over["date"] // 100)
        .prod()
        .sub(1.0)
    )
    intra_monthly.index.name = "ym"
    over_monthly.index.name = "ym"

    ff6    = pd.read_csv(ret_dir / "ff6_monthly_returns.csv")
    ff6_rf = ff6.set_index("yyyymm")["RF"].to_dict()   # {yyyymm: monthly RF (decimal)}

    return {
        "crsp_taq":      crsp_taq,
        "crsp_ret":      crsp_ret,
        "intra_monthly": intra_monthly,
        "over_monthly":  over_monthly,
        "ff6":           ff6,
        "ff6_rf":        ff6_rf,
    }


def get_port_monthly_ret(wt, ym, data, suffix=""):
    """Single-month excess returns (full/intra/over). Excess = raw − total_wt·RF
    (intra/over each get RF/2). Output keys: ret_ex{suffix}, intra_ex{suffix}, over_ex{suffix}.
    Underinvest convention: missing returns filled 0, weights not renormalized — safe here
    because the return panel (mthprevcap > $1M) is broader than the portfolio universe
    (MIN_MKTCAP_M = $200M), so missing is not driven by small-cap censoring."""
    crsp_ret      = data["crsp_ret"]
    intra_monthly = data["intra_monthly"]
    over_monthly  = data["over_monthly"]
    ret_col = "ret"

    monthly = crsp_ret[crsp_ret["ym"] == ym].set_index("permno")[ret_col]
    if monthly.empty:
        return {
            f"ret_ex{suffix}": np.nan,
            f"intra_ex{suffix}": np.nan,
            f"over_ex{suffix}": np.nan,
        }

    str_cols = [str(p) for p in wt.index if str(p) in intra_monthly.columns]
    wt_sub   = wt.reindex([int(c) for c in str_cols]).fillna(0).values
    total_wt = float(wt.sum())
    rf = data["ff6_rf"].get(ym, 0.0)

    r_full  = (wt * monthly.reindex(wt.index).fillna(0)).sum()
    r_intra = (intra_monthly.loc[ym, str_cols].fillna(0).values @ wt_sub
               if ym in intra_monthly.index else np.nan)
    r_over  = (over_monthly.loc[ym,  str_cols].fillna(0).values @ wt_sub
               if ym in over_monthly.index  else np.nan)
    return {
        f"ret_ex{suffix}": r_full - total_wt * rf,
        f"intra_ex{suffix}": r_intra - total_wt * rf / 2 if not np.isnan(r_intra) else np.nan,
        f"over_ex{suffix}": r_over - total_wt * rf / 2 if not np.isnan(r_over) else np.nan,
    }


def get_port_daily_ret(wt, ym, data, suffix=""):
    """Single-month daily excess returns using month-ym weights on daily returns."""
    str_cols = [str(p) for p in wt.index if str(p) in data.columns]
    daily = data[data["ym"] == ym].copy()
    if daily.empty or not str_cols:
        return pd.DataFrame(columns=["date", "special_day", "non_special_day", f"ret_ex{suffix}"])

    wt_sub = wt.reindex([int(c) for c in str_cols]).fillna(0).values
    total_wt = float(wt.sum())
    r_full = daily[str_cols].fillna(0).values @ wt_sub

    out = daily[["date", "special_day", "non_special_day"]].copy()
    out[f"ret_ex{suffix}"] = r_full - total_wt * daily["RF"].fillna(0).values
    return out


def stargazer_with_tstats(models, covariate_order=None, latex_filename=None, digits=2):
    from stargazer.stargazer import Stargazer
    from IPython.display import HTML

    stargazer = Stargazer(models)
    stargazer.significant_digits(digits)
    if covariate_order is not None:
        existing = set(stargazer.cov_names)
        filtered = [c for c in covariate_order if c in existing]
        if filtered:
            stargazer.covariate_order(filtered)
    stargazer.custom_note_label("t statistics in parentheses")
    for i, md in enumerate(stargazer.model_data):
        for cov in md["cov_names"]:
            if cov in models[i].tvalues.index:
                md["cov_std_err"][cov] = float(models[i].tvalues[cov])
            elif cov == "Intercept" and "const" in models[i].tvalues.index:
                md["cov_std_err"][cov] = float(models[i].tvalues["const"])
    if latex_filename is not None:
        latex = stargazer.render_latex()
        lines = latex.splitlines()
        n_models = len(models)

        # normalize stargazer output across versions: split \begin{table}, inject \centering
        if lines and lines[0].startswith(r"\begin{table}[!htbp]"):
            lines[0] = r"\begin{table}[!htbp]"
            lines.insert(1, r"\centering")

        for i, line in enumerate(lines):
            if line.startswith(r"\begin{tabular}{"):
                lines[i] = rf"\begin{{tabularx}}{{\textwidth}}{{l *{{{n_models}}}{{>{{\centering\arraybackslash}}X}}}}"
                break

        for i, line in enumerate(lines):
            if line.strip() == r"\end{tabular}":
                lines[i] = r"\end{tabularx}"
                break

        latex_file = Path(result_path) / latex_filename
        with open(latex_file, "w") as f:
            f.write("\n".join(lines))
    return HTML(stargazer.render_html())


def _build_ym_permno_lookup(df, value_col, ym_col="ym", id_col="permno"):
    # Backtest helper: ym -> Series(index=permno, value=value_col).
    out = {}
    for ym, g in df.groupby(ym_col, sort=True):
        out[int(ym)] = g.set_index(id_col)[value_col]
    return out


def _vw_excess_return(ret, cap_sub, rf):
    if len(cap_sub) == 0:
        return np.nan
    ws = float(cap_sub.sum())
    if ws <= 0:
        return np.nan
    w = cap_sub / ws
    raw = float((ret.reindex(w.index).fillna(0.0) * w).sum())
    return raw - rf


def SR_diff_test_HC1(r1, r2):
    """
    Sharpe ratio difference test using HC1 heteroskedasticity-robust variance.
    Does NOT correct for autocorrelation.
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)

    if len(r1) != len(r2):
        raise ValueError("Return series must have equal length")

    T = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = np.mean(r1 ** 2), np.mean(r2 ** 2)

    var1, var2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    s1, s2 = np.sqrt(var1), np.sqrt(var2)

    Delta_hat = mu1 / s1 - mu2 / s2

    grad = np.array([
        1 / s1 + mu1 ** 2 / s1 ** 3,
        -(1 / s2 + mu2 ** 2 / s2 ** 3),
        -mu1 / (2 * s1 ** 3),
        mu2 / (2 * s2 ** 3),
    ])

    y = np.column_stack([
        r1 - mu1,
        r2 - mu2,
        r1 ** 2 - g1,
        r2 ** 2 - g2,
    ])

    Psi = (y.T @ y) / T
    k = 4
    Psi *= T / (T - k)

    var_D = grad.T @ Psi @ grad / T
    SE = np.sqrt(var_D)

    t_stat = Delta_hat / SE
    return t_stat


def SR_diff_test_indep(r1, r2, lags=10):
    """Sharpe-ratio difference test for TWO INDEPENDENT samples (different time
    indices, e.g. High-overlap months vs Low-overlap months, or Special days vs
    Non-special days). Per-sample SR variance is estimated by GMM on moments
    (r - mu, r^2 - g) with Newey-West HAC long-run variance (Bartlett kernel,
    `lags` lags). Independence makes cov(SR1, SR2) = 0, so
        Var(SR1 - SR2) = Var(SR1) + Var(SR2).
    Returns t_stat = (SR1 - SR2) / sqrt(Var(SR1) + Var(SR2)).
    Inputs may have different lengths; NaNs are dropped per sample.
    """
    def _sr_and_var(r, lags):
        r = np.asarray(r, dtype=float)
        r = r[np.isfinite(r)]
        T = len(r)
        if T < 3:
            return np.nan, np.nan
        mu = float(r.mean())
        g = float((r ** 2).mean())
        sigma2 = g - mu ** 2
        if sigma2 <= 0:
            return np.nan, np.nan
        sigma = np.sqrt(sigma2)
        SR = mu / sigma
        grad = np.array([1.0 / sigma + mu ** 2 / sigma ** 3,
                         -mu / (2.0 * sigma ** 3)])
        # m_t = (r_t - mu, r_t^2 - g). Newey-West Bartlett long-run variance.
        m = np.column_stack([r - mu, r ** 2 - g])
        Psi = (m.T @ m) / T
        max_lag = min(lags, T - 1)
        for j in range(1, max_lag + 1):
            G = (m[j:].T @ m[:-j]) / T
            w = 1.0 - j / (lags + 1)
            Psi = Psi + w * (G + G.T)
        var_SR = float(grad @ Psi @ grad) / T
        return SR, var_SR

    SR1, var1 = _sr_and_var(r1, lags)
    SR2, var2 = _sr_and_var(r2, lags)
    if not (np.isfinite(SR1) and np.isfinite(SR2)) or var1 <= 0 or var2 <= 0:
        return np.nan
    return (SR1 - SR2) / np.sqrt(var1 + var2)


def _ols_intercept_slope_r2(y, x):
    """Single reg y ~ x (with intercept). Returns (intercept, slope, adj_R2)."""
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    mx, my = np.mean(x), np.mean(y)
    var_x = np.var(x, ddof=1)
    if var_x <= 0:
        return np.nan, np.nan, np.nan
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    slope = cov_xy / var_x
    intercept = my - slope * mx
    var_y = np.var(y, ddof=1)
    r2 = (cov_xy ** 2 / (var_x * var_y)) if (var_y > 0) else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - 2) if n > 2 else np.nan
    return intercept, slope, adj_r2


def _weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size == 0:
        return np.nan
    return float(np.sum(w * x) / np.sum(w))


def _cluster_mean_stats(x, cluster, w=None):
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "cluster": cluster})
    if w is not None:
        df["w"] = pd.to_numeric(w, errors="coerce")
        df = df[np.isfinite(df["x"]) & np.isfinite(df["w"]) & (df["w"] > 0)].copy()
    else:
        df = df[np.isfinite(df["x"])].copy()
    if df.empty:
        return pd.Series({"mean": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_cluster": 0})

    df["cluster"] = df["cluster"].astype("string")
    df = df[df["cluster"].notna() & (df["cluster"] != "")]
    if df.empty:
        return pd.Series({"mean": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_cluster": 0})

    if w is None:
        mean = float(df["x"].mean())
    else:
        mean = _weighted_mean(df["x"].to_numpy(dtype=float), df["w"].to_numpy(dtype=float))

    n_cluster = int(df["cluster"].nunique())
    if n_cluster < 2 or len(df) < 2:
        return pd.Series({"mean": mean, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_cluster": n_cluster})

    if w is None:
        md = smf.ols("x ~ 1", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["cluster"]})
    else:
        md = smf.wls("x ~ 1", data=df, weights=df["w"]).fit(cov_type="cluster", cov_kwds={"groups": df["cluster"]})
    se = float(md.bse.iloc[0])
    return pd.Series({
        "mean": mean,
        "se": se,
        "ci_lo": mean - 1.96 * se,
        "ci_hi": mean + 1.96 * se,
        "n_cluster": n_cluster,
    })


def _nw_mean_ci(s, lags):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    maxlags = int(max(0, min(lags, len(s) - 1)))
    md = smf.ols("x ~ 1", data=pd.DataFrame({"x": s})).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    mean = float(s.mean())
    se = float(md.bse.iloc[0])
    return mean, se, mean - 1.96 * se, mean + 1.96 * se, len(s)
