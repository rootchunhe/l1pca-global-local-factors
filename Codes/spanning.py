import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

from constants import (
    ret_path,
    PORT_RG, PORT_LNF, PORT_MKTG,
)




def load_external_factor_returns(file_map=None):
    if file_map is None:
        file_map = {
            "ivol_21d": "[usa]_[ivol_capm_21d]_[monthly]_[vw_cap].csv",
            "ivol_252d": "[usa]_[ivol_capm_252d]_[monthly]_[vw_cap].csv",
            "low_risk": "[usa]_[low_risk]_[monthly]_[vw_cap].csv",
            "seasonality": "[usa]_[seasonality]_[monthly]_[vw_cap].csv",
            "momentum": "[usa]_[momentum]_[monthly]_[vw_cap].csv",
            "short_term_reversal": "[usa]_[short_term_reversal]_[monthly]_[vw_cap].csv",
        }
    ret_dir = Path(ret_path)
    items = list(file_map.items())
    fac0, fname0 = items[0]
    out = pd.read_csv(ret_dir / fname0, usecols=["date", "ret"]).rename(columns={"ret": fac0})
    out["yyyymm"] = pd.to_datetime(out["date"]).dt.strftime("%Y%m").astype(int)
    out = out[["yyyymm", fac0]]
    for fac, fname in items[1:]:
        df = pd.read_csv(ret_dir / fname, usecols=["date", "ret"]).rename(columns={"ret": fac})
        df["yyyymm"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m").astype(int)
        df = df[["yyyymm", fac]]
        out = out.merge(df, on="yyyymm", how="outer")
    return out



def prepare_insample_spanning_df(ret_ext, ff6_df, g_port=PORT_RG, lnf_port=None, ret_col="ret_ex"):
    """Build the wide monthly regression panel for spanning tests.
    Inputs: ret_ext, ff6_df, JKP external factors — all expected as monthly decimal returns.
    Output: all return columns scaled by 1200 (= 12 × 100) to annualized percent."""
    if lnf_port is None:
        if str(g_port).startswith(PORT_MKTG):
            lnf_port = str(g_port).replace(PORT_MKTG, PORT_LNF)
        else:
            lnf_port = str(g_port).replace(PORT_RG, PORT_LNF)

    need_cols = ["ym", "port", f"{ret_col}_is"]
    s = ret_ext[need_cols].copy()
    piv = s.pivot(index="ym", columns="port", values=[f"{ret_col}_is"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index().rename(columns={"ym": "yyyymm"})

    reg = pd.DataFrame({
        "yyyymm": piv["yyyymm"],
        "g_rf": piv[f"{ret_col}_is_{g_port}"],
        "lnf": piv[f"{ret_col}_is_{lnf_port}"],
    })

    ff6 = ff6_df.copy()

    use_cols = [c for c in ["yyyymm", "RF", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "BAB"] if c in ff6.columns]
    reg = reg.merge(ff6[use_cols], on="yyyymm", how="inner")

    ext = load_external_factor_returns()
    reg = reg.merge(ext, on="yyyymm", how="left").sort_values("yyyymm").reset_index(drop=True)

    num_cols = [c for c in reg.columns if c != "yyyymm"]
    reg[num_cols] = reg[num_cols].apply(pd.to_numeric, errors="coerce")
    # × 1200 = 12×100: monthly decimal → annualized %. Applies to ALL non-yyyymm cols, so
    # every input must be monthly decimal (else over-scaled).
    reg[num_cols] = reg[num_cols] * 1200
    return reg



def run_traditional_spanning(reg_df, f1, f2=None):
    ff6_terms = [c for c in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in reg_df.columns]
    has_bab = "BAB" in reg_df.columns

    def _forms(y):
        m1 = f"{y} ~ MKT_RF"
        m2 = f"{y} ~ MKT_RF+BAB" if has_bab else m1
        rhs_ff6 = "+".join(ff6_terms)
        m3 = f"{y} ~ {rhs_ff6}"
        m4 = f"{y} ~ {rhs_ff6}+BAB" if has_bab else m3
        return [m1, m2, m3, m4]

    formulas = _forms(f1)
    if f2 is not None:
        formulas += _forms(f2)
    return [smf.ols(f, data=reg_df).fit(cov_type="HC1") for f in formulas]



def compare_capm_spanning(reg_df, span_cols=None, pair_cols=()):
    if span_cols is None:
        span_cols = ["ivol_21d", "ivol_252d", "low_risk"]
    if len(pair_cols) != 2:
        raise ValueError("pair_cols must be provided explicitly as decomposition of market factor, e.g. ('Rg_rf', 'lnf').")
    p1, p2 = pair_cols
    models = []
    for col in span_cols:
        if col not in reg_df.columns:
            continue
        models.append(smf.ols(f"{col} ~ MKT_RF", data=reg_df).fit(cov_type="HC1"))
        models.append(smf.ols(f"{col} ~ {p1}+{p2}", data=reg_df).fit(cov_type="HC1"))
    return models



def compare_ff6_spanning(reg_df, span_cols=None, pair_cols=(), add_bab=False):
    if span_cols is None:
        span_cols = ["ivol_21d", "ivol_252d", "low_risk"]
    if len(pair_cols) != 2:
        raise ValueError("pair_cols must be provided explicitly as decomposition of market factor, e.g. ('Rg_rf', 'lnf').")
    p1, p2 = pair_cols
    ff6_terms = [c for c in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in reg_df.columns]
    if add_bab and "BAB" in reg_df.columns:
        ff6_terms = ff6_terms + ["BAB"]
    rhs_ff6 = "+".join(ff6_terms)
    rhs_pair_terms = [c for c in ["SMB", "HML", "RMW", "CMA", "MOM"] if c in reg_df.columns]
    if add_bab and "BAB" in reg_df.columns:
        rhs_pair_terms = rhs_pair_terms + ["BAB"]
    rhs_pair_ff6 = "+".join([p1, p2] + rhs_pair_terms)
    models = []
    for col in span_cols:
        if col not in reg_df.columns:
            continue
        models.append(smf.ols(f"{col} ~ {rhs_ff6}", data=reg_df).fit(cov_type="HC1"))
        models.append(smf.ols(f"{col} ~ {rhs_pair_ff6}", data=reg_df).fit(cov_type="HC1"))
    return models



def run_span_bab_models(reg_df, pair_cols=()):
    if len(pair_cols) != 2:
        raise ValueError("pair_cols must be provided explicitly as decomposition of market factor, e.g. ('Rg_rf', 'lnf').")
    p1, p2 = pair_cols

    ff4_terms = [c for c in ["MKT_RF", "SMB", "HML", "MOM"] if c in reg_df.columns]
    ff6_terms = [c for c in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in reg_df.columns]
    if "MKT_RF" not in ff4_terms or "MKT_RF" not in ff6_terms:
        raise ValueError("MKT_RF is required for BAB spanning models.")

    rhs_ff4 = "+".join(ff4_terms)
    rhs_ff6 = "+".join(ff6_terms)
    rhs_pair_ff4 = "+".join([p1, p2] + [c for c in ["SMB", "HML", "MOM"] if c in reg_df.columns])
    rhs_pair_ff6 = "+".join([p1, p2] + [c for c in ["SMB", "HML", "RMW", "CMA", "MOM"] if c in reg_df.columns])

    formulas = [
        "BAB ~ MKT_RF",
        f"BAB ~ {p1}+{p2}",
        f"BAB ~ {rhs_ff4}",
        f"BAB ~ {rhs_pair_ff4}",
        f"BAB ~ {rhs_ff6}",
        f"BAB ~ {rhs_pair_ff6}",
    ]
    return [smf.ols(f, data=reg_df).fit(cov_type="HC1") for f in formulas]



def run_span_bab_vol_timed_models(reg_df, pair_cols=()):
    if len(pair_cols) != 2:
        raise ValueError("pair_cols must be provided explicitly as decomposition of market factor, e.g. ('Rg_rf', 'lnf').")
    p1, p2 = pair_cols

    ff6_daily = pd.read_csv(Path(ret_path) / "ff6_daily_returns.csv")
    ff6_daily["date"] = pd.to_numeric(ff6_daily["date"], errors="coerce")
    ff6_daily["yyyymm"] = (ff6_daily["date"] // 100).astype("Int64")
    ff6_daily["BAB2"] = pd.to_numeric(ff6_daily["BAB"], errors="coerce") ** 2
    bab_vols = (
        ff6_daily.groupby("yyyymm")["BAB2"]
        .sum()
        .apply(lambda x: np.sqrt(x) * np.sqrt(12) * 100.0)
        .sort_index()
    )
    bab_vols_lag1 = bab_vols.shift(1)
    reg_ym = pd.to_numeric(reg_df["yyyymm"], errors="coerce").dropna().astype(int)
    ym_min, ym_max = int(reg_ym.min()), int(reg_ym.max())
    aligned_lag1 = bab_vols_lag1[(bab_vols_lag1.index >= ym_min) & (bab_vols_lag1.index <= ym_max)].dropna().sort_index()

    med = aligned_lag1.median()
    # Median month is assigned to "low" in both the subsample split (bab_models1)
    # and the dummy variable (bab_models2). Use <= for low, > for high consistently.
    bab_vols_low = aligned_lag1[aligned_lag1 <= med].index.to_list()
    bab_vols_high = aligned_lag1[aligned_lag1 > med].index.to_list()

    print(
        "run_span_bab_vol_timed_models: grouping by lagged BAB vol "
        f"(t-1 month annualized realized vol from daily BAB), "
        f"median={med:.2f}%, low_n={len(bab_vols_low)}, high_n={len(bab_vols_high)}"
    )

    vol_lag_df = bab_vols_lag1.rename("bab_vol_lag1").reset_index()
    reg_aug = reg_df.merge(vol_lag_df, on="yyyymm", how="left")
    # Use log lagged BAB volatility in the continuous-spec regressions.
    reg_aug["lag_bab_vol"] = np.where(reg_aug["bab_vol_lag1"] > 0, np.log(reg_aug["bab_vol_lag1"]), np.nan)

    reg_low = reg_aug[reg_aug["yyyymm"].isin(bab_vols_low)].copy()
    reg_high = reg_aug[reg_aug["yyyymm"].isin(bab_vols_high)].copy()

    formulas_vol_timed = [
        "BAB ~ MKT_RF",
        f"BAB ~ {p1}+{p2}",
        "BAB ~ MKT_RF + SMB + HML + RMW + CMA + MOM",
        f"BAB ~ {p1}+{p2}+SMB+HML+RMW+CMA+MOM",
    ]
    bab_models1 = []
    for f in formulas_vol_timed:
        bab_models1.append(smf.ols(f, data=reg_low).fit(cov_type="HC1"))
        bab_models1.append(smf.ols(f, data=reg_high).fit(cov_type="HC1"))

    reg_vol = reg_aug.dropna(subset=["bab_vol_lag1"]).copy()
    # bab_models2: dummy = 1 if lag_bab_vol <= median (matches bab_models1 boundary); 6 specs w/ dummy + interactions
    reg_vol = reg_vol.copy()
    reg_vol["low_bab_vol"] = (reg_vol["bab_vol_lag1"] <= med).astype(int)
    formulas2_dummy = [
        "BAB ~ MKT_RF",
        "BAB ~ low_bab_vol + MKT_RF",
        "BAB ~ low_bab_vol * MKT_RF",
        f"BAB ~ {p1}+{p2}",
        f"BAB ~ low_bab_vol + {p1}+{p2}",
        f"BAB ~ low_bab_vol * ({p1}+{p2})",
    ]
    bab_models2 = [smf.ols(f, data=reg_vol).fit(cov_type="HC1") for f in formulas2_dummy]
    # bab_models3: continuous lag_bab_vol + interactions
    formulas3 = [
        "BAB ~ MKT_RF",
        "BAB ~ lag_bab_vol + MKT_RF",
        "BAB ~ lag_bab_vol * MKT_RF",
        f"BAB ~ {p1}+{p2}",
        f"BAB ~ lag_bab_vol + {p1}+{p2}",
        f"BAB ~ lag_bab_vol * ({p1}+{p2})",
    ]
    bab_models3 = [smf.ols(f, data=reg_vol).fit(cov_type="HC1") for f in formulas3]

    bab_group_ym = pd.concat([
        pd.DataFrame({"ym": bab_vols_low, "Group": "Low bab_vol"}),
        pd.DataFrame({"ym": bab_vols_high, "Group": "High bab_vol"}),
    ], ignore_index=True)
    return bab_models1, bab_models2, bab_models3, bab_group_ym
