import os
import json
import urllib.request
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from pathlib import Path
import openassetpricing as oap
from numba import jit

from constants import (
    data_path,
    hf_5min_path,
    fret30min_ym_path,
    ret_path,
    spy_path,
    spy_c_hf_5min_ym_path,
    N_BARS_PER_DAY as INTRADAY_BARS_PER_DAY,
)

SPY_PERMNO = 84398
HF_MIN_BAR_COUNT = 38        # minimum 5-min bars for a stock to be included on a given day

def _roll_beta_1D(y, x, window, min_obs):
    """Rolling cov(y,x)/var(x). Returns 1-D array. Numba JIT compatible."""
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        sum_y, sum_x, sum_xy, sum_xx, cnt = 0.0, 0.0, 0.0, 0.0, 0
        for k in range(window):
            idx = i - window + 1 + k
            yi, xi = y[idx], x[idx]
            if not (np.isnan(yi) or np.isnan(xi)):
                sum_y  += yi
                sum_x  += xi
                sum_xy += yi * xi
                sum_xx += xi * xi
                cnt    += 1
        if cnt >= min_obs:
            mean_y = sum_y / cnt
            mean_x = sum_x / cnt
            vx = sum_xx / cnt - mean_x * mean_x
            if vx > 1e-12:
                cov_yx = sum_xy / cnt - mean_y * mean_x
                out[i] = cov_yx / vx
    return out

_roll_beta_1D = jit(nopython=True, cache=True)(_roll_beta_1D)


def _roll_betafp_fp(stk_ret, mkt_ret, window_corr=1260, window_vol=252, min_obs_corr=504, min_obs_vol=120):
    """Frazzini-Pedersen (2014) BetaFP: beta = corr_1260d(r3,r3) * (vol_252d_stk / vol_252d_mkt).
    Returns 1-D array. Numba JIT if available."""
    n = len(stk_ret)
    out = np.full(n, np.nan, dtype=np.float64)
    stk_ret = np.asarray(stk_ret, dtype=np.float64)
    mkt_ret = np.asarray(mkt_ret, dtype=np.float64)
    log_stk = np.log1p(stk_ret)
    log_mkt = np.log1p(mkt_ret)
    r3_stk = np.full(n, np.nan, dtype=np.float64)
    r3_mkt = np.full(n, np.nan, dtype=np.float64)
    for i in range(2, n):
        if not (np.isnan(log_stk[i]) or np.isnan(log_stk[i-1]) or np.isnan(log_stk[i-2])):
            r3_stk[i] = log_stk[i] + log_stk[i-1] + log_stk[i-2]
        if not (np.isnan(log_mkt[i]) or np.isnan(log_mkt[i-1]) or np.isnan(log_mkt[i-2])):
            r3_mkt[i] = log_mkt[i] + log_mkt[i-1] + log_mkt[i-2]
    i_start = max(window_corr - 1, window_vol - 1)
    for i in range(i_start, n):
        sum_s, sum_m, sum_ss, sum_mm, cnt = 0.0, 0.0, 0.0, 0.0, 0
        for k in range(window_vol):
            idx = i - window_vol + 1 + k
            s, m = log_stk[idx], log_mkt[idx]
            if not (np.isnan(s) or np.isnan(m)):
                sum_s += s
                sum_m += m
                sum_ss += s * s
                sum_mm += m * m
                cnt += 1
        if cnt < min_obs_vol:
            continue
        mean_s = sum_s / cnt
        mean_m = sum_m / cnt
        var_s = sum_ss / cnt - mean_s * mean_s
        var_m = sum_mm / cnt - mean_m * mean_m
        if var_m < 1e-12 or var_s < 1e-12:
            continue
        vol_stk = np.sqrt(var_s)
        vol_mkt = np.sqrt(var_m)
        sum_s, sum_m, sum_ss, sum_mm, sum_sm, cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for k in range(window_corr):
            idx = i - window_corr + 1 + k
            s, m = r3_stk[idx], r3_mkt[idx]
            if not (np.isnan(s) or np.isnan(m)):
                sum_s += s
                sum_m += m
                sum_ss += s * s
                sum_mm += m * m
                sum_sm += s * m
                cnt += 1
        if cnt < min_obs_corr:
            continue
        mean_s = sum_s / cnt
        mean_m = sum_m / cnt
        cov = sum_sm / cnt - mean_s * mean_m
        var_s = sum_ss / cnt - mean_s * mean_s
        var_m = sum_mm / cnt - mean_m * mean_m
        denom = np.sqrt(var_s * var_m)
        if denom < 1e-12:
            continue
        corr = cov / denom
        if corr < -1.0:
            corr = -1.0
        elif corr > 1.0:
            corr = 1.0
        out[i] = corr * vol_stk / vol_mkt
    return out


_roll_betafp_fp = jit(nopython=True, cache=True)(_roll_betafp_fp)


# Paths imported from constants.py. Ensure raw-data directories exist.
for _d in (data_path, hf_5min_path, fret30min_ym_path,
           ret_path, spy_path, spy_c_hf_5min_ym_path):
    os.makedirs(_d, exist_ok=True)

_conn = None
def lazy_wrds_conn():
    global _conn
    if _conn is None:
        import wrds
        _conn = wrds.Connection()
    return _conn


def _drop_exact_duplicates(df, label):
    """Drop row-wise duplicates and print a small audit message."""
    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)
    if removed:
        print(f"{label}: dropped {removed:,} exact duplicate rows")
    return df


def _assert_unique_keys(df, keys, label):
    """Fail fast when a table expected to be keyed by `keys` is not unique."""
    dup = df[df.duplicated(keys, keep=False)].copy()
    if dup.empty:
        return
    sample = (
        dup[keys]
        .value_counts()
        .rename("n_rows")
        .reset_index()
        .head(10)
        .to_string(index=False)
    )
    raise ValueError(
        f"{label} must be unique on {keys}, but found {dup[keys].drop_duplicates().shape[0]:,} "
        f"duplicate key(s). Sample:\n{sample}"
    )


def get_crsp_taq_monthly_info(start_year=2006, end_year=2025):
    """Saves crsp_taq_monthly_info.csv: monthly panel for CRSP equities linked to TAQ symbols.
    Columns: permno, ym, ticker, ret, lag_mktcap, sym_root, sym_suffix."""
    if start_year < 2006:
        start_year = 2006
        print("start_year is set to 2006 (taqm library starts in 2006)")
    
    if end_year < 2011:
        end_year = 2011
        print("end_year is set to 2011")

    file_name = "crsp_taq_monthly_info.csv"
    file_path = os.path.join(data_path, file_name)

    if os.path.exists(file_path):
        print(f"{file_name} exists, skip download.")
        return

    conn = lazy_wrds_conn()

    print("=== Download CRSP monthly data ===")

    start_date = f"{start_year}-01-01"
    end_date   = f"{end_year}-12-31"

    crsp_df = conn.raw_sql(f"""
        SELECT
            permno,
            cusip9,
            permco,
            ticker,
            siccd,
            naics,
            mthret as ret,
            mthprevcap as lag_mktcap,
            mthprevprc as lag_close,
            yyyymm as ym
        FROM crspm.wrds_msfv2_query
        WHERE mthcaldt >= '{start_date}'
            AND mthcaldt <= '{end_date}'
            AND securitytype = 'EQTY'
            AND securitysubtype = 'COM'
            AND sharetype IN ('NS', 'AD')
            AND primaryexch IN ('A', 'N', 'Q')
            AND ticker IS NOT NULL
            AND cusip9 IS NOT NULL
            AND mthprevcap > 1000
        ORDER BY ym, permno;
    """, date_cols=['mthcaldt'])

    print(f"CRSP rows loaded: {len(crsp_df):,}")
    crsp_df.loc[crsp_df["ret"].abs() > 50, "ret"] = np.nan # Missing Return Codes: -66, -77, -88, -99

    def get_taq_id_panel(df, id_col):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        if 'sym_root' not in df.columns:
            split = df['symbol_15'].astype('string').str.strip().str.split(n=1, expand=True)
            df['sym_root'] = split[0]
            df['sym_suffix'] = split[1]

        df['ym'] = df['date'].dt.year * 100 + df['date'].dt.month

        df = df[
            df[id_col].notna() &
            df['date'].notna() &
            df['sym_root'].notna()
        ].copy()

        df = df.sort_values([id_col, 'ym', 'date'], kind='mergesort')

        first = (
            df.groupby([id_col,'ym'], sort=False)
              .first()[['date','sym_root','sym_suffix']]
              .rename(columns={
                  'date':'date0',
                  'sym_root':'sym_root0',
                  'sym_suffix':'sym_suffix0'
              })
        )

        last = (
            df.groupby([id_col,'ym'], sort=False)
              .last()[['date','sym_root','sym_suffix']]
        )

        n_symbols = (
            df[[id_col,'ym','sym_root','sym_suffix']]
            .drop_duplicates()
            .groupby([id_col,'ym'], sort=False)
            .size()
            .rename('n_symbols')
        )

        changed_month = (
            last['sym_root'].fillna('') != first['sym_root0'].fillna('')
        ) | (
            last['sym_suffix'].fillna('') != first['sym_suffix0'].fillna('')
        )

        # namechg = first date the new (last) symbol appears (downstream filter splits old/new here)
        grp = df.groupby([id_col, 'ym'], sort=False)
        on_last_sym = (df['sym_root'].fillna('') == grp['sym_root'].transform('last').fillna('')) & \
                      (df['sym_suffix'].fillna('') == grp['sym_suffix'].transform('last').fillna(''))
        chg_date = df.loc[on_last_sym].groupby([id_col, 'ym'], sort=False)['date'].min()

        namechg = pd.Series(pd.NaT, index=last.index, name='namechg')
        namechg.loc[changed_month] = chg_date.reindex(last.index)[changed_month]

        out = (
            last[['sym_root','sym_suffix']]
            .join(first[['sym_root0','sym_suffix0']], how='left')
            .join(namechg, how='left')
            .join(n_symbols, how='left')
            .reset_index()
        )

        no_chg = out['n_symbols'].fillna(1) <= 1
        out.loc[no_chg, ['namechg','sym_root0','sym_suffix0']] = [pd.NaT, pd.NA, pd.NA]

        out = out[[id_col,'ym','sym_root','sym_suffix','namechg','sym_root0','sym_suffix0']]
        return out.sort_values([id_col,'ym'])

    print("=== Match before 2011 using WRDS permno TAQ link ===")

    # https://wrds-www.wharton.upenn.edu/documents/1336/NYSE_Daily_TAQ_to_CRSP_Linking_BPNFjWm.pdf
    link = conn.raw_sql(f"""
        SELECT permno, date, sym_root, sym_suffix
        FROM wrdsapps.taqmclink
        WHERE date >= '{start_date}'
          AND date <  '2011-01-01'
    """)
    link['sym_suffix'] = link['sym_suffix'].fillna('')
    permno_link = get_taq_id_panel(link, 'permno')

    crsp1 = crsp_df[crsp_df.ym < 201013].copy()
    crsp1 = pd.merge(crsp1, permno_link, on=['permno','ym'], how='left')

    print("=== Match 2011+ using TAQ master file (cusip9) ===")

    cusip_link = []
    for year in range(2011, end_year + 1):
        mastm = conn.raw_sql(f"""
            SELECT date, cusip as cusip9, symbol_15
            FROM taqm_{year}.mastm_{year}
        """)
        cusip_link.append(get_taq_id_panel(mastm, 'cusip9'))

    cusip_link = pd.concat(cusip_link, ignore_index=True)

    crsp2 = crsp_df[crsp_df.ym > 201012].copy()
    crsp2 = pd.merge(crsp2, cusip_link, on=['cusip9','ym'], how='left')

    crsp = pd.concat([crsp1, crsp2], ignore_index=True)

    print("=== Fallback: ticker match using TAQ iid table ===")

    iid_all = []
    for year in range(start_year, end_year + 1):
        temp = conn.raw_sql(f"""
            SELECT {year} AS year, sym_root, sym_suffix, avg(total_dollar_m) as avg_total_dollar_m
            FROM taqm_{year}.wrds_iid_{year}
            GROUP BY sym_root, sym_suffix
        """)
        iid_all.append(temp)

    iid_all = pd.concat(iid_all, ignore_index=True)
    iid_all['sym_suffix'] = iid_all['sym_suffix'].fillna('')
    iid_all['sym_id'] = iid_all['sym_root'] + iid_all['sym_suffix']

    # match by sym_root=ticker
    iid_root = (
        iid_all.sort_values('avg_total_dollar_m', ascending=False)
        .drop_duplicates(subset=['year', 'sym_root'], keep='first')
        [['year', 'sym_root']]
    )

    crsp_missing = crsp[crsp['sym_root'].isna()].copy()
    crsp_keep    = crsp[crsp['sym_root'].notna()].copy()

    crsp_missing['year'] = (crsp_missing['ym'] // 100).astype(int)
    crsp_missing_clean = crsp_missing.drop(columns=['sym_root', 'sym_suffix'])

    m1 = crsp_missing_clean.merge(
        iid_root,
        left_on=['year', 'ticker'],
        right_on=['year', 'sym_root'],
        how='left'
    )
    m1['sym_suffix'] = ''  # object dtype, compatible with m2, and avoids all-NA column triggering concat warning

    still_missing = m1['sym_root'].isna()

    # match by sym_root+sym_suffix=ticker
    if still_missing.any():
        m2 = m1.loc[still_missing].drop(columns=['sym_root', 'sym_suffix']).merge(
            iid_all[['year', 'sym_root', 'sym_suffix', 'sym_id']],
            left_on=['year', 'ticker'],
            right_on=['year', 'sym_id'],
            how='left'
        )
        m2['sym_suffix'] = m2['sym_suffix'].replace('', np.nan)
        m2 = m2.drop(columns=['sym_id'], errors='ignore')

        crsp_filled = pd.concat([m1[~still_missing], m2], ignore_index=True)
    else:
        crsp_filled = m1

    crsp = pd.concat([crsp_keep, crsp_filled.drop(columns=['year'])], ignore_index=True)
    crsp = _drop_exact_duplicates(crsp, "crsp_taq_monthly_info")
    _assert_unique_keys(crsp, ["permno", "ym"], "crsp_taq_monthly_info")
    crsp['lag_mktcap'] = crsp['lag_mktcap']/1e3 # now in millions
    crsp.to_csv(file_path, index=False)
    print(f"saved to {file_path}")


def get_crsp_daily_returns(start_year=2006, end_year=2025):
    """Saves ret_overnight_data.csv, ret_intraday_data.csv, ret_daily_data.csv:
    date x permno pivot of overnight, intraday, and total daily returns."""
    daily_path = os.path.join(ret_path, "ret_daily_data.csv")
    if os.path.exists(daily_path):
        print("ret_daily_data.csv exists, skip download.")
        return

    info_path = os.path.join(data_path, "crsp_taq_monthly_info.csv")
    if not os.path.exists(info_path):
        get_crsp_taq_monthly_info(start_year=start_year, end_year=end_year)
    info = pd.read_csv(info_path, low_memory=False)
    permno_set = set(info["permno"].astype(int).unique()) | {SPY_PERMNO}

    conn = lazy_wrds_conn()
    start_date = f"{start_year - 1}-01-01"
    end_date_str = f"{end_year}-12-31"

    daily = conn.raw_sql(f"""
        SELECT
            permno,
            dlycaldt AS date,
            dlyret AS ret,
            dlyprc / dlyopen - 1 AS intraday_ret,
            (dlyret + 1) * dlyopen / dlyprc - 1 AS overnight_ret
        FROM crspm.wrds_dsfv2_query
        WHERE dlycaldt >= '{start_date}' AND dlycaldt <= '{end_date_str}'
          AND dlyprevprc > 0.1
          AND dlyprevcap > 1000
          AND dlyprc > 0
          AND dlyopen > 0.1
        ORDER BY permno, dlycaldt
    """)
    print(f"CRSP daily loaded: {len(daily):,} rows")

    daily["date"] = pd.to_datetime(daily["date"])
    daily["permno"] = daily["permno"].astype(int)
    daily = daily[daily["permno"].isin(permno_set)]
    # CRSP missing return codes -66/-77/-88/-99 also corrupt the derived intraday/overnight cols.
    bad_ret = daily["ret"].abs() > 50
    daily.loc[bad_ret, ["ret", "intraday_ret", "overnight_ret"]] = np.nan

    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    # Drop each permno's first in-window row: its overnight (close-to-open) return
    # references a prior close that lies outside the panel window.
    daily = daily[daily.groupby("permno").cumcount() > 0].reset_index(drop=True)
    daily = daily.groupby(["permno", "date"], as_index=False).first()

    permnos_kept = daily["permno"].unique().tolist()
    base_index = pd.DatetimeIndex(
        daily[daily["permno"] == SPY_PERMNO]["date"].drop_duplicates().sort_values().values
    )

    overnight = daily.pivot(index="date", columns="permno", values="overnight_ret").reindex(base_index)
    intraday = daily.pivot(index="date", columns="permno", values="intraday_ret").reindex(base_index)
    daily_ret = daily.pivot(index="date", columns="permno", values="ret").reindex(base_index)

    overnight.insert(0, "date", pd.to_datetime(base_index).strftime("%Y%m%d").astype(int))
    intraday.insert(0, "date", pd.to_datetime(base_index).strftime("%Y%m%d").astype(int))
    daily_ret.insert(0, "date", pd.to_datetime(base_index).strftime("%Y%m%d").astype(int))

    overnight = overnight.rename(columns={SPY_PERMNO: "SPY"})
    intraday = intraday.rename(columns={SPY_PERMNO: "SPY"})
    daily_ret = daily_ret.rename(columns={SPY_PERMNO: "SPY"})
    cols = ["date", "SPY"] + sorted([c for c in overnight.columns if c not in ("date", "SPY")])
    overnight[cols].to_csv(os.path.join(ret_path, "ret_overnight_data.csv"), index=False, float_format="%.7f")
    intraday[cols].to_csv(os.path.join(ret_path, "ret_intraday_data.csv"), index=False, float_format="%.7f")
    daily_ret[cols].to_csv(os.path.join(ret_path, "ret_daily_data.csv"), index=False, float_format="%.7f")

    print(f"saved to {ret_path}, permno count: {len(permnos_kept)}")


def get_taq_data_by_date(date_str):
    """Saves {date}_all_tickers_5min.csv: sym_root x bar_time matrix of 5-min log
    returns for one trading day. 30-min is derived on-the-fly inside get_fret_ym."""
    file_name = f"{date_str}_all_tickers_5min.csv"
    file_path = os.path.join(hf_5min_path, file_name)
    if os.path.exists(file_path):
        print(f"{file_name} exists, skip download.")
        return
    
    conn = lazy_wrds_conn()

    yyyymmdd = date_str.replace('-', '')
    year = yyyymmdd[:4]

    sql = f"""
    WITH trades AS (
        SELECT
            sym_root,
            sym_suffix,
            time_m,
            price,
            date_bin(
                INTERVAL '5 minutes',
                date + time_m,
                date + TIME '09:30:00'
            ) AS bucket
        FROM taqm_{year}.ctm_{yyyymmdd}
        WHERE tr_corr IN ('00','01')
          AND (tr_scond IS NULL OR tr_scond !~ '[BCGHLMNOPQRUVWZ456789]')
          AND ex IN ('A','N','P','Q','T')
          AND time_m >= '09:30:00'
          AND time_m <  '16:00:00'
          AND date IS NOT NULL
          AND time_m IS NOT NULL
    )

    SELECT DISTINCT ON (sym_root, sym_suffix, bucket)
        sym_root,
        sym_suffix,
        to_char(bucket + INTERVAL '5 minutes', 'HH24:MI') AS bar_time,
        price
    FROM trades
    ORDER BY
        sym_root,
        sym_suffix,
        bucket,
        time_m DESC;
    """

    df = conn.raw_sql(sql)
    df['sym_suffix'] = df['sym_suffix'].fillna('__NULL__')
    counts = df.groupby(['sym_root','sym_suffix'])['bar_time'].transform('size')
    df = df[counts >= HF_MIN_BAR_COUNT]
    df_wide = df.pivot_table(
        index=["sym_root", "sym_suffix"],
        columns="bar_time",
        values="price",
        aggfunc="last"
    ).sort_index(axis=1)
    df_wide = df_wide.ffill(axis=1)
    # 5-min log returns. diff's first column (label '09:35') has no prior bar so it's NaN;
    # iloc[:, 1:] drops that column. fillna(0) treats remaining missing 5-min bars as 0
    # (untraded). The 30-min aggregation (rolling-6 sum) is done in get_fret_ym.
    df_ret5 = np.log(df_wide).diff(axis=1).iloc[:, 1:].fillna(0)

    df_ret5 = df_ret5.reset_index(drop=False)
    df_ret5['sym_suffix'] = df_ret5['sym_suffix'].replace('__NULL__', np.nan)
    df_ret5.to_csv(file_path, index=False)
    print(f"saved to {file_path} with {df_ret5['sym_root'].nunique()} unique sym_roots")


def get_sp500_list_ym(start_year=2006, end_year=2025):
    """Saves sp500_list_ym.csv: ym x permno S&P 500 membership panel.
    Columns: ym_mbr_start/end, full_month, mbr_flag (added/deleted/continued)."""
    out_path = os.path.join(spy_path, "sp500_list_ym.csv")
    if os.path.exists(out_path):
        print("sp500_list_ym.csv exists, skip download.")
        return

    conn = lazy_wrds_conn()
    start_date = pd.to_datetime(f"{start_year}-01-01")
    end_date = pd.to_datetime(f"{end_year}-12-31")

    df = conn.raw_sql("""
        SELECT permno, mbrstartdt, mbrenddt
        FROM crspm.dsp500list_v2
        WHERE mbrflg = 'NORM'
    """, date_cols=['mbrstartdt', 'mbrenddt'])

    df = df[df['mbrstartdt'].notna()].copy()
    df['mbrenddt'] = df['mbrenddt'].fillna(end_date)
    df = df[df['mbrenddt'] >= start_date].copy()
    df = df[df['mbrstartdt'] <= end_date].copy()

    ym_range = pd.date_range(start_date, end_date, freq='ME')
    ym_df = pd.DataFrame({
        'ym': ym_range.year * 100 + ym_range.month,
        'month_first': ym_range - pd.offsets.MonthBegin(1),
        'month_last': ym_range
    })

    df['_k'] = 1
    ym_df['_k'] = 1
    merged = df.merge(ym_df, on='_k').drop(columns=['_k'])
    overlap = (merged['month_first'] <= merged['mbrenddt']) & (merged['month_last'] >= merged['mbrstartdt'])
    merged = merged[overlap]
    merged['ym_mbr_start'] = merged[['mbrstartdt', 'month_first']].max(axis=1)
    merged['ym_mbr_end'] = merged[['mbrenddt', 'month_last']].min(axis=1)

    out = merged.groupby(['permno', 'ym']).agg({'ym_mbr_start': 'min', 'ym_mbr_end': 'max'}).reset_index()
    month_first = pd.to_datetime(out['ym'].astype(str), format='%Y%m')
    month_last = month_first + pd.offsets.MonthEnd(0)
    out['full_month'] = ((out['ym_mbr_start'].dt.normalize() == month_first) & (out['ym_mbr_end'] == month_last)).astype(int)
    is_added = out['ym_mbr_start'].dt.normalize() > month_first
    is_deleted = out['ym_mbr_end'] < month_last
    out['mbr_flag'] = np.where(is_added & is_deleted, 'temporary',
                               np.where(is_added, 'added', np.where(is_deleted, 'deleted', 'continued')))
    out = out[['ym', 'permno', 'ym_mbr_start', 'ym_mbr_end', 'full_month', 'mbr_flag']]
    out = out[out['ym'] >= start_year * 100 + 1].sort_values(['ym', 'permno']).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"saved to {out_path}, {len(out):,} rows")


def get_taq_5min_by_year(year, sym_root, sym_suffix=None):
    conn = lazy_wrds_conn()
    suffix_filter = "AND sym_suffix IS NULL" if sym_suffix is None else f"AND sym_suffix = '{sym_suffix}'"
    sql = f"""
    WITH trades AS (
        SELECT
            date,
            time_m,
            price,
            date_bin(
                INTERVAL '5 minutes',
                date + time_m,
                date + TIME '09:30:00'
            ) AS bucket
        FROM taqm_{year}.ctm_{year}
        WHERE sym_root = '{sym_root}'
          {suffix_filter}
          AND tr_corr IN ('00','01')
          AND (tr_scond IS NULL OR tr_scond !~ '[BCGHLMNOPQRUVWZ456789]')
          AND ex IN ('A','N','P','Q','T')
          AND time_m >= '09:30:00'
          AND time_m <  '16:00:00'
          AND date IS NOT NULL
          AND time_m IS NOT NULL
    )
    SELECT DISTINCT ON (date, bucket)
        date,
        bucket + INTERVAL '5 minutes' AS bar_ts,
        price
    FROM trades
    ORDER BY date, bucket, time_m DESC
    """
    return conn.raw_sql(sql)


def get_spy_5min_data(start_year=2006, end_year=2025):
    """Saves SPY_5min_fret.csv: datetime x SPY, 5min log returns for SPY."""
    out_path = os.path.join(spy_path, "SPY_5min_fret.csv")
    if os.path.exists(out_path):
        print("SPY_5min_fret.csv exists, skip download.")
        return

    parts = []
    for year in range(start_year, end_year + 1):
        print(f"SPY 5min {year} querying...")
        df = get_taq_5min_by_year(year, 'SPY', sym_suffix=None)
        if df.empty:
            continue
        df = df.sort_values(['date', 'bar_ts'])
        df['datetime'] = pd.to_datetime(df['bar_ts'])
        # Diff per trading date: full-series diff would put overnight returns into the
        # first bar of each day, contaminating downstream RV/BV and TOD estimates.
        df['ret'] = df.groupby('date')['price'].transform(lambda x: np.log(x).diff())
        df = df[['datetime', 'ret']].rename(columns={'ret': 'SPY'})
        df = df.dropna(subset=['SPY'])
        parts.append(df)

    spy_fret = pd.concat(parts, ignore_index=True)
    spy_fret.to_csv(out_path, index=False)
    print(f"saved to {out_path}, {len(spy_fret):,} rows")


def get_truncated_rets(data, alpha=4, omega=0.49):
    """Jump truncation: per-day RV/BV (21d lagged rolling) -> threshold -> truncation flag."""
    data = data.copy()
    data.columns = ['datetime', 'ret']
    data['date'] = pd.to_datetime(data['datetime']).dt.strftime('%Y%m%d').astype(int)
    data['ret_lag'] = data.groupby('date')['ret'].shift(1)
    data['ret2'] = data['ret'] ** 2
    data['ret2_bv'] = data['ret'].abs() * data['ret_lag'].abs()
    Vols = data.groupby('date').ret2.sum().to_frame().rename(columns={'ret2': 'RV'})
    # bfill needed for init ~21 days: Python access to taqm starts in start_year, no warm-up.
    Vols['RV'] = Vols['RV'].rolling(21).mean().shift(1).ffill().bfill()
    Vols['BV'] = data.groupby('date').ret2_bv.sum() * np.pi / 2
    Vols['BV'] = Vols['BV'].rolling(21).mean().shift(1).ffill().bfill()
    Vols['RV_BV_min'] = Vols[['RV', 'BV']].min(axis=1)
    Vols = Vols.reset_index()
    data = pd.merge(data, Vols, on='date', how='left')
    data['ret2_threshold'] = (alpha ** 2) * data['RV_BV_min'] * (1 / INTRADAY_BARS_PER_DAY) ** (omega * 2)
    data['truncation'] = (data['ret2'] < data['ret2_threshold']).astype(int)
    data['ret2_trunc'] = data['ret2'] * data['truncation']
    return data, Vols


def _stk_to_cret(stk_df, tod, alpha=4, omega=0.49):
    if stk_df.shape[0] < 100:  # skip near-empty stock-months (a full month has ~1600 5-min bars)
        return None
    stk_df = stk_df.sort_values("datetime").copy()
    data, _ = get_truncated_rets(stk_df[["datetime", "ret"]], alpha=alpha, omega=omega)
    data["minute_index"] = pd.to_datetime(data["datetime"]).dt.hour * 60 + pd.to_datetime(data["datetime"]).dt.minute
    data = data.merge(tod, on="minute_index", how="left")
    data["truncation"] = (data["ret2"] < data["ret2_threshold"] * data["tod_smooth"]).astype(int)
    data["ret_c"] = data["ret"] * data["truncation"]
    data = data[data.minute_index > 575][["datetime", "ret_c"]].rename(columns={"ret_c": "ret"})
    return data


def get_spy_tod(start_year=2006, end_year=2025, save_fig=True):
    """Saves SPY_daily_trailmonth_RV_lagged.csv (daily RV),
    tod_smooth.csv (minute_index x tod_smooth),
    and SPY_5min_cret.csv (truncated 5min returns).
    Time-of-day smoothing follows Bollerslev-Todorov-Li (2013)."""
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess

    get_spy_5min_data(start_year, end_year)
    spy_fRet = pd.read_csv(os.path.join(spy_path, "SPY_5min_fret.csv"))
    spy_fRet['datetime'] = pd.to_datetime(spy_fRet['datetime'])

    spy_data, Vols = get_truncated_rets(spy_fRet)
    Vols.to_csv(os.path.join(spy_path, "SPY_daily_trailmonth_RV_lagged.csv"), index=False)

    spy_data['year'] = spy_data['date'] // 10000
    spy_data['minute_index'] = (
        pd.to_datetime(spy_data['datetime']).dt.hour * 60
        + pd.to_datetime(spy_data['datetime']).dt.minute
    )

    tods = []
    for year in spy_data['year'].unique():
        sub = spy_data[spy_data['year'] == year]
        tod_temp = sub.groupby('minute_index').ret2_trunc.sum() / sub.ret2_trunc.sum() * INTRADAY_BARS_PER_DAY
        tod_temp = tod_temp.to_frame().reset_index().rename(columns={'ret2_trunc': 'tod'})
        tod_temp['tod_smooth'] = lowess(tod_temp.tod.values, (tod_temp.minute_index.values - 580) / 390)[:, 1]
        tod_temp['year'] = year
        tods.append(tod_temp)

    tod = pd.concat(tods, ignore_index=True)
    tod_results = tod.groupby('minute_index').tod_smooth.mean().to_frame()
    tod_results.to_csv(os.path.join(spy_path, "tod_smooth.csv"))

    if save_fig:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for t in tods:
            ax.plot(t['minute_index'], t['tod'], label=str(int(t['year'].iloc[0])))
        ax.plot(tod_results.index, tod_results['tod_smooth'], label="unconditional", color='black', linewidth=2)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlabel("minute_index")
        plt.tight_layout()
        plt.savefig(os.path.join(spy_path, "tod_smooth.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

    spy_data = pd.merge(spy_data, tod_results, on='minute_index', how='left')
    spy_data['truncation'] = (spy_data['ret2'] < spy_data['ret2_threshold'] * spy_data['tod_smooth']).astype(int)
    spy_data['ret_c'] = spy_data['ret'] * spy_data['truncation']

    spy_cRet = spy_data.loc[spy_data.minute_index > 575, ['datetime', 'ret_c']].rename(columns={'ret_c': 'SPY'})
    spy_cRet.to_csv(os.path.join(spy_path, "SPY_5min_cret.csv"), index=False)
    print(f"saved SPY cret to {spy_path}")


def get_fret_ym(ym_str, files_for_ym, ym_map_5, ym_map_30=None):
    """Returns 5min (datetime, permno, ret) long DataFrame for one month (filtered to
    ym_map_5 universe); saves fret30min_{ym_str}.csv as a side effect (filtered to
    ym_map_30 universe). 30-min is built by rolling-6 sum on the 5-min bars."""
    bar_cols_5  = [f"{t//60:02d}:{t%60:02d}" for t in range(9*60+40, 16*60+1, 5)]
    bar_cols_30 = [c for c in [f"{t//60:02d}:{t%60:02d}" for t in range(10*60+30, 16*60+1, 30)]
                   if c[-2:] in {"00", "30"}]
    if ym_map_30 is None:
        ym_map_30 = ym_map_5

    out_path_30 = os.path.join(fret30min_ym_path, f"fret30min_{ym_str}.csv")
    save_30 = not os.path.exists(out_path_30)

    parts_5  = []
    parts_30 = []

    for date_str, f in sorted(files_for_ym, key=lambda x: x[0]):
        df5 = pd.read_csv(f)
        df5["sym_suffix"] = df5["sym_suffix"].fillna("")
        missing = [c for c in bar_cols_5 if c not in df5.columns]
        if missing:
            print(f"  Skip {f.name}: missing bars (circuit breaker): {missing}")
            continue

        # Merge and apply date-aware namechg filter (shared logic for both freqs)
        def _merge_and_filter(df, bar_cols, ym_map):
            merged = df[["sym_root", "sym_suffix"] + bar_cols].merge(
                ym_map, on=["sym_root", "sym_suffix"], how="inner"
            )
            valid = (
                merged["namechg_str"].isna()
                | (merged["is_old"] & (date_str < merged["namechg_str"]))
                | (~merged["is_old"] & (date_str >= merged["namechg_str"]))
            )
            out = merged[valid][["permno"] + bar_cols].drop_duplicates()
            dup = out[out.duplicated(subset=["permno"], keep=False)]
            if not dup.empty:
                sample = (
                    dup[["permno"]]
                    .value_counts()
                    .rename("n_rows")
                    .reset_index()
                    .head(10)
                    .to_string(index=False)
                )
                raise ValueError(
                    f"Non-unique TAQ symbol mapping for {ym_str} on {date_str}. "
                    f"Expected one row per permno after date/name-change filtering. Sample:\n{sample}"
                )
            return out

        m5 = _merge_and_filter(df5, bar_cols_5, ym_map_5)
        long5 = m5.melt(id_vars=["permno"], value_vars=bar_cols_5, var_name="bar_time", value_name="ret")
        long5["datetime"] = pd.to_datetime(date_str + " " + long5["bar_time"])
        parts_5.append(long5[["datetime", "permno", "ret"]])

        if save_30:
            # Derive 30-min from 5-min in-place: rolling-6 sum on the 5-min bars.
            # For each kept 30-min bar (10:30+), the rolling window only consumes
            # 5-min bars at 10:05+, so it doesn't depend on the "09:35" padding bar.
            df30 = df5[["sym_root", "sym_suffix"]].copy()
            df30[bar_cols_5] = df5[bar_cols_5].T.rolling(window=6).sum().T
            present_30 = [c for c in bar_cols_30 if c in df30.columns]
            if present_30:
                m30 = _merge_and_filter(df30, present_30, ym_map_30)
                long30 = m30.melt(id_vars=["permno"], value_vars=present_30, var_name="bar_time", value_name="ret")
                long30["datetime"] = pd.to_datetime(date_str + " " + long30["bar_time"])
                parts_30.append(long30[["datetime", "permno", "ret"]])

    if save_30 and parts_30:
        month_long_30 = pd.concat(parts_30, ignore_index=True).dropna(subset=["ret"])
        pivot_30 = month_long_30.pivot_table(
            index="datetime", columns="permno", values="ret", aggfunc="last"
        )
        pivot_30.columns = [str(c) for c in pivot_30.columns]
        pivot_30.index.name = "datetime"
        pivot_30.to_csv(out_path_30)

    if not parts_5:
        return pd.DataFrame(columns=["datetime", "permno", "ret"])
    return pd.concat(parts_5, ignore_index=True).dropna(subset=["ret"])


def get_sp500_cRet_and_cBetas(start_year=2006, end_year=2025):
    """Saves cRetData_{ym}.csv (monthly 5min truncated returns, datetime x permno)
    and sp500_monthly_cBetas_lagged.csv (permno x yyyymm cBeta)."""
    from collections import defaultdict

    cBeta_path = os.path.join(data_path, "sp500_monthly_cBetas_lagged.csv")
    if os.path.exists(cBeta_path):
        print(f"{os.path.basename(cBeta_path)} exists, skip.")
        return

    get_crsp_taq_monthly_info(start_year, end_year)
    get_sp500_list_ym(start_year, end_year)
    get_spy_5min_data(start_year, end_year)
    get_spy_tod(start_year, end_year, save_fig=False)

    sp500_ym = pd.read_csv(os.path.join(spy_path, "sp500_list_ym.csv"))
    sp500_ym = sp500_ym[(sp500_ym.ym >= start_year * 100 + 1) & (sp500_ym.ym <= end_year * 100 + 12)].copy()
    sp500_permno_set = set(sp500_ym["permno"].astype(int).unique())
    sp500_pairs = sp500_ym[["ym", "permno"]].drop_duplicates().copy()

    crsp_taq = pd.read_csv(os.path.join(data_path, "crsp_taq_monthly_info.csv"), low_memory=False)
    crsp_taq = crsp_taq[
        (crsp_taq.ym >= start_year * 100 + 1) & (crsp_taq.ym <= end_year * 100 + 12)
    ].copy()
    crsp_taq = _drop_exact_duplicates(crsp_taq, "crsp_taq_monthly_info (sp500 cRet)")
    _assert_unique_keys(crsp_taq, ["permno", "ym"], "crsp_taq_monthly_info (sp500 cRet)")
    crsp_taq["sym_suffix"]  = crsp_taq["sym_suffix"].fillna("")
    crsp_taq["sym_suffix0"] = crsp_taq["sym_suffix0"].fillna("")
    crsp_taq["namechg"]     = pd.to_datetime(crsp_taq["namechg"], errors="coerce")
    crsp_taq["namechg_str"] = crsp_taq["namechg"].dt.strftime("%Y-%m-%d")

    map_current = crsp_taq[["sym_root", "sym_suffix", "ym", "permno", "namechg_str"]].copy()
    map_current["is_old"] = False
    renamed = crsp_taq[crsp_taq["namechg"].notna() & crsp_taq["sym_root0"].notna()].copy()
    if not renamed.empty:
        map_old = renamed[["sym_root0", "sym_suffix0", "ym", "permno", "namechg_str"]].rename(
            columns={"sym_root0": "sym_root", "sym_suffix0": "sym_suffix"}
        ).copy()
        map_old["is_old"] = True
        permno_map_all = pd.concat([map_current, map_old], ignore_index=True)
    else:
        permno_map_all = map_current
    permno_map_all = _drop_exact_duplicates(permno_map_all, "full symbol map")

    permno_map = permno_map_all.merge(sp500_pairs.assign(_in_sp500=1), on=["ym", "permno"], how="inner")
    permno_map = _drop_exact_duplicates(permno_map, "sp500 symbol map")
    _assert_unique_keys(permno_map, ["ym", "sym_root", "sym_suffix", "is_old"], "sp500 symbol map")

    spy_cret = pd.read_csv(os.path.join(spy_path, "SPY_5min_cret.csv"))
    spy_cret["datetime"] = pd.to_datetime(spy_cret["datetime"])
    spy_cret["yyyymm"] = spy_cret["datetime"].dt.year * 100 + spy_cret["datetime"].dt.month

    tod = pd.read_csv(os.path.join(spy_path, "tod_smooth.csv"))

    hf_files = sorted(Path(hf_5min_path).glob("*_all_tickers_5min.csv"))
    if not hf_files:
        raise FileNotFoundError(f"No hf_5min files in {hf_5min_path}. Run get_taq_data_by_date for trading days first.")

    files_by_ym = defaultdict(list)
    for f in hf_files:
        stem = f.stem.replace("_all_tickers_5min", "")
        if "-" not in stem or len(stem) < 7:
            continue
        year = int(stem[:4])
        if year < start_year or year > end_year:
            continue
        ym = year * 100 + int(stem[5:7])
        files_by_ym[ym].append((stem, f))

    # Per-month: call get_fret_ym (saves 30min CSV, returns 5min long), filter to S&P500
    parts = []
    for ym in sorted(files_by_ym.keys()):
        year, month = ym // 100, ym % 100
        ym_str = f"{year:04d}-{month:02d}"
        ym_map = permno_map[permno_map["ym"] == ym][
            ["sym_root", "sym_suffix", "permno", "namechg_str", "is_old"]
        ]
        ym_map_30 = permno_map_all[permno_map_all["ym"] == ym][
            ["sym_root", "sym_suffix", "permno", "namechg_str", "is_old"]
        ]
        _assert_unique_keys(ym_map, ["sym_root", "sym_suffix", "is_old"], f"sp500 symbol map for {ym_str}")
        if ym_map.empty:
            continue
        long5 = get_fret_ym(ym_str, files_by_ym[ym], ym_map, ym_map_30=ym_map_30)
        if long5.empty:
            continue
        long5 = long5[long5["permno"].isin(sp500_permno_set)]
        if not long5.empty:
            parts.append(long5)

    if not parts:
        print("  No 5min data found. Aborting.")
        return

    hf_long = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["datetime", "permno"])
    print(f"  fret5min: {len(hf_long):,} rows loaded (S&P 500 only)")

    grouped_hf = hf_long.groupby("permno")
    n_permnos = len(grouped_hf)
    print(f"  Computing cRet for {n_permnos} permnos...")
    cret_parts = []
    for i, (p, stk) in enumerate(grouped_hf):
        out = _stk_to_cret(stk[["datetime", "ret"]], tod)
        if out is not None:
            out["permno"] = p
            cret_parts.append(out)
        if (i + 1) % 200 == 0:
            print(f"  cRet: {i+1}/{n_permnos} permnos")
    cRet_all = pd.concat(cret_parts, ignore_index=True)
    cRet_all["datetime"] = pd.to_datetime(cRet_all["datetime"])
    dt = cRet_all["datetime"]
    cRet_all["yyyymm"] = dt.dt.year * 100 + dt.dt.month

    continued = sp500_ym[sp500_ym.mbr_flag == "continued"][["ym", "permno"]].drop_duplicates()
    print("  Saving cRetData by month...")
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yyyymm = year * 100 + month
            ym_str = f"{year:04d}-{month:02d}"
            permnos_ym = continued[continued.ym == yyyymm].permno.astype(int).tolist()
            cr_ym = cRet_all[(cRet_all.yyyymm == yyyymm) & (cRet_all.permno.isin(permnos_ym))]
            if cr_ym.empty:
                continue
            cr_pivot = cr_ym.pivot(index="datetime", columns="permno", values="ret").reset_index()
            spy_ym = spy_cret[spy_cret["yyyymm"] == yyyymm][["datetime", "SPY"]]
            cr_pivot = cr_pivot.merge(spy_ym, on="datetime", how="inner")
            permno_cols = sorted([c for c in cr_pivot.columns if c not in ["datetime", "SPY"]], key=lambda x: int(x) if str(x).isdigit() else x)
            cr_pivot = cr_pivot[["datetime", "SPY"] + permno_cols]
            cr_pivot.to_csv(os.path.join(spy_c_hf_5min_ym_path, f"cRetData_{ym_str}.csv"), index=False)
    print(f"  cRet saved to {spy_c_hf_5min_ym_path}")

    print("  Preparing cBeta inputs (align cRet with SPY)...")
    cRet_all["SPY"] = cRet_all["datetime"].map(spy_cret.set_index("datetime")["SPY"])
    # SPY-NaN rows are excluded via groupby.sum's skipna default; groups where all
    # SPY=NaN sum to spy_masked_sq=0 → cBeta=NaN → dropped by the .notna() filter.
    cRet_all["spy_ret"] = cRet_all["SPY"] * cRet_all["ret"]
    cRet_all["spy_masked_sq"] = np.where(cRet_all["ret"] != 0, cRet_all["SPY"] ** 2, 0.0)

    print("  Computing cBetas (groupby sum)...")
    cBeta_df = cRet_all.groupby(["permno", "yyyymm"], as_index=False)[["spy_ret", "spy_masked_sq"]].sum()
    cBeta_df["cBeta"] = np.where(cBeta_df["spy_masked_sq"] == 0, np.nan, cBeta_df["spy_ret"] / cBeta_df["spy_masked_sq"])
    cBeta_df = cBeta_df[["permno", "yyyymm", "cBeta"]]
    cBeta_df = cBeta_df[cBeta_df["cBeta"].notna() & np.isfinite(cBeta_df["cBeta"])]
    cBeta_df = cBeta_df.sort_values(["permno", "yyyymm"]).reset_index(drop=True)
    contemporaneous = cBeta_df["cBeta"].copy()
    cBeta_df["cBeta"] = cBeta_df.groupby("permno")["cBeta"].shift(1)
    # Use lag-1 cBeta by construction; only the first available month for each permno
    # falls back to the contemporaneous estimate because no prior month exists.
    is_initial_month = cBeta_df["cBeta"].isna()
    cBeta_df.loc[is_initial_month, "cBeta"] = contemporaneous[is_initial_month]
    cBeta_df = cBeta_df[cBeta_df["cBeta"].notna() & np.isfinite(cBeta_df["cBeta"])]
    cBeta_df.to_csv(cBeta_path, index=False)
    print(f"  cBetas saved to {cBeta_path}")


def get_ff6_factors(start_year=2006):
    """Saves ff6_daily_returns.csv and ff6_monthly_returns.csv:
    Fama-French 6 factors (MKT_RF, SMB, HML, RMW, CMA, MOM) + RF + BAB."""
    import requests
    import io

    daily_path = os.path.join(ret_path, "ff6_daily_returns.csv")
    monthly_path = os.path.join(ret_path, "ff6_monthly_returns.csv")
    if os.path.exists(daily_path) and os.path.exists(monthly_path):
        print("ff6_daily_returns.csv and ff6_monthly_returns.csv exist, skip download.")
        return

    conn = lazy_wrds_conn()
    start_date = f"{start_year-1}-01-01"

    ff_daily = conn.raw_sql(f"""
        SELECT * FROM ff.fivefactors_daily
        WHERE date >= '{start_date}'
    """)
    ff_daily["date"] = pd.to_datetime(ff_daily["date"]).dt.strftime("%Y%m%d").astype(int)
    _ff_cols = {"mktrf": "MKT_RF", "smb": "SMB", "hml": "HML", "rmw": "RMW", "cma": "CMA", "umd": "MOM", "rf": "RF"}
    ff_daily = ff_daily.rename(columns={k: v for k, v in _ff_cols.items() if k in ff_daily.columns})

    ff_monthly = conn.raw_sql(f"""
        SELECT * FROM ff.fivefactors_monthly
        WHERE date >= '{start_date}'
    """)
    ff_monthly["date"] = pd.to_datetime(ff_monthly["date"]).dt.strftime("%Y%m")
    ff_monthly["yyyymm"] = ff_monthly["date"].astype(int)
    ff_monthly = ff_monthly.rename(columns={k: v for k, v in _ff_cols.items() if k in ff_monthly.columns}).drop(columns=["date", "year", "month", "dateff"], errors="ignore")

    def _get_bab_daily():
        url = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Daily.xlsx"
        resp = requests.get(url)
        resp.raise_for_status()
        with io.BytesIO(resp.content) as f:
            df = pd.read_excel(f, sheet_name=0, skiprows=18, engine="openpyxl")
        df = df.rename(columns={df.columns[0]: "date"})[["date", "USA"]]
        df = df[pd.to_datetime(df["date"], errors="coerce").notnull()].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").astype(int)
        return df.rename(columns={"USA": "BAB"}).reset_index(drop=True)

    def _get_bab_monthly():
        url = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx"
        resp = requests.get(url)
        resp.raise_for_status()
        with io.BytesIO(resp.content) as f:
            df = pd.read_excel(f, sheet_name=0, skiprows=18, engine="openpyxl")
        df = df.rename(columns={df.columns[0]: "date"})[["date", "USA"]]
        df = df[pd.to_datetime(df["date"], errors="coerce").notnull()].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        df["yyyymm"] = df["date"].astype(int) // 100
        df = df.rename(columns={"USA": "BAB"}).reset_index(drop=True)
        return df[["yyyymm", "BAB"]]

    ff_daily = pd.merge(ff_daily, _get_bab_daily(), on="date", how="left")
    ff_monthly = pd.merge(ff_monthly, _get_bab_monthly(), on="yyyymm", how="left")
    ff_monthly = ff_monthly[["yyyymm"] + [c for c in ff_monthly.columns if c != "yyyymm"]]

    ff_daily.to_csv(daily_path, index=False)
    ff_monthly.to_csv(monthly_path, index=False)
    print(f"saved to {ret_path}")


def get_monthly_lagged_characteristics(start_year=2006, end_year=2025):
    """Saves monthly_lagged_betas.csv: permno x yyyymm panel with final, no-suffix columns.

    Output columns:
        - permno, yyyymm, ticker, size, size_wrds, size_grp, Beta60m, BetaFP, beta_252d

    Source priority:
        - size: JKP (lagged me) by default, then WRDS size proxy (log of market cap in $B).
          lag_mktcap is in $M (rescaled in get_crsp_taq_monthly_info), so log(lag_mktcap/1000)
          gives log($B); this matches JKP Size_jkp = log(me/1000) since JKP `me` is in $M.
          size_wrds is kept as a separate reference column for data-quality checks.
        - Beta60m: JKP beta_60m (lagged). OAP Beta is NOT used because OAP uses EW market (ewretd).
          Missing values are filled by Beta60m_my.
        - BetaFP: JKP betabab_1260d (lagged), then BetaFP_oap, then BetaFP_my.
        - beta_252d: JKP beta_252d (lagged), then beta_252d_my.
    """
    out_path = os.path.join(data_path, "monthly_lagged_betas.csv")
    if os.path.exists(out_path):
        print(f"{os.path.basename(out_path)} exists, skip.")
        return
    cutoff_year = start_year
    get_crsp_taq_monthly_info(start_year, end_year)

    # 1. crsp_taq base panel
    crsp_taq = pd.read_csv(os.path.join(data_path, "crsp_taq_monthly_info.csv"), low_memory=False)
    crsp_taq = crsp_taq[["permno", "ym", "ticker", "lag_mktcap"]].copy()
    crsp_taq["permno"] = crsp_taq["permno"].astype(int)
    crsp_taq["size_wrds"] = np.log(crsp_taq["lag_mktcap"] / 1000)
    crsp_taq = crsp_taq.drop(columns=["lag_mktcap"])
    crsp_taq = crsp_taq[(crsp_taq.ym >= cutoff_year * 100 + 1) & (crsp_taq.ym <= end_year * 100 + 12)].copy()
    crsp_taq = crsp_taq.sort_values(["permno", "ym"]).drop_duplicates(["permno", "ym"], keep="last")

    # 2. OAP: BetaFP only (lagged), used as fallback between JKP and my
    openap = oap.OpenAP()
    oap_all = openap.dl_signal("pandas", ["BetaFP"])
    oap_all = oap_all[oap_all.yyyymm >= (cutoff_year - 1) * 100 + 12].sort_values(["permno", "yyyymm"])
    oap_all["BetaFP"] = oap_all.groupby("permno")["BetaFP"].shift(1)
    oap_all = oap_all[oap_all.yyyymm >= cutoff_year * 100 + 1]
    oap_all = oap_all[["permno", "yyyymm", "BetaFP"]].rename(columns={"BetaFP": "BetaFP_oap"})
    oap_all["permno"] = oap_all["permno"].astype(int)

    # 3. contrib.global_factor (JKP)
    conn = lazy_wrds_conn()
    jkp_end = f"{end_year}-12-31"
    jkp = conn.raw_sql(f"""
        SELECT permno, eom, size_grp, me, beta_60m, betabab_1260d, beta_252d
        FROM contrib.global_factor
        WHERE excntry='USA' AND permno IS NOT NULL
        AND eom >= '2004-12-01' AND eom <= '{jkp_end}'
    """) # Keep a broader source universe here; the effective analysis universe is pinned by
         # crsp_taq filters downstream, so extra upstream filters are avoided to reduce merge loss.
    jkp["permno"] = jkp["permno"].astype(int)
    jkp["yyyymm"] = pd.to_datetime(jkp["eom"]).dt.year * 100 + pd.to_datetime(jkp["eom"]).dt.month
    jkp = jkp.sort_values(["permno", "yyyymm"])
    jkp["Size_jkp"] = np.log(jkp["me"] / 1000)
    jkp["Size_jkp"] = jkp.groupby("permno")["Size_jkp"].shift(1)
    jkp["Beta60m_jkp"] = jkp.groupby("permno")["beta_60m"].shift(1)
    jkp["BetaFP_jkp"] = jkp.groupby("permno")["betabab_1260d"].shift(1)
    jkp["beta_252d_jkp"] = jkp.groupby("permno")["beta_252d"].shift(1)
    jkp = jkp[["permno", "yyyymm", "size_grp", "Size_jkp", "Beta60m_jkp", "BetaFP_jkp", "beta_252d_jkp"]]

    # 4. Beta60m_my, BetaFP_my, beta_252d_my
    beta60m_my = pd.DataFrame(columns=["permno", "yyyymm", "Beta60m_my"])
    beta_fp_my = pd.DataFrame(columns=["permno", "yyyymm", "BetaFP_my"])
    beta_252d_my = pd.DataFrame(columns=["permno", "yyyymm", "beta_252d_my"])
    daily_path = os.path.join(ret_path, "ret_daily_data.csv")
    ff_path = os.path.join(ret_path, "ff6_daily_returns.csv")

    if os.path.exists(daily_path) and os.path.exists(ff_path):
        print("  Computing Beta60m_my, BetaFP_my, beta_252d_my from saved data...")
        daily = pd.read_csv(daily_path, low_memory=False)
        ff = pd.read_csv(ff_path, low_memory=False)
        ff["date"] = ff["date"].astype(int)
        daily["date"] = daily["date"].astype(int)
        daily = daily.merge(ff[["date", "MKT_RF", "RF"]], on="date", how="inner")
        permno_cols = [c for c in daily.columns if str(c).isdigit()]

        long = daily.melt(
            id_vars=["date", "MKT_RF", "RF"],
            value_vars=permno_cols,
            var_name="permno",
            value_name="ret",
        ).dropna(subset=["ret"])
        long["yyyymm"] = (long["date"] // 100).astype(int)
        long["permno"] = long["permno"].astype(int)
        long["stk_excess"] = long["ret"] - long["RF"]
        long["mkt_excess"] = long["MKT_RF"]
        long = long.sort_values(["permno", "date"])

        # Beta60m_my: monthly CAPM beta from FF market factor (MKT_RF), then lag by one month
        ff["yyyymm"] = (ff["date"] // 100).astype(int)
        mkt_mon = ff.groupby("yyyymm")["MKT_RF"].apply(lambda x: (1 + x).prod() - 1).reset_index(name="mkt_excess")

        mon = long.groupby(["permno", "yyyymm"])["stk_excess"].apply(lambda x: (1 + x).prod() - 1).reset_index()
        mon = mon.merge(mkt_mon, on="yyyymm", how="left")
        mon = mon.sort_values(["permno", "yyyymm"]).reset_index(drop=True)

        def roll_beta(g):
            y = g["stk_excess"].values.astype(np.float64)
            x = g["mkt_excess"].values.astype(np.float64)
            b = _roll_beta_1D(y, x, 60, 20)
            return pd.Series(b, index=g.index)

        mon["Beta60m_my"] = mon.groupby("permno", group_keys=False).apply(roll_beta).values
        mon["Beta60m_my"] = mon.groupby("permno")["Beta60m_my"].shift(1)
        beta60m_my = mon[["permno", "yyyymm", "Beta60m_my"]].dropna(subset=["Beta60m_my"])

        def roll_betafp(g):
            g = g.sort_values("date").reset_index(drop=True)
            y = g["stk_excess"].values.astype(np.float64)
            x = g["mkt_excess"].values.astype(np.float64)
            raw = _roll_betafp_fp(y, x)
            return pd.DataFrame({"raw": raw}, index=g.index)

        fp_raw = long.groupby("permno", group_keys=False).apply(roll_betafp)
        long["BetaFP_my"] = fp_raw["raw"].values
        fp_mon = long.groupby(["permno", "yyyymm"])["BetaFP_my"].last().reset_index()
        fp_mon["BetaFP_my"] = fp_mon.groupby("permno")["BetaFP_my"].shift(1)
        beta_fp_my = fp_mon[["permno", "yyyymm", "BetaFP_my"]].dropna(subset=["BetaFP_my"])

        def roll_beta_252d(g):
            g = g.sort_values("date").reset_index(drop=True)
            y = g["stk_excess"].values.astype(np.float64)
            x = g["mkt_excess"].values.astype(np.float64)
            return pd.Series(_roll_beta_1D(y, x, 252, 120), index=g.index)

        long["beta_252d_my"] = long.groupby("permno", group_keys=False).apply(roll_beta_252d).values
        b252_mon = long.groupby(["permno", "yyyymm"])["beta_252d_my"].last().reset_index()
        b252_mon["beta_252d_my"] = b252_mon.groupby("permno")["beta_252d_my"].shift(1)
        beta_252d_my = b252_mon.dropna(subset=["beta_252d_my"])

    # 5. Merge all (left join on crsp_taq)
    print("  Merging...")
    out = crsp_taq.merge(
        jkp, left_on=["permno", "ym"], right_on=["permno", "yyyymm"], how="left"
    ).drop(columns=["yyyymm"], errors="ignore")
    out = out.merge(
        oap_all, left_on=["permno", "ym"], right_on=["permno", "yyyymm"], how="left"
    ).drop(columns=["yyyymm"], errors="ignore")
    out = out.merge(
        beta60m_my, left_on=["permno", "ym"], right_on=["permno", "yyyymm"], how="left"
    ).drop(columns=["yyyymm"], errors="ignore")
    out = out.merge(
        beta_fp_my, left_on=["permno", "ym"], right_on=["permno", "yyyymm"], how="left"
    ).drop(columns=["yyyymm"], errors="ignore")
    out = out.merge(
        beta_252d_my, left_on=["permno", "ym"], right_on=["permno", "yyyymm"], how="left"
    ).drop(columns=["yyyymm"], errors="ignore")
    out = out.rename(columns={"ym": "yyyymm"})

    # Final versions only (no suffix columns in output)
    out["size"] = out["Size_jkp"].combine_first(out["size_wrds"])
    out["Beta60m"] = out["Beta60m_jkp"].combine_first(out["Beta60m_my"])
    out["BetaFP"] = out["BetaFP_jkp"].combine_first(out["BetaFP_oap"]).combine_first(out["BetaFP_my"])
    out["beta_252d"] = out["beta_252d_jkp"].combine_first(out["beta_252d_my"])

    out = out[["permno", "yyyymm", "ticker", "size", "size_wrds", "size_grp", "Beta60m", "BetaFP", "beta_252d"]]
    out = out.sort_values(["yyyymm", "permno"]).reset_index(drop=True)

    out.to_csv(out_path, index=False)
    print(f"saved to {out_path}")


def get_more_jkp_chars_lagged(start_year=2006, end_year=2025):
    """Saves monthly_more_jkp_chars_lagged.csv: permno x yyyymm with selected JKP chars lagged by one month."""
    out_path = os.path.join(data_path, "monthly_more_jkp_chars_lagged.csv")
    if os.path.exists(out_path):
        print(f"{os.path.basename(out_path)} exists, skip.")
        return

    # Selected JKP characteristics from Jensen, T. I., Kelly, B., & Pedersen,
    # L. H. (2023), "Is there a replication crisis in finance?", Journal of
    # Finance 78(5), 2465-2518.
    #
    # We keep only signals from the low risk, seasonality, short-term reversal,
    # and momentum themes, with the following filters:
    # 1. Exclude accounting-based variables (slow-moving Compustat fundamentals):
    #    earnings_variability, ocfq_saleq_std, dbnetis_at, kz_index, lti_gr1a,
    #    pi_nix, sti_gr1a.
    # 2. Exclude signals with lookback horizon > 1 year. The only retained
    #    exceptions are betabab_1260d and seas_1_1na. Excluded >1y signals:
    #    beta_60m, corr_1260d, seas_2_5an, seas_6_10na, seas_6_10an,
    #    seas_11_15an, seas_11_15na, seas_16_20an.
    # 3. Exclude microcap liquidity proxies / zero-trade measures:
    #    zero_trades_21d, zero_trades_126d, zero_trades_252d.
    char_cols = [
        # Low risk theme.
        "beta_dimson_21d",
        "betabab_1260d",
        "betadown_252d",
        "ivol_capm_21d",
        "ivol_capm_252d",
        "ivol_ff3_21d",
        "ivol_hxz4_21d",
        "rmax1_21d",
        "rmax5_21d",
        "rvol_21d",
        "turnover_126d",
        # Seasonality theme.
        "coskew_21d",
        # Short-term reversal theme.
        "iskew_capm_21d",
        "iskew_ff3_21d",
        "iskew_hxz4_21d",
        "ret_1_0",
        "rmax5_rvol_21d",
        "rskew_21d",
        # Momentum theme. seas_1_1na is the only retained >1y exception here.
        "prc_highprc_252d",
        "resff3_6_1",
        "resff3_12_1",
        "ret_3_1",
        "ret_6_1",
        "ret_9_1",
        "ret_12_1",
        "seas_1_1na",
    ]

    conn = lazy_wrds_conn()
    select_cols = ", ".join(["permno", "eom"] + char_cols)
    jkp = conn.raw_sql(f"""
        SELECT {select_cols}
        FROM contrib.global_factor
        WHERE excntry='USA' AND permno IS NOT NULL
        AND eom >= '{start_year-2}-12-01'
        AND eom <= '{end_year}-12-31'
    """) # Keep a broader source universe here; the effective analysis universe is pinned by
         # crsp_taq filters downstream, so extra upstream filters are avoided to reduce merge loss.

    jkp["permno"] = jkp["permno"].astype(int)
    jkp["yyyymm"] = pd.to_datetime(jkp["eom"]).dt.year * 100 + pd.to_datetime(jkp["eom"]).dt.month
    jkp = jkp.sort_values(["permno", "yyyymm"]).reset_index(drop=True)

    jkp[char_cols] = jkp.groupby("permno")[char_cols].shift(1)
    out = jkp[["permno", "yyyymm"] + char_cols].copy()
    out.to_csv(out_path, index=False)
    print(f"saved to {out_path}, {len(out):,} rows")


def _get_fred_api_key(raise_if_missing=True):
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip()
    if raise_if_missing:
        raise RuntimeError("FRED_API_KEY not found. Add it to .env or set as environment variable.")
    return None


def _load_alfred_release_dates(release_id):
    """Fetch release dates for release_id (10=CPI, 46=PPI, 50=unemployment).

    Tries FRED JSON API first if FRED_API_KEY is available; falls back to
    ALFRED Excel download otherwise. Returns empty DataFrame if both fail.
    """
    def _parse(dates_df):
        dates_df["date"] = pd.to_datetime(dates_df["date"], errors="coerce").dt.normalize()
        return (
            dates_df[["date"]]
            .dropna()
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )

    # --- attempt 1: FRED JSON API ---
    api_key = _get_fred_api_key(raise_if_missing=False)
    if api_key:
        try:
            url = (
                f"https://api.stlouisfed.org/fred/release/dates"
                f"?release_id={release_id}&api_key={api_key}"
                f"&file_type=json&limit=10000&sort_order=asc"
            )
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            return _parse(pd.DataFrame(data["release_dates"]))
        except Exception as e:
            print(f"  [release {release_id}] FRED API failed ({type(e).__name__}: {e}); trying ALFRED Excel fallback.")

    # --- attempt 2: ALFRED Excel download ---
    try:
        url = f"https://alfred.stlouisfed.org/release/downloaddates?rid={release_id}&ff=xlsx"
        releases = pd.read_excel(url, sheet_name="Release Dates", usecols=["Release Dates"]).rename(
            columns={"Release Dates": "date"}
        )
        return _parse(releases)
    except Exception as e:
        print(
            f"  [release {release_id}] ALFRED Excel download also failed ({type(e).__name__}: {e}). "
            f"Returning empty — CPI/PPI/unemployment event flags will be missing for this run."
        )
        return pd.DataFrame(columns=["date"])


def _load_usmpd_fomc_events():
    # Source: https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/
    usmpd_url = "https://www.frbsf.org/wp-content/uploads/USMPD.xlsx"
    fomc_events = pd.read_excel(
        usmpd_url,
        sheet_name="Statements",
        usecols=["Date", "date_time", "Unscheduled"],
    ).rename(columns={"Date": "date", "Unscheduled": "unscheduled"})

    fomc_events["date"] = pd.to_datetime(fomc_events["date"], errors="coerce").dt.normalize()
    fomc_events["date_time"] = pd.to_datetime(fomc_events["date_time"], errors="coerce")
    fomc_events["unscheduled"] = pd.to_numeric(fomc_events["unscheduled"], errors="coerce")

    fomc_events = (
        fomc_events.dropna(subset=["date", "date_time", "unscheduled"])
        .drop_duplicates(subset=["date", "date_time"])
        .sort_values(["date", "date_time"])
        .reset_index(drop=True)
    )

    fomc_events["is_fomc"] = 1
    fomc_events["fomc_timestamp"] = fomc_events["date_time"].dt.strftime("%H:%M")
    fomc_events["fomc_scheduled"] = fomc_events["unscheduled"].eq(0).astype(int)
    return fomc_events[["date", "is_fomc", "fomc_timestamp", "fomc_scheduled"]]


def get_announcement_dates(start_year=2006, end_year=2025):
    """Saves announcement_dates.csv: event-day panel of earnings and macro announcement dates."""
    out_path = os.path.join(data_path, "announcement_dates.csv")
    if os.path.exists(out_path):
        print(f"{os.path.basename(out_path)} exists, skip.")
        return

    print(f"[1/7] loading S&P 500 membership ...")
    sp500_membership_path = os.path.join(spy_path, "sp500_list_ym.csv")
    if not os.path.exists(sp500_membership_path):
        get_sp500_list_ym(start_year, end_year)

    start_ts = pd.Timestamp(f"{start_year}-01-01")
    end_ts = pd.Timestamp(f"{end_year}-12-31")

    membership = pd.read_csv(sp500_membership_path)
    membership["ym_mbr_start"] = pd.to_datetime(membership["ym_mbr_start"]).dt.normalize()
    membership["ym_mbr_end"] = pd.to_datetime(membership["ym_mbr_end"]).dt.normalize()
    membership["permno"] = membership["permno"].astype(int)
    membership = membership.dropna(subset=["permno", "ym_mbr_start", "ym_mbr_end"]).copy()
    membership_intervals = (
        membership[["permno", "ym_mbr_start", "ym_mbr_end"]]
        .drop_duplicates()
        .sort_values(["permno", "ym_mbr_start", "ym_mbr_end"])
        .reset_index(drop=True)
    )

    print(f"[2/7] connecting to WRDS ...")
    conn = lazy_wrds_conn()
    print(f"[3/7] querying IBES earnings announcements {start_year}-{end_year} (this may take a minute) ...")
    earnings_sql = f"""
        SELECT DISTINCT
            COALESCE(NULLIF(TRIM(oftic), ''), ticker) AS ticker,
            cusip,
            fpedats,
            anndats_act,
            anntims_act
        FROM tr_ibes.statsum_epsus
        WHERE fiscalp LIKE 'Q%%'
          AND anndats_act IS NOT NULL
          AND fpedats IS NOT NULL
          AND anndats_act BETWEEN '{(start_ts - pd.Timedelta(days=7)).date()}' AND '{end_ts.date()}'
    """
    earnings_raw = conn.raw_sql(earnings_sql, date_cols=["fpedats", "anndats_act"])
    print(f"  IBES raw: {len(earnings_raw):,} rows")
    # statsum_epsus is not event-level, so collapse repeated summary rows before counting announcements.
    earnings_raw["fpedats"] = pd.to_datetime(earnings_raw["fpedats"], errors="coerce").dt.normalize()
    earnings_raw["anndats_act"] = pd.to_datetime(earnings_raw["anndats_act"], errors="coerce").dt.normalize()
    earnings_raw = earnings_raw.dropna(subset=["ticker", "cusip", "fpedats", "anndats_act"]).copy()
    earnings_raw["ticker"] = earnings_raw["ticker"].str.strip().str.upper()
    earnings_raw["cusip"] = earnings_raw["cusip"].str.strip()
    earnings_raw = earnings_raw.drop_duplicates(subset=["cusip", "fpedats", "anndats_act", "anntims_act"]).copy()
    # If one firm-date maps to multiple fiscal periods, drop it instead of choosing one period arbitrarily.
    conflict_counts = earnings_raw.groupby(["cusip", "anndats_act"])["fpedats"].nunique().rename("n_fpedats")
    earnings_raw = earnings_raw.merge(conflict_counts, on=["cusip", "anndats_act"], how="left")
    earnings_raw = earnings_raw.loc[earnings_raw["n_fpedats"].eq(1)].drop(columns="n_fpedats").copy()

    cal = mcal.get_calendar("NYSE")
    schedule = cal.schedule(
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=(end_ts + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    trading_dates = pd.Index(schedule.index.normalize())
    close_by_date = pd.Series(
        schedule["market_close"].dt.tz_convert("America/New_York").dt.strftime("%H%M%S").astype(int).values,
        index=trading_dates,
    )
    next_trading_date = pd.Series(trading_dates[1:].values, index=trading_dates[:-1])

    announce_time = pd.to_datetime(earnings_raw["anntims_act"], format="%H:%M:%S", errors="coerce")
    missing_time = announce_time.isna()
    announce_time.loc[missing_time] = pd.to_datetime(
        earnings_raw.loc[missing_time, "anntims_act"],
        format="%H:%M",
        errors="coerce",
    )
    earnings_raw["announce_time_num"] = pd.to_numeric(announce_time.dt.strftime("%H%M%S"), errors="coerce")
    earnings_raw = earnings_raw.loc[earnings_raw["anndats_act"].isin(trading_dates)].copy()
    earnings_raw["market_close_num"] = earnings_raw["anndats_act"].map(close_by_date)
    # Treat after-close announcements as next-trading-day events.
    after_close = earnings_raw["announce_time_num"].gt(earnings_raw["market_close_num"])
    earnings_raw["event_date"] = earnings_raw["anndats_act"]
    earnings_raw.loc[after_close, "event_date"] = earnings_raw.loc[after_close, "anndats_act"].map(next_trading_date)
    earnings_raw = earnings_raw.dropna(subset=["event_date"]).copy()
    earnings_raw = earnings_raw.loc[earnings_raw["event_date"].between(start_ts, end_ts)].copy()
    earnings_raw["ym"] = earnings_raw["event_date"].dt.year * 100 + earnings_raw["event_date"].dt.month
    # LEAD is grouped by calendar quarter of the adjusted event date.
    earnings_raw["report_q"] = earnings_raw["event_date"].dt.to_period("Q")

    print(f"[4/7] querying IBES-CRSP link table ...")
    # IBES<->CRSP via wrdsapps.ibcrsphist: ticker + sdate<=event_date<=edate. Score 0-6; keep all.
    ibcrsp_link = conn.raw_sql("""
        SELECT ticker, permno, sdate, edate, score
        FROM wrdsapps.ibcrsphist
        WHERE score <= 6
    """)
    ibcrsp_link = ibcrsp_link.dropna(subset=["permno", "ticker"]).copy()
    ibcrsp_link["permno"] = ibcrsp_link["permno"].astype(int)
    ibcrsp_link["ticker"] = ibcrsp_link["ticker"].str.strip().str.upper()
    ibcrsp_link["sdate"] = pd.to_datetime(ibcrsp_link["sdate"])
    ibcrsp_link["edate"] = pd.to_datetime(ibcrsp_link["edate"])

    earnings_mapped = earnings_raw.merge(ibcrsp_link, on="ticker", how="inner")
    earnings_mapped = earnings_mapped.loc[
        earnings_mapped["sdate"].le(earnings_mapped["event_date"])
        & earnings_mapped["edate"].ge(earnings_mapped["event_date"])
    ].copy()
    earnings_mapped = (
        earnings_mapped.sort_values(["ticker", "event_date", "score"])
        .drop_duplicates(["ticker", "fpedats", "event_date"], keep="first")
        # If the same (permno, fpedats, event_date) is reached via multiple tickers (rare:
        # ticker rename within the same event), keep the lowest-score match.
        .sort_values(["permno", "fpedats", "event_date", "score", "ticker"])
        .drop_duplicates(["permno", "fpedats", "event_date"], keep="first")
        .reset_index(drop=True)
    )
    earnings_mapped = earnings_mapped[["permno", "ticker", "fpedats", "event_date", "report_q"]].copy()

    sp500_earnings = earnings_mapped.merge(membership_intervals, on="permno", how="inner")
    sp500_earnings = sp500_earnings.loc[
        sp500_earnings["ym_mbr_start"].le(sp500_earnings["event_date"])
        & sp500_earnings["event_date"].le(sp500_earnings["ym_mbr_end"])
    ].copy()
    sp500_earnings = (
        sp500_earnings.sort_values(["permno", "report_q", "event_date", "ticker"])
        .drop_duplicates(["permno", "fpedats", "event_date"], keep="first")
        .reset_index(drop=True)
    )

    earnings_daily = (
        sp500_earnings.groupby("event_date", as_index=False)["permno"]
        .nunique()
        .rename(columns={"event_date": "date", "permno": "earnings_count"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    quarter_daily = (
        sp500_earnings.groupby(["report_q", "event_date"], as_index=False)["permno"]
        .nunique()
        .rename(columns={"event_date": "date", "permno": "quarter_earnings_count"})
    )
    quarter_daily["weekday"] = quarter_daily["date"].dt.dayofweek
    quarter_daily = quarter_daily.loc[quarter_daily["weekday"].isin([1, 2, 3])].copy()
    quarter_daily["week_start"] = quarter_daily["date"] - pd.to_timedelta(quarter_daily["date"].dt.weekday, unit="D")
    quarter_weekly = (
        quarter_daily.groupby(["report_q", "week_start"], as_index=False)["quarter_earnings_count"]
        .sum()
        .rename(columns={"quarter_earnings_count": "week_earnings_count"})
    )
    quarter_weekly = quarter_weekly.loc[quarter_weekly["week_earnings_count"].ge(50)].copy()
    # Following Chan and Marsh (2022), LEAD days are Tue-Thu in the first week of quarter q
    # with at least 50 S&P 500 announcers across that Tue-Thu week. Here q is the calendar quarter of event_date.
    # Reference: Chan, K. F., & Marsh, T. (2022). Asset pricing on earnings announcement days. Journal of Financial Economics, 144(3), 1022-1042.
    first_lead_week = quarter_weekly.groupby("report_q")["week_start"].min().rename("first_lead_week")
    lead_dates = quarter_daily.merge(first_lead_week, on="report_q", how="inner")
    lead_dates = lead_dates.loc[lead_dates["week_start"].eq(lead_dates["first_lead_week"]), ["date"]].drop_duplicates()
    lead_dates["lead_day"] = 1
    lead_daily = earnings_daily.merge(lead_dates, on="date", how="left")
    lead_daily["lead_day"] = lead_daily["lead_day"].fillna(0).astype(int)

    print(f"[5/7] fetching CPI / PPI / unemployment release dates from FRED ...")
    cpi_dates = _load_alfred_release_dates(10)
    cpi_dates = cpi_dates.loc[cpi_dates["date"].between(start_ts, end_ts)].reset_index(drop=True)
    cpi_dates["cpi_day"] = 1

    ppi_dates = _load_alfred_release_dates(46)
    ppi_dates = ppi_dates.loc[ppi_dates["date"].between(start_ts, end_ts)].reset_index(drop=True)
    ppi_dates["ppi_day"] = 1

    unrate_dates = _load_alfred_release_dates(50)
    unrate_dates = unrate_dates.loc[unrate_dates["date"].between(start_ts, end_ts)].reset_index(drop=True)
    unrate_dates["unrate_day"] = 1

    print(f"[6/7] fetching FOMC event dates ...")
    fomc_events = _load_usmpd_fomc_events()
    fomc_events = fomc_events.loc[fomc_events["date"].between(start_ts, end_ts)].reset_index(drop=True)
    fomc_dates = fomc_events.loc[
        fomc_events["fomc_scheduled"].eq(1),
        ["date", "fomc_timestamp", "fomc_scheduled"],
    ].drop_duplicates(subset=["date"]).copy()
    fomc_dates["fomc_day"] = 1

    # Full NYSE trading-day panel so downstream non-special-day filters can use a simple
    # inner merge without missing rows that genuinely have no event.
    all_trading_dates = pd.DataFrame({
        "date": trading_dates[(trading_dates >= start_ts) & (trading_dates <= end_ts)]
    }).sort_values("date").reset_index(drop=True)

    dates = all_trading_dates.merge(lead_daily, on="date", how="left")
    dates = dates.merge(cpi_dates, on="date", how="left")
    dates = dates.merge(ppi_dates, on="date", how="left")
    dates = dates.merge(unrate_dates, on="date", how="left")
    dates = dates.merge(fomc_dates, on="date", how="left")

    dates["earnings_count"] = dates["earnings_count"].fillna(0).astype(int)
    dates["fomc_timestamp"] = dates["fomc_timestamp"].fillna("")
    for col in ["lead_day", "cpi_day", "ppi_day", "unrate_day", "fomc_day", "fomc_scheduled"]:
        dates[col] = dates[col].fillna(0).astype(int)

    dates = dates[
        [
            "date",
            "lead_day",
            "earnings_count",
            "cpi_day",
            "ppi_day",
            "unrate_day",
            "fomc_day",
            "fomc_timestamp",
            "fomc_scheduled",
        ]
    ].sort_values("date").reset_index(drop=True)

    print(f"[7/7] writing output ...")
    dates.to_csv(out_path, index=False)
    print(f"  saved to {out_path}, {len(dates):,} rows")


def _rpna_jkp_map_cte(start_year, end_year):
    """SQL CTE: jkp_map (permno, rp_entity_id, valid_from, valid_to) for JKP US universe.
    Time-aware via crspm.wrds_msfv2_query (mthcaldt ranges) ∩
    rpna.rpa_company_mappings (range_start, range_end on CUSIP)."""
    return f"""
        WITH jkp_permnos AS (
            SELECT DISTINCT permno
            FROM contrib.global_factor
            WHERE excntry='USA' AND permno IS NOT NULL
              AND eom BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
        ),
        permno_cusip AS (
            SELECT c.permno, c.cusip9,
                   MIN(c.mthcaldt) AS pc_start,
                   MAX(c.mthcaldt) AS pc_end
            FROM crspm.wrds_msfv2_query c
            INNER JOIN jkp_permnos j ON c.permno = j.permno
            WHERE c.cusip9 IS NOT NULL
              AND c.mthcaldt BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
            GROUP BY c.permno, c.cusip9
        ),
        rp_cusip AS (
            SELECT rp_entity_id, data_value AS cusip9,
                   range_start,
                   COALESCE(range_end, DATE '2099-12-31') AS range_end
            FROM rpna.rpa_company_mappings
            WHERE data_type = 'CUSIP'
        ),
        jkp_map AS (
            SELECT pc.permno, rc.rp_entity_id,
                   GREATEST(pc.pc_start, rc.range_start)              AS valid_from,
                   LEAST   (pc.pc_end,   rc.range_end)                AS valid_to
            FROM permno_cusip pc
            INNER JOIN rp_cusip rc ON rc.cusip9 = pc.cusip9
            WHERE GREATEST(pc.pc_start, rc.range_start)
                <= LEAST  (pc.pc_end,   rc.range_end)
        )
    """


def get_rpna_entity_map(start_year=2006, end_year=2025):
    """Saves rpna_jkp_entity_map.csv: (permno, rp_entity_id, valid_from, valid_to)
    time-aware mapping for JKP US universe."""
    out_path = os.path.join(data_path, "rpna_jkp_entity_map.csv")
    if os.path.exists(out_path):
        print(f"{os.path.basename(out_path)} exists, skip.")
        return

    conn = lazy_wrds_conn()
    df = conn.raw_sql(f"""
        {_rpna_jkp_map_cte(start_year, end_year)}
        SELECT * FROM jkp_map
    """)
    df["permno"] = df["permno"].astype(int)
    df.to_csv(out_path, index=False)
    print(f"saved to {out_path}, {len(df):,} rows")


def get_rpna_jkp_news(start_year=2006, end_year=2025):
    """Saves rpna_jkp_news.parquet: granular RP DJ news for JKP US universe.
    SQL filters: provider=DJ, entity=COMP, country=US, relevance>=10,
    news_type NOT IN (RNS-SEC13F, RNS-SEC144, TABULAR-MATERIAL),
    event_similarity_days IS NULL OR >= 1 (novelty: keeps first-occurrence + 24h-distinct echoes).
    Time-aware permno join: rpa_date_utc ∈ [valid_from, valid_to] of jkp_map.
    Python dedup: drop_duplicates on (rp_story_id, rp_entity_id, rp_story_event_index, permno),
    keeping highest-relevance row (handles residual entity→permno expansion duplicates)."""
    out_path = os.path.join(data_path, "rpna_jkp_news.parquet")
    if os.path.exists(out_path):
        print(f"{os.path.basename(out_path)} exists, skip.")
        return

    conn = lazy_wrds_conn()
    map_cte = _rpna_jkp_map_cte(start_year, end_year)

    parts = []
    for year in range(start_year, end_year + 1):
        print(f"RPA {year} querying...")
        df = conn.raw_sql(f"""
            {map_cte}
            SELECT n.rp_story_id, n.rp_entity_id, m.permno,
                   n.rpa_date_utc, n.timestamp_utc,
                   n.relevance, n.event_sentiment_score, n.nip, n.css,
                   n.event_similarity_days,
                   n.topic, n."group" AS group_name, n.category,
                   n.rp_story_event_index, n.rp_story_event_count,
                   n.news_type, n.fact_level
            FROM rpna.rpa_full_equities_{year} n
            INNER JOIN jkp_map m
              ON n.rp_entity_id = m.rp_entity_id
             AND CAST(n.rpa_date_utc AS DATE) BETWEEN m.valid_from AND m.valid_to
            WHERE n.entity_type  = 'COMP'
              AND n.country_code = 'US'
              AND n.provider_id  = 'DJ'
              AND n.relevance   >= 10
              -- NULL = no prior similar event (novel, keep); >=1 = ≥24h since last echo.
              AND (n.event_similarity_days IS NULL OR n.event_similarity_days >= 1)
              AND n.news_type NOT IN ('RNS-SEC13F', 'RNS-SEC144', 'TABULAR-MATERIAL')
        """)
        print(f"  {year}: {len(df):,} rows")
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out["permno"] = out["permno"].astype(int)
    n_pre = len(out)
    out = (
        out.sort_values("relevance", ascending=False)
           .drop_duplicates(["rp_story_id", "rp_entity_id", "rp_story_event_index", "permno"])
           .sort_index()
           .reset_index(drop=True)
    )
    print(f"dedup: {n_pre:,} -> {len(out):,} rows")
    out.to_parquet(out_path, index=False)
    print(f"saved to {out_path}")
