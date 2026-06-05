import os

import pandas_market_calendars as mcal

from data import (
    ret_path,
    get_crsp_taq_monthly_info,
    get_ff6_factors,
    get_crsp_daily_returns,
    get_spy_tod,
    get_taq_data_by_date,
    get_sp500_list_ym,
    get_sp500_cRet_and_cBetas,
    get_monthly_lagged_characteristics,
    get_more_jkp_chars_lagged,
    get_announcement_dates,
    get_rpna_entity_map,
    get_rpna_jkp_news,
)

start_year = 2006
end_year = 2025
# CRSP monthly + daily return panels go one year further so we have OOS returns
# for the last month of the hf-data sample (e.g. OOS for ym=YYYY12 needs ret(YYYY+1, 01)).
ret_end_year = 2026
cal = mcal.get_calendar("NYSE")
start_date = f"{start_year}-01-01"
end_date = f"{end_year}-12-31"
schedule = cal.schedule(start_date=start_date, end_date=end_date)
early_closes = cal.early_closes(schedule)
full_days = schedule[~schedule.index.isin(early_closes.index)]
trading_days = full_days.index.strftime("%Y-%m-%d").tolist()

if __name__ == "__main__":
    print("*" * 30 + " Downloading CRSP-TAQ monthly info" + "*" * 30)
    get_crsp_taq_monthly_info(start_year, ret_end_year)

    print("*"*30+" Downloading FF6 factors"+ "*"*30)
    get_ff6_factors(start_year)

    print("*"*30+" Downloading CRSP daily returns"+ "*"*30)
    get_crsp_daily_returns(start_year, ret_end_year)

    print("*"*30+" Downloading SPY 5min + TOD"+ "*"*30)
    get_spy_tod(start_year, end_year)

    print("*"*30+" Downloading TAQ high-frequency data"+ "*"*30)
    for date in trading_days:
        get_taq_data_by_date(date)

    print("*"*30+" Downloading S&P 500 list by ym"+ "*"*30)
    get_sp500_list_ym(start_year, end_year)

    print("*"*30+" Building S&P500 cRet and cBetas from hf_5min"+ "*"*30)
    get_sp500_cRet_and_cBetas(start_year, end_year)

    print("*"*30+" Building monthly lagged characteristics"+ "*"*30)
    get_monthly_lagged_characteristics(start_year, end_year)

    print("*"*30+" Building more JKP lagged characteristics"+ "*"*30)
    get_more_jkp_chars_lagged(start_year, end_year)

    print("*"*30+" Building announcement dates"+ "*"*30)
    get_announcement_dates(start_year, end_year)

    print("*"*30+" Building RavenPack JKP entity map"+ "*"*30)
    get_rpna_entity_map(start_year, end_year)

    print("*"*30+" Downloading RavenPack JKP granular news"+ "*"*30)
    get_rpna_jkp_news(start_year, end_year)

    print("*" * 30 + " JKP factors (manual download)" + "*" * 30)
    print("Download from https://jkpfactors.com/factor-returns, place in Data/Returns/")
    for f in [
        "[usa]_[ivol_capm_21d]_[monthly]_[vw_cap].csv",
        "[usa]_[ivol_capm_252d]_[monthly]_[vw_cap].csv",
        "[usa]_[low_risk]_[monthly]_[vw_cap].csv",
        "[usa]_[momentum]_[monthly]_[vw_cap].csv",
        "[usa]_[seasonality]_[monthly]_[vw_cap].csv",
        "[usa]_[short_term_reversal]_[monthly]_[vw_cap].csv",        
    ]:
        status = "exists" if os.path.exists(os.path.join(ret_path, f)) else "NOT FOUND"
        print(f"  {f}: {status}")
