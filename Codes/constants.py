"""Project-wide constants. Leaf module — no project imports."""
from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────
data_path = str(Path(__file__).resolve().parent.parent / "Data") + "/"
hf_5min_path = data_path + "fRet5min/"
fret30min_ym_path = data_path + "fRet30min_ym/"
ret_path = data_path + "Returns/"
spy_path = data_path + "SPY/"
spy_c_hf_5min_ym_path = data_path + "sp500_cRet5min_ym/"
result_path = str(Path(data_path).parent / "Results")
plot_path = str(Path(data_path).parent / "Plots")
case_path = str(Path(data_path).parent / "CaseStudy")

def init_paths():
    for d in (result_path, plot_path, case_path):
        Path(d).mkdir(parents=True, exist_ok=True)

spy_5min_path = Path(spy_path) / "SPY_5min_fret.csv"
TIME_COLS_30 = ["10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
MINUTE_MAP = {tc: 630 + i * 30 for i, tc in enumerate(TIME_COLS_30)}

FREQ = 5
N_BARS_PER_DAY = 390 // FREQ - 1
KN_WINDOW_MINUTES = 120
KN = KN_WINDOW_MINUTES // FREQ
PCA_GLOBAL_FACTOR_FOLD = 5     # init: lowest 1/5 of S&P 500 by news proxy -> Rg0
RG_SELECT_FOLD = 10            # Rg = lowest 1/10 of the universe by lnf_exposure_proxy
MIN_MKTCAP_M = 200
MIN_CLOSE_PRICE = 5
N_BARS_30 = len(TIME_COLS_30)
RG_SELECT_SCALE = 0.8           # shrink_coef in residual; details in decomposition.py

# ── portfolio name labels (`port` column in long-format return panels) ──────
PORT_VWMKT   = "vwMKT"
PORT_BFP_D1  = "BetaFP_D1_vw"
PORT_BFP_D10 = "BetaFP_D10_vw"
PORT_RG0     = "Rg0"
PORT_RG      = "Rg"
PORT_MKTG    = "MKTg"
PORT_LNF     = "LNF"


# ── case-study spec (used by cases.py) ──────────────────────────────────────
CASE_SPEC = {
    "oil_shock": {
        "naics": 211,
        "event_date": "2014-11",
        # Keep upstream oil E&P firms.
        # Remove downstream/services firms: PSX, HP
        "remove": [13356, 32707],
    },
    "svb": {
        "naics": 522,
        "event_date": "2023-03",
        # Remove firms lacking continuous 3-month window data (t-1,t,t+1):
        # SIVB, SBNY (delisted)
        # Remove credit card / consumer finance firms (not deposit-run exposure):
        # SYF, AXP, COF, DFS
        "remove": [11786, 90090, 14776, 59176, 81055, 92121],
    },
    "deepseek": {
        "naics": 334413,
        "event_date": "2025-01",
        # Remove non-semiconductor firms sharing NAICS classification: ENPH, FSLR
        "remove": [13323, 91611],
    },
}

