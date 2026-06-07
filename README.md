# The News-Contaminated Market: Local Factors and Asset Pricing

Code for the empirical analysis in `The News-Contaminated Market: Local Factors and Asset Pricing`.

Paper link: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6480458)

The value-weighted market return mechanically aggregates two distinct sources of risk: a global component that affects all firms and local components that affect only subsets of firms. The code implements the decomposition of the market into a global factor and an orthogonal residual, the local news factor (`LNF`), using high-frequency data to capture the time-varying factor structure.

In the code:

- `Rg` denotes the unscaled global factor, defined as the value-weighted return of stocks in the lowest decile of $|\widehat{\gamma}_{i,t}|$, where $\widehat{\gamma}_{i,t}$ is each stock's proxy for aggregate exposure to local factors. It is constructed in two stages: $\ell_1$-rotated high-frequency PCA on the S&P 500, then realized-beta estimation over the broader liquid cross section. See Section 3.2 of the paper for detail.
- `MKTg` denotes the global component of the value-weighted market return, equal to $\widehat{\phi}\cdot R_g$, where $\widehat{\phi}$ is the projection coefficient of the market on $R_g$; by construction, $\text{vwMKT} = \text{MKTg} + \text{LNF}$.
- `LNF` denotes the residual component of the value-weighted market return orthogonal to the global factor, namely the local news factor.

The main empirical applications in the code are:

- intraday beta dispersion
- conditional security market line (SML) analysis
- in-sample spanning tests
- out-of-sample signal sorting
- beta-shock dynamics
- event case studies
- robustness sweeps and RavenPack news intensity diagnostics

High-frequency PCA, the $\ell_1$-rotation, the intraday beta-dispersion analyses, and the event case studies are restricted to S&P 500 stocks; the remaining exercises use the broader liquid cross section.

## Repository Layout

This repository is notebook-driven. The reusable logic lives in the following Python modules under `Codes/`:

- `data_loader.py`: end-to-end data build script
- `data.py`: WRDS pulls, public-source downloads, cleaning, and preprocessing utilities
- `pca.py`: monthly PCA, $\ell_1$-rotation, and stock-level local-factor identification
- `analysis_utils.py`: shared utilities for portfolio analysis, statistics, and table formatting
- `decomposition.py`: Rg / MKTg / LNF factor construction, benchmark portfolios, and reporting (perf tables, conditional stats, robustness sweep)
- `port_sort.py`: SML, HLR, beta-shock, and signal-sorting portfolio analyses
- `spanning.py`: in-sample spanning regressions and CAPM-vs-decomposition comparisons
- `cases.py`: event case-study driver
- `news.py`: RavenPack DJ news loader and news-intensity plot by β decile × size group
- `constants.py`: project paths and shared constants (port names, factor-construction coefficients, etc.)

Notebooks and paper mapping:

| Paper section | Corresponding notebook(s) | Notes |
| --- | --- | --- |
| Section 3, subsection `The global--local decomposition and factor construction` | `nb_beta_dispersion.ipynb`, `nb_insample_spanning.ipynb`, `nb_cases.ipynb` | `nb_beta_dispersion.ipynb` produces the monthly PCA outputs; `nb_insample_spanning.ipynb` implements the two-stage factor construction; `nb_cases.ipynb` provides local-cluster case illustrations |
| Section 4, subsection `Decomposition and portfolio performance` | `nb_insample_spanning.ipynb` | Builds `Rg`/`LNF` portfolios, performance summaries, and related plots/tables |
| Section 4, subsection `The slope of the SML` | `nb_sml.ipynb`, `nb_sml_subsample.ipynb`, `nb_sml_alt_conditional.ipynb`, `nb_capm_tales_HLR.ipynb` | Baseline SML, subsample SML, alternative conditioning variables, and intraday/overnight beta-sort evidence |
| Section 4, subsection `Spanning the low-risk anomalies` | `nb_insample_spanning.ipynb`, `nb_insample_spanning_bab.ipynb`, `nb_outofsample_sorting.ipynb` | In-sample spanning, BAB-specific regressions, and out-of-sample signal sorting |
| Section 4, subsection `Beta shock and the intraday beta dispersion` | `nb_beta_dispersion.ipynb`, `nb_beta_shock.ipynb` | Intraday dispersion facts and beta-shock/event-time analyses |
| RavenPack news intensity by β decile × size group | `nb_news_count.ipynb` | News-count / NIP / CSS aggregates per (size_grp, β decile) |
| Robustness (κ sweep) | `nb_robustness.ipynb` | Sensitivity of Rg / LNF Sharpe ratios and spanning α to the `RG_SELECT_SCALE` shrinkage coefficient |

The code expects the following sibling directories under the project root:

```text
L1PCA/
├── Codes/
├── Data/
├── Results/
├── Plots/
└── CaseStudy/
```

Paths in the code are built relative to the parent of `Codes`, so this layout matters.

## Conceptual Mapping

The empirical workflow follows the paper closely:

1. Build monthly high-frequency return panels for eligible S&P 500 stocks (5-minute) and the broader liquid cross section (30-minute).
2. Run PCA within each month on the 5-minute S&P 500 panel.
3. Apply the $\ell_1$-rotation to identify local factor loadings; aggregate row-wise to a stock-level local-exposure proxy `news_proxy` ($\widehat{\gamma}_{i,t}^{(0)}$ in the paper).
4. Form the initial global-factor proxy $g_t^{(0)}$ as the equal-weighted return of S&P 500 stocks in the lowest quintile of `news_proxy`. Construct the initial LNF as $\text{LNF}_t^{(0)} = r_{\text{vwMKT},t} - \kappa\,\widehat{\phi}_t^{(0)}\,g_t^{(0)}$, where $\widehat{\phi}_t^{(0)}$ is the high-frequency projection coefficient of vwMKT on $g_t^{(0)}$ and $\kappa=0.8$ is a shrinkage parameter (set below one because $g_t^{(0)}$ is recovered on a narrower universe and the downstream realized exposures are themselves noisy; Appendix B and Table tab:robustness in the paper analyze the choice and report empirical sensitivity). Across the broader liquid cross section, $\widehat{\gamma}_{i,t}$ is each stock's 30-minute realized beta on $\text{LNF}_t^{(0)}$. The global factor $R_g$ is the value-weighted return of stocks in the lowest decile of $|\widehat{\gamma}_{i,t}|$.
5. Define `LNF` as the residual of the value-weighted market with respect to `MKTg`, where `MKTg` is the projection of vwMKT on `Rg`.
6. Use these objects in SML, spanning, anomaly, and beta-shock exercises.

The cross section is further restricted to stock-months with non-missing lagged BetaFP (the Frazzini-Pedersen beta from JKP), so the BetaFP D1/D10 benchmark portfolios share the same monthly universe as Rg/MKTg/LNF.

## Data Requirements

The pipeline relies on a mix of WRDS pulls, public downloads, and a small manual step.

WRDS access is required for:

- CRSP monthly returns and market cap data
- TAQ intraday data
- S&P 500 membership/linking data
- IBES earnings announcement dates
- Fama-French factor tables on WRDS
- `contrib.global_factor` characteristic data
- RavenPack News Analytics (`rpna.rpa_full_equities_*` and `rpna.rpa_company_mappings`) for the news-intensity diagnostics

Public sources used by the code include:

- AQR BAB factor files
- Open Asset Pricing (`openassetpricing`)
- macro/FOMC release calendars used in `get_announcement_dates`
- FRED / ALFRED release dates (requires `FRED_API_KEY`; see Environment)

Manual download step:

- JKP factor return CSV files from [jkpfactors.com](https://jkpfactors.com/factor-returns)
- place them in `../Data/Returns/`

The expected JKP files are printed by `python data_loader.py`.

## Environment

The codebase is written in Python and uses notebooks for analysis. Core packages imported by the project include:

- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `statsmodels`
- `numba`
- `wrds`
- `pandas_market_calendars`
- `openassetpricing`
- `requests`
- `openpyxl`
- `jupyter`

There is no pinned environment file in this repository, so install these packages manually in your preferred environment.

`numba` is used in `pca.py` to JIT-compile the high-frequency PCA and $\ell_1$-rotation pipeline, with a parallel inner loop for the rotation search via `numba.prange`. The first invocation incurs a one-time compile cost cached for subsequent runs. The outer per-month loop is kept serial to avoid contention with BLAS threading inside the SVD.

A `.env` file placed at the project root (alongside `Codes/`) is read for optional API keys:

```text
FRED_API_KEY=<your key>
```

The FRED key is only required by the FRED/ALFRED release-date pull inside `data.py::get_announcement_dates`; absent a key, that step falls back to the ALFRED Excel download.

## Build Pipeline

To build the data artifacts from scratch, from the `Codes/` directory:

```bash
python data_loader.py
```

What this script does:

- downloads CRSP-TAQ monthly linking and return data
- downloads Fama-French daily and monthly factors plus BAB
- downloads SPY data and builds time-of-day adjustments
- pulls daily TAQ high-frequency files
- constructs monthly S&P 500 high-frequency return panels and lagged continuous betas
- builds lagged characteristics
- builds announcement-day flags
- builds the RavenPack entity map and granular DJ news parquet for the JKP US universe

Notes:

- the full TAQ pull is large and slow
- WRDS credentials are required
- some steps are skipped automatically if output files already exist
- the JKP factor-return files still need to be downloaded manually

## Recommended Execution Order

If you want to reproduce the main analysis from raw data, the clean order is:

1. Run `python data_loader.py`.
2. Run `nb_beta_dispersion.ipynb` to create the monthly PCA and dispersion output files.
3. Run `nb_insample_spanning.ipynb` to build portfolio-level return panels.
4. Run the downstream notebooks for SML, spanning, beta-shock, sorting, robustness, and news-intensity exercises.

If the cached result files already exist in `../Results/`, most notebooks can be run directly.

## Important Functions

Useful entry points across the modules:

- `pca.run_monthly_analysis(ym_list)`: runs the monthly PCA / sparse-rotation pipeline and writes result CSVs
- `pca.find_local_factors(...)`: $\ell_1$-rotation identification of local factors (see Note on the $\ell_1$ Rotation)
- `analysis_utils.load_port_analysis_data()`: loads the combined monthly inputs for portfolio analysis
- `decomposition.build_rg_factors(...)`: per-month Rg / MKTg / LNF construction and excess-return panel
- `decomposition.decompose_market_returns(...)`: single-month vwMKT → MKTg + LNF decomposition
- `decomposition.compute_robustness_at_kappa(...)`: κ-sweep driver behind `nb_robustness.ipynb`
- `port_sort.build_sml_panel(...)`: builds the daily panel used in SML tests
- `port_sort.get_beta_shock_panel()`: prepares the beta-shock analysis panel
- `spanning.compare_capm_spanning(...)`: CAPM vs Rg + LNF spanning regressions
- `cases.run_case_study(case_name)`: runs event case studies
- `news.plot_news_intensity_by_beta(...)`: news intensity by β decile × size group

## Note on the $\ell_1$ Rotation

The local-factor identification step uses code adapted from Freyaldenhoven's implementation of the $\ell_1$-rotation criterion:

Freyaldenhoven, Simon (2025). "Identification through sparsity in factor models: The $\ell_1$-rotation criterion." *Quantitative Economics*. [doi:10.3982/QE2369](https://doi.org/10.3982/QE2369).

In this codebase, the relevant implementation lives in `pca.py`, mainly through:

- `find_local_factors(...)`
- `_find_min_rotation(...)`
- related spherical-coordinate objective helpers

## Output Locations

The project writes outputs outside `Codes`:

- `../Data/`: downloaded and constructed datasets
- `../Results/`: intermediate and final CSV outputs
- `../Plots/`: saved figures
- `../CaseStudy/`: per-case outputs for the event studies (correlation heatmaps, summary tables, and stock lists)
