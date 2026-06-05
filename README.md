# The News-Contaminated Market: Local Factors and Asset Pricing

Code for the empirical analysis in `The News-Contaminated Market: Local Factors and Asset Pricing`.

Paper link: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6480458)

The core idea of the project is that the observed value-weighted market return is not a pure market factor. It combines a pervasive global component that reflects broad market-wide risk with local components driven by shocks that affect subsets of firms. The code therefore decomposes the market into a global factor and an orthogonal residual local news factor, `LNF`, using high-frequency data to capture the time-varying factor structure.

In the code:

- `Rg` denotes the unscaled global factor, defined via a two-step construction — PCA on S&P 500 to bootstrap a noisy global signal, then extended to the broader cross section — as the value-weighted return of stocks in the lowest decile of estimated local-factor exposure, measured by `|\hat{\gamma}_{i,t}|`
- `MKTg` denotes the projection of the market onto that global factor
- `LNF` denotes the component of the value-weighted market return orthogonal to the global factor, that is, the residual local-news component

The main empirical applications in the code are:

- intraday beta dispersion
- conditional security market line (SML) analysis
- in-sample spanning tests
- out-of-sample signal sorting
- beta-shock dynamics
- event case studies
- robustness sweeps and RavenPack news intensity diagnostics

## Repository Layout

This repository is notebook-driven. The reusable logic lives in the following Python modules under `Codes/`:

- `data_loader.py`: end-to-end data build script
- `data.py`: WRDS pulls, public-source downloads, cleaning, and preprocessing utilities
- `pca.py`: monthly PCA, sparse `\ell_1` rotation, and stock-level local-factor identification
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
| Section 3, subsection `The global--local decomposition and factor construction` | `nb_beta_dispersion.ipynb`, `nb_insample_spanning.ipynb`, `nb_cases.ipynb` | `nb_beta_dispersion.ipynb` produces the monthly PCA outputs; `nb_insample_spanning.ipynb` implements the two-step factor construction; `nb_cases.ipynb` provides local-cluster case illustrations |
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

1. Build monthly high-frequency panels for the S&P 500 universe.
2. Run PCA on within-month intraday returns.
3. Use sparse `\ell_1` rotation to identify local factors and a stock-level `news_proxy`.
4. Form a global factor from stocks with low local-news exposure.
5. Define the local factor as the residual between the market and the global component.
6. Use those objects in SML, spanning, anomaly, and beta-shock exercises.

Cross-section further requires non-missing AQR BetaFP, so the BetaFP D1/D10 benchmarks share the Rg/MKTg/LNF universe.

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
- `pca.find_local_factors(...)`: `\ell_1`-rotation identification of local factors (see Note on the `\ell_1` Rotation)
- `analysis_utils.load_port_analysis_data()`: loads the combined monthly inputs for portfolio analysis
- `decomposition.build_rg_factors(...)`: per-month Rg / MKTg / LNF construction and excess-return panel
- `decomposition.decompose_market_returns(...)`: single-month vwMKT → MKTg + LNF decomposition
- `decomposition.compute_robustness_at_kappa(...)`: κ-sweep driver behind `nb_robustness.ipynb`
- `port_sort.build_sml_panel(...)`: builds the daily panel used in SML tests
- `port_sort.get_beta_shock_panel()`: prepares the beta-shock analysis panel
- `spanning.compare_capm_spanning(...)`: CAPM vs Rg + LNF spanning regressions
- `cases.run_case_study(case_name)`: runs event case studies
- `news.plot_news_intensity_by_beta(...)`: news intensity by β decile × size group

## Note on the `\ell_1` Rotation

The sparse local-factor identification step uses code adapted from Freyaldenhoven's implementation of the `\ell_1` rotation criterion:

Freyaldenhoven, Simon (2026). `Identification through sparsity in factor models: The l1-rotation criterion.` Quantitative Economics. [doi:10.3982/QE2369](https://doi.org/10.3982/QE2369).

In this codebase, the relevant implementation lives in `pca.py`, mainly through:

- `find_local_factors(...)`
- `_find_min_rotation(...)`
- related spherical-coordinate objective helpers

## Output Locations

The project writes outputs outside `Codes`:

- `../Data/`: downloaded and constructed datasets
- `../Results/`: intermediate and final CSV outputs
- `../Plots/`: saved figures
- `../CaseStudy/`: case-study artifacts

## Practical Notes

- The code assumes a long sample, currently set to 2006-2025 in the loaders.
- Some routines intentionally reuse existing files and skip recomputation.
- The notebooks are the main interface; the modules are shared back-end utilities rather than a packaged library.
- The repository currently has no tests or packaging metadata, so reproducibility depends on the local environment and data access.
