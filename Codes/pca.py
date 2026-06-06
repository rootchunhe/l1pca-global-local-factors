import numpy as np
import pandas as pd
import numba
from pathlib import Path
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from constants import (
    data_path, result_path, plot_path,
    FREQ, N_BARS_PER_DAY, KN, PCA_GLOBAL_FACTOR_FOLD,
)
from analysis_utils import get_rBetas


def get_monthly_data(ym, max_zero_ratio=0.2, verbose=False):
    """Load monthly cRet data. Returns dSTK, dSPY, stock_info."""
    data_dir = Path(data_path)
    cret_path = data_dir / "sp500_cRet5min_ym" / f"cRetData_{ym}.csv"
    cbeta_path = data_dir / "sp500_monthly_cBetas_lagged.csv"

    sp500 = pd.read_csv(cret_path)
    sp500["datetime"] = pd.to_datetime(sp500["datetime"])
    sp500["SPY"] = sp500["SPY"].fillna(0)
    date_key = sp500["datetime"].dt.date

    daily_count = date_key.value_counts().sort_index()
    valid_dates = daily_count[daily_count == N_BARS_PER_DAY].index.tolist()
    sp500 = sp500[date_key.isin(valid_dates)].sort_values("datetime").reset_index(drop=True)
    T = len(valid_dates)
    if T == 0:
        raise ValueError(f"No days with exactly {N_BARS_PER_DAY} bars in {ym}")

    permno_cols = [c for c in sp500.columns if c not in ["datetime", "SPY"] and str(c).isdigit()]
    has_nan = sp500[permno_cols].isna().any()
    keep_cols = [c for c in permno_cols if not has_nan[c]]

    zero_ratio = (sp500[keep_cols] == 0).mean()
    ok = zero_ratio <= max_zero_ratio
    n_before = len(keep_cols)
    keep_cols = [c for c in keep_cols if ok[c]]
    if verbose and len(keep_cols) < n_before:
        print(f"Dropped {n_before - len(keep_cols)} stocks with zero_ratio > {max_zero_ratio}")
    permnos = np.array(sorted([int(c) for c in keep_cols]))

    cBetas = pd.read_csv(cbeta_path)
    yyyymm = int(ym.replace("-", ""))
    cBetas = cBetas.sort_values(["permno", "yyyymm"])
    cBetas_ym = cBetas[cBetas["yyyymm"] == yyyymm].copy()
    permnos = np.intersect1d(permnos, cBetas_ym["permno"].astype(int).values)
    permnos = np.sort(permnos)

    crsp_path = data_dir / "crsp_taq_monthly_info.csv"
    crsp_ym = pd.DataFrame()
    if crsp_path.exists():
        crsp = pd.read_csv(crsp_path, low_memory=False)
        crsp_ym = crsp[crsp["ym"] == yyyymm].drop_duplicates("permno", keep="last")

    # permco dedup by lag_mktcap
    if not crsp_ym.empty and "permco" in crsp_ym.columns and "lag_mktcap" in crsp_ym.columns:
        crsp_ym = crsp_ym[crsp_ym["permno"].isin(permnos)].copy()
        crsp_ym["lag_mktcap"] = crsp_ym["lag_mktcap"].fillna(0)
        keep_permnos = crsp_ym.sort_values("lag_mktcap", ascending=False).drop_duplicates("permco", keep="first")["permno"].values
        permnos = np.intersect1d(permnos, keep_permnos)
        permnos = np.sort(permnos)

    ticker_map = crsp_ym.set_index("permno")["ticker"].to_dict() if not crsp_ym.empty and "ticker" in crsp_ym.columns else {}
    permco_map = crsp_ym.set_index("permno")["permco"].to_dict() if not crsp_ym.empty and "permco" in crsp_ym.columns else {}
    cBetas_ym = cBetas_ym.sort_values("permno")
    cbeta_map = cBetas_ym.set_index("permno")["cBeta"].to_dict()

    mktcap = crsp_ym.set_index("permno")["lag_mktcap"].reindex(permnos).fillna(0).values
    wt = mktcap / mktcap.sum()

    stock_info = pd.DataFrame({"permno": permnos})
    stock_info["permco"] = stock_info["permno"].map(permco_map)
    stock_info["ticker"] = stock_info["permno"].map(ticker_map)
    stock_info["cBeta0"] = stock_info["permno"].map(cbeta_map)
    stock_info["wt"] = wt

    permnos_list = permnos.tolist()
    dSPY = sp500["SPY"].values
    dSTK = sp500[[str(p) for p in permnos_list]].values

    N = len(permnos)
    if verbose:
        print(f"ym={ym}: N={N} permnos, T={T} days, {N_BARS_PER_DAY} bars/day")
    assert dSTK.shape[0] == T * N_BARS_PER_DAY
    assert dSTK.shape[1] == N
    assert np.array_equal(stock_info["permno"].values, permnos)

    return dSTK, dSPY, stock_info


def run_PCA(X, r=3):
    """X (T x N), T obs N assets. SVD on X. Return F, B, d, ranks."""
    T, N = X.shape
    r = min(r, N, T)
    r_max = 10

    # SVD on X (T x N): X = U @ diag(s) @ Vh, V = Vh.T (N x min(T,N))
    _, s, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T
    d = s**2 / N
    B = np.sqrt(N) * V[:, :r]
    F = X @ B / N

    rank_ax = _rank_aitxiu(d, N, T, kmax=r_max)
    rank_pel = _rank_pelger(d, N, kmax=r_max)
    rank_kong = _rank_kong(X, rank_ax + 1, r_max)
    s_sf = s / np.sqrt(T)
    rank_sf = _rank_local_sf(V, s_sf, N, r_max)
    ranks = np.array([rank_ax, rank_pel, rank_kong, rank_sf], dtype=int)

    return F, B, d, ranks


def _rank_aitxiu(lambdas, p, n, md=None, kmax=None):
    """Penalised-PC rank test: r̂ = argmin_{1≤j≤kmax} (λ_j / p + j·g).
    Ref: Aït-Sahalia & Xiu (2017, J. Econometrics). j is 1-indexed (= the rank itself)."""
    if md is None:
        md = np.sqrt(p)
    if kmax is None:
        kmax = min(10, n)
    kmax = min(kmax, n, len(lambdas))
    lambdas = np.sort(lambdas)[::-1]
    kappa = 0.5
    mu = 0.02 * lambdas[int(min(p, n) / 2) - 1]
    g = mu * ((np.log(p) / n) ** 0.5 + md / p) ** kappa
    js = np.arange(1, kmax + 1)               # candidate rank values, 1-indexed
    objective = lambdas[:kmax] / p + js * g    # AX (2017) eq.
    return int(js[np.argmin(objective)])       # return the rank j, not the array index

def _rank_pelger(lambdas, p, gamma=0.13, kmax=10):
    """Perturbed eigenvalue-ratio rank test: shift eigenvalues by √p·median(λ), take
    largest k with ratio > 1+γ. Ref: Pelger (2019, J. Econometrics 208(1), 23-42),
    Theorem 6. Floored at 1."""
    kmax = min(kmax, len(lambdas) - 1)
    if kmax <= 0:
        return 1
    lambdas = np.sort(lambdas)[::-1]
    g = np.sqrt(p) * np.median(lambdas)
    lambdahat = lambdas[: kmax + 1] + g
    er = lambdahat[:-1] / lambdahat[1:]
    k = np.where(er > 1 + gamma)[0]
    return int(k[-1] + 1) if len(k) > 0 else 1


def _rank_kong(data, r_est, kmax):
    """Block-PCA factor-count estimator: argmin of (block-averaged residual MSE + Bai-Ng
    penalty) on √n-sized non-overlapping blocks. Ref: Kong (2017, Biometrika 104(2),
    397-410), eq. (5). data (n × p)."""
    n, p = data.shape
    kkn = max(int(n**0.5), 1)
    kmax = min(kkn - 1, kmax)
    nb = n // kkn
    r_est = min(r_est, kmax)
    mse = np.zeros((nb, kmax))
    for k in range(nb):
        sub = data[k * kkn : (k + 1) * kkn, :]
        M = sub @ sub.T / p / kkn
        eigvals, eigvecs = np.linalg.eigh(M)
        idx = np.argsort(eigvals)[::-1]
        ev = eigvecs[:, idx[:kmax]]
        for r in range(1, kmax + 1):
            Fr = sub @ sub.T @ ev[:, :r] / p / np.sqrt(kkn)
            FrInv = np.linalg.pinv(Fr.T @ Fr, rcond=1e-10)
            resid = sub.T @ sub - sub.T @ Fr @ FrInv @ Fr.T @ sub
            mse[k, r - 1] = np.trace(resid) / p / kkn
    mset = mse.mean(axis=0)
    mset = mset + mset[r_est - 1] * np.arange(1, kmax + 1) * (p + kkn) / (p * kkn) * np.log(p * kkn / (p + kkn))
    return np.argmin(mset) + 1


def _rank_local_sf(V, s, n, r_max, tau=0.5):
    """Locality-aware eigenvalue-ratio rank test: modify each eigenvalue by a loading-
    sparsity factor Ŝ_k² (top-z squared loadings / mean), then take argmax of the ratio
    chain (with a noise-floor mock at k=0). Ref: Freyaldenhoven (2022, J. Econometrics
    229(1), 80-102), Section 4.1.3. Port of `n_factors.m`
    (https://simonfreyaldenhoven.github.io/code/no_of_factors.zip). Floored at 1."""
    Lambda = V[:, :r_max] @ np.diag(s[:r_max])
    z = int(round(min(0.7 * n**tau * np.sqrt(np.log(np.log(max(n, 3)))), n)))
    z = max(z, 1)
    sorted_abs = np.sort(np.abs(Lambda), axis=0)[::-1]
    largest_z = sorted_abs[:z, :]
    Shat = np.zeros(r_max)
    T2 = np.zeros(r_max)
    for k in range(r_max):
        Shat[k] = (np.sum(largest_z[:, k] ** 2) / z) / np.sqrt(np.sum(Lambda[:, k] ** 2) / n)
        T2[k] = np.sum(Lambda[:, k] ** 2) * Shat[k] ** 2
    mock = float(np.sum(s[r_max - 1:] ** 2))
    incl_mock_T2 = np.concatenate(([mock], T2))
    T2_ratio = incl_mock_T2[:r_max] / T2[:r_max]
    return max(1, int(np.argmax(T2_ratio)))


def find_local_factors(X, r, Lambda0=None, seed=0):
    """L1-rotation for sparse factor loadings. X (T,n). Returns (Lambda0, Lambda_rotated).
    Ref: Freyaldenhoven (2026, QE). Lambda_rotated may have <r columns (no PC pad)."""
    X = np.asarray(X, dtype=np.float64)
    if np.isnan(X).any():
        raise ValueError("The input X cannot have missing values. Please impute these first.")
    T, n = X.shape
    if Lambda0 is None:
        _, _, Vh = np.linalg.svd(X / np.sqrt(T), full_matrices=False)
        V = Vh.T
        Lambda0 = np.sqrt(n) * V[:, :r]
    else:
        Lambda0 = np.asarray(Lambda0, dtype=np.float64)
        gram_round = np.round(Lambda0.T @ Lambda0)
        target = np.eye(r) * n
        if not np.array_equal(gram_round, target):
            raise ValueError(
                "The initial estimate Lambda0 should be an orthonormal basis of the loading space. "
                "Either drop argument (PCs will be used), or orthonormalize."
            )
    rmat_min = _find_min_rotation(Lambda0, seed=seed)
    Lambda_rotated = _collate_solutions(rmat_min, Lambda0, X)
    return Lambda0, Lambda_rotated


def _gridsize(r, mult=1):
    base = {2: 300, 3: 500, 4: 1000, 5: 2000}.get(r, 3000 if 5 < r < 9 else 5000)
    return base * mult


@numba.njit(cache=True)
def _spherical_to_cart(theta):
    r = len(theta) + 1
    R = np.zeros(r)
    R[0] = np.cos(theta[0])
    for kk in range(1, r - 1):
        prod_sin = 1.0
        for jj in range(kk):
            prod_sin *= np.sin(theta[jj])
        R[kk] = prod_sin * np.cos(theta[kk])
    prod_sin = 1.0
    for jj in range(r - 1):
        prod_sin *= np.sin(theta[jj])
    R[r - 1] = prod_sin
    return R


@numba.njit(cache=True)
def _obj_spherical(theta, Lambda):
    R = _spherical_to_cart(theta)
    proj = Lambda @ R
    total = 0.0
    for ii in range(proj.shape[0]):
        total += abs(proj[ii])
    return total


@numba.njit(cache=True)
def _sort_simplex(simplex, f_vals):
    npts = f_vals.shape[0]
    ndim = simplex.shape[1]
    for ii in range(npts):
        for jj in range(ii + 1, npts):
            if f_vals[jj] < f_vals[ii]:
                tmp_f = f_vals[ii]
                f_vals[ii] = f_vals[jj]
                f_vals[jj] = tmp_f
                for kk in range(ndim):
                    tmp_x = simplex[ii, kk]
                    simplex[ii, kk] = simplex[jj, kk]
                    simplex[jj, kk] = tmp_x


@numba.njit(cache=True)
def _nelder_mead_l1(theta0, Lambda, max_iter=2000, tol=1e-8):
    ndim = len(theta0)
    simplex = np.empty((ndim + 1, ndim))
    for ii in range(ndim):
        simplex[0, ii] = theta0[ii]
    for ii in range(ndim):
        for jj in range(ndim):
            simplex[ii + 1, jj] = theta0[jj]
        simplex[ii + 1, ii] += 0.05

    f_vals = np.empty(ndim + 1)
    for ii in range(ndim + 1):
        f_vals[ii] = _obj_spherical(simplex[ii], Lambda)

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(max_iter):
        _sort_simplex(simplex, f_vals)
        if f_vals[ndim] - f_vals[0] < tol:
            break

        centroid = np.zeros(ndim)
        for ii in range(ndim):
            for jj in range(ndim):
                centroid[jj] += simplex[ii, jj]
        for jj in range(ndim):
            centroid[jj] /= ndim

        x_r = np.empty(ndim)
        for jj in range(ndim):
            x_r[jj] = centroid[jj] + alpha * (centroid[jj] - simplex[ndim, jj])
        f_r = _obj_spherical(x_r, Lambda)

        if f_vals[0] <= f_r < f_vals[ndim - 1]:
            for jj in range(ndim):
                simplex[ndim, jj] = x_r[jj]
            f_vals[ndim] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = np.empty(ndim)
            for jj in range(ndim):
                x_e[jj] = centroid[jj] + gamma * (x_r[jj] - centroid[jj])
            f_e = _obj_spherical(x_e, Lambda)
            if f_e < f_r:
                for jj in range(ndim):
                    simplex[ndim, jj] = x_e[jj]
                f_vals[ndim] = f_e
            else:
                for jj in range(ndim):
                    simplex[ndim, jj] = x_r[jj]
                f_vals[ndim] = f_r
            continue

        x_c = np.empty(ndim)
        for jj in range(ndim):
            x_c[jj] = centroid[jj] + rho * (simplex[ndim, jj] - centroid[jj])
        f_c = _obj_spherical(x_c, Lambda)
        if f_c < f_vals[ndim]:
            for jj in range(ndim):
                simplex[ndim, jj] = x_c[jj]
            f_vals[ndim] = f_c
            continue

        for ii in range(1, ndim + 1):
            for jj in range(ndim):
                simplex[ii, jj] = simplex[0, jj] + sigma * (simplex[ii, jj] - simplex[0, jj])
            f_vals[ii] = _obj_spherical(simplex[ii], Lambda)

    _sort_simplex(simplex, f_vals)
    return simplex[0]


@numba.njit(cache=True, parallel=True)
def _find_min_rotation_numba(Lambda, no_draws, initial_theta):
    r = Lambda.shape[1]
    R = np.empty((r, no_draws))
    for rep in numba.prange(no_draws):
        best_theta = _nelder_mead_l1(initial_theta[:, rep], Lambda)
        R[:, rep] = _spherical_to_cart(best_theta)
    return R


def _find_min_rotation(Lambda, seed=0):
    """Lambda (n x r). Return R (r x no_draws), each col unit vector minimizing L1.
    seed for reproducibility."""
    n, r = Lambda.shape
    no_draws = _gridsize(r)
    initial = np.random.default_rng(seed).standard_normal((r, no_draws))
    initial = initial / np.linalg.norm(initial, axis=0, keepdims=True)
    theta = np.zeros((r - 1, no_draws))
    for kk in range(r - 2):
        theta[kk] = np.arctan2(np.linalg.norm(initial[kk + 1 :], axis=0), initial[kk])
    theta[r - 2] = np.arctan2(initial[r - 1], initial[r - 2])
    return _find_min_rotation_numba(
        np.ascontiguousarray(Lambda, dtype=np.float64),
        no_draws,
        np.ascontiguousarray(theta, dtype=np.float64),
    )


def _consolidate_local_mins(Lambda_0, rmat_min_unique, value_counts, sorting_column):
    if sorting_column != 1:
        order = np.argsort(value_counts[:, sorting_column - 1])[::-1]
        rmat_min_unique = rmat_min_unique[:, order]
        value_counts = value_counts[order]

    factorno = Lambda_0.shape[1]
    n = Lambda_0.shape[0]
    if rmat_min_unique.shape[1] == 0:
        return (
            np.empty((n, 0), dtype=Lambda_0.dtype),
            np.empty((factorno, 0), dtype=Lambda_0.dtype),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    Lambda_rotated = Lambda_0 @ rmat_min_unique[:, 0:1]
    R = rmat_min_unique[:, 0:1].copy()
    sol_frequency = np.array([value_counts[0, 1]], dtype=np.float64)
    fval = np.array([value_counts[0, 0]], dtype=np.float64)

    for kk in range(1, rmat_min_unique.shape[1]):
        temp = np.hstack([Lambda_rotated, Lambda_0 @ rmat_min_unique[:, kk : kk + 1]])
        min_e_temp = np.min(np.linalg.eigvalsh(temp.T @ temp) / n)
        min_e_rot = np.min(np.linalg.eigvalsh(Lambda_rotated.T @ Lambda_rotated) / n)
        if min_e_temp > np.sqrt(1 / factorno) / 3 and min_e_temp > min_e_rot / 4:
            Lambda_rotated = temp
            R = np.hstack([R, rmat_min_unique[:, kk : kk + 1]])
            sol_frequency = np.append(sol_frequency, value_counts[kk, 1])
            fval = np.append(fval, value_counts[kk, 0])

    return Lambda_rotated, R, sol_frequency, fval


def _collate_solutions(rmat_min, Lambda_0, X, epsilon_rot=0.05):
    """Combine local minima into rotated loadings. Output may have <factorno columns
    (omits MATLAB's PC fill-in; keeps only identified local-factor directions)."""
    T, n = X.shape
    factorno, no_randomgrid = rmat_min.shape
    l1_min = np.sum(np.abs(Lambda_0 @ rmat_min), axis=0)
    sort_idx = np.argsort(l1_min)
    l1_sort = l1_min[sort_idx]
    rmat_sort = rmat_min[:, sort_idx]
    rmat_sort = rmat_sort * np.sign(rmat_sort[0, :])

    for i in range(no_randomgrid):
        for j in range(i, no_randomgrid):
            if np.linalg.norm(rmat_sort[:, i] - rmat_sort[:, j]) / np.sqrt(factorno) < epsilon_rot:
                rmat_sort[:, j] = rmat_sort[:, i]
                l1_sort[j] = l1_sort[i]

    l1_unique, start_idx, group_id = np.unique(l1_sort, return_index=True, return_inverse=True)
    value_counts = np.column_stack([l1_unique, np.bincount(group_id)])
    rmat_unique = rmat_sort[:, start_idx]

    h_n = 1.0 / np.log(n)
    amount_sparsity = np.sum(np.abs(Lambda_0 @ rmat_unique) < h_n, axis=0)
    value_counts = np.column_stack([value_counts, amount_sparsity])

    non_outlier = value_counts[:, 1] / _gridsize(factorno) >= 0.005
    rmat_unique = rmat_unique[:, non_outlier]
    value_counts = value_counts[non_outlier]

    Lambda_rotated, R, sol_frequency, fval = _consolidate_local_mins(
        Lambda_0, rmat_unique, value_counts, sorting_column=3
    )

    return Lambda_rotated


def _movsum_backward(x, kn, axis=0):
    """Backward-looking rolling sum with vectorized window aggregation."""
    n = x.shape[axis]
    out = np.full_like(x, np.nan, dtype=np.result_type(x, np.float64))

    x_work = np.asarray(x, dtype=np.result_type(x, np.float64))
    x_fill = np.where(np.isnan(x_work), 0.0, x_work)
    x_nan = np.isnan(x_work).astype(np.int64)

    cs = np.cumsum(x_fill, axis=axis)
    nan_cs = np.cumsum(x_nan, axis=axis)

    pad_shape = list(x.shape)
    pad_shape[axis] = 1
    zeros = np.zeros(pad_shape, dtype=cs.dtype)
    zeros_i = np.zeros(pad_shape, dtype=nan_cs.dtype)

    cs_pad = np.concatenate([zeros, cs], axis=axis)
    nan_cs_pad = np.concatenate([zeros_i, nan_cs], axis=axis)

    out_slice = tuple(slice(None) if a != axis else slice(kn - 1, n) for a in range(x.ndim))
    start_slice = tuple(slice(None) if a != axis else slice(0, n - kn + 1) for a in range(x.ndim))
    end_slice = tuple(slice(None) if a != axis else slice(kn, n + 1) for a in range(x.ndim))

    win_sum = cs_pad[end_slice] - cs_pad[start_slice]
    win_nan = nan_cs_pad[end_slice] - nan_cs_pad[start_slice]
    out[out_slice] = np.where(win_nan > 0, np.nan, win_sum)
    return out


def get_sBeta_disp(MKT, STK, T, kn):
    """Spot-beta dispersion across stocks, pooled within month.
    Formulas and bias correction follow Andersen, Thyrsgaard & Todorov (2021),
    Quantitative Economics 12(2), 647-682.

    Returns (disps, disp_xs):
      disp_xs[κ, j] = per-stock bias-corrected (β-1)²
      disps[κ]      = nanmean over j
    """
    N = STK.shape[1]
    n = MKT.size // T
    dM = np.reshape(MKT, (n, T), order="F")
    dM_trunc = 1 - (dM == 0)

    n_out = n - kn + 1
    Ds = np.full((n_out, N), np.nan)
    Bs = np.full((n_out, N), np.nan)

    sliding_window_view = np.lib.stride_tricks.sliding_window_view

    for i in range(N):
        dS = np.reshape(STK[:, i], (n, T), order="F")
        dS_trunc = 1 - (dS == 0)

        alpha = np.nanmean(dS_trunc * (dM**2))
        if alpha < (0.1 / np.log(n)) ** 2:
            continue

        qvm = np.sum(_movsum_backward(dS_trunc * (dM**2), kn, axis=0), axis=1)
        qvms = np.sum(_movsum_backward(dS * dM, kn, axis=0), axis=1)
        sBetas = qvms[kn - 1 :] / qvm[kn - 1 :]

        ratio = np.broadcast_to((qvms / np.where(qvm == 0, np.nan, qvm))[:, None], (n, T))
        resid = dS - dM * np.where(np.isnan(ratio), 0, ratio)
        var_num = np.sum(
            _movsum_backward(dS_trunc * (dM**2) * (resid**2), kn, axis=0), axis=1
        )
        Var_sBeta = var_num / np.where(qvm == 0, np.nan, qvm**2)
        Var_sBeta = Var_sBeta[kn - 1 :]

        trunc_sparse = (
            dS_trunc[:-1] * dS_trunc[1:] * dM_trunc[:-1] * dM_trunc[1:]
        )
        dS_sparse = (dS[:-1] + dS[1:]) * trunc_sparse
        dM_sparse = (dM[:-1] + dM[1:]) * trunc_sparse
        c2_raw = _movsum_backward((dS_sparse - dM_sparse) * dM_sparse, kn - 1, axis=0)
        C2 = kn / (kn - 1) / 2 * np.sum(c2_raw[kn - 2 :], axis=1)
        Ds[:, i] = (C2 / qvm[kn - 1 :]) ** 2

        # batched rolling-window diff/sum for bias correction
        view_s = sliding_window_view(dS_trunc * (dM**2), kn, axis=0)
        temp_s = np.diff(view_s, axis=-1)
        V_breve = np.sum(temp_s[..., 0::2], axis=(1, 2))

        view_c = sliding_window_view(dS_trunc * dM * (dS - dM), kn, axis=0)
        temp_c = np.diff(view_c, axis=-1)
        C_breve = np.sum(temp_c[..., 0::2], axis=(1, 2))

        btemp1 = -((sBetas - 1) ** 2 - Var_sBeta) * (V_breve**2) / (qvm[kn - 1 :] ** 2)
        btemp2 = 3 / 2 * (C_breve**2) / (qvm[kn - 1 :] ** 2)
        Bs[:, i] = btemp1 + btemp2

    disp_xs = Ds - Bs
    disps = np.nanmean(disp_xs, axis=1)
    return disps, disp_xs


def get_eta_and_variance_ratio(dSPY, Rg, T, kn):
    """Return (n-kn+1, 2) array [eta, vr]."""
    n = dSPY.size // T
    dSPY_2d = np.reshape(dSPY, (n, T), order="F")
    Rg_2d = np.reshape(Rg, (n, T), order="F")
    QV_SPY = np.sum(_movsum_backward(dSPY_2d**2, kn, axis=0), axis=1)[kn - 1 :]
    QV_Rg = np.sum(_movsum_backward(Rg_2d**2, kn, axis=0), axis=1)[kn - 1 :]
    QV_cross = np.sum(_movsum_backward(dSPY_2d * Rg_2d, kn, axis=0), axis=1)[kn - 1 :]
    eta = QV_cross / np.where(QV_Rg == 0, np.nan, QV_Rg) - 1
    vr = QV_Rg / np.where(QV_SPY == 0, np.nan, QV_SPY)
    return np.column_stack([eta, vr])


def get_PCA_results(ym, n=N_BARS_PER_DAY, kn=KN):
    """Pipeline: load, PCA, local factors.
    Returns (ResultTable, summary_row, intday_disp_tb, eta_vr_res)."""
    dSTK, dSPY, stock_info = get_monthly_data(ym, verbose=False)
    # convert per-bar returns to annualized-vol units (PCA numerical conditioning only;
    # all 4 rank tests, β's, and L1-rotation outputs are scale-invariant)
    scale = np.sqrt(n - 1) * np.sqrt(252)
    dSTK, dSPY = dSTK * scale, dSPY * scale
    T = dSTK.shape[0] // n
    N = dSTK.shape[1]

    ResultTable = stock_info.copy()
    cBetas = stock_info["cBeta0"].values
    disps_spy, disps_spy_xs = get_sBeta_disp(dSPY, dSTK, T, kn)
    ResultTable["cBetaSPY"] = get_rBetas(dSPY, dSTK)

    PCtemp, Btemp, dtemp, ranks = run_PCA(dSTK, r=3)
    PC1_var_prop = dtemp[0] / dtemp.sum()
    PCbetas = np.array([np.sum(PCtemp[:, k] * dSPY) / np.sum(dSPY**2) for k in range(3)])
    ResultTable["PC1_wt"] = Btemp[:, 0] / Btemp[:, 0].sum()

    n_out = n - kn + 1
    disps_rg = np.full(n_out, np.nan)
    intday_disp_tb = np.full((n_out, 4), np.nan)
    eta_vr_res = np.full((n_out, 2), np.nan)

    my_rank = int(np.ceil(np.median(ranks[:3])))   # SF excluded: its argmax ratio test collapses rank to ~1
    if my_rank <= 1:
        print(f"  {ym}: rank=1, skipping local-factor pipeline (cBeta*, news_proxy → NaN)")
    if my_rank > 1:
        _, Lambda = find_local_factors(dSTK, my_rank, seed=int(ym.replace("-", "")))  # seed per month
        n_clusters = Lambda.shape[1]

        if n_clusters > 0:
            stk_news = np.sum(np.abs(Lambda), axis=1)
            cutoff_pct = 1.0 / PCA_GLOBAL_FACTOR_FOLD
            top_news_stks = np.where(stk_news > np.quantile(stk_news, 1-cutoff_pct))[0] 
            bot_news_stks = np.where(stk_news < np.quantile(stk_news, cutoff_pct))[0]

            # Intraday-only diagnostics.
            # Rg_intra_scaling: same-month intraday β-on-Rg / lagged daily CAPM β (cBeta0).
            # Asymmetric numerator/denominator is intentional — puts Rg on SPY-comparable scale.
            # orth_scaling_intra = Cov(SPY,Rg)/Var(Rg) at intraday frequency.
            Rg = np.mean(dSTK[:, bot_news_stks], axis=1)
            Rg[dSPY == 0] = 0
            Rg_intra_scaling = np.mean(get_rBetas(Rg, dSTK)[bot_news_stks]) / np.mean(cBetas[bot_news_stks])
            orth_scaling_intra = np.sum(Rg * dSPY) / np.sum(Rg**2)
            dMKTg = orth_scaling_intra * Rg
            Rg = Rg_intra_scaling * Rg

            disps_rg, disps_rg_xs = get_sBeta_disp(Rg, dSTK, T, kn)
            eta_vr_res = get_eta_and_variance_ratio(dSPY, Rg, T, kn)

            ResultTable["cBetaRg"] = get_rBetas(Rg, dSTK)
            ResultTable["cBetaMKTg"] = get_rBetas(dMKTg, dSTK)
            ResultTable["cBetaLNF"] = get_rBetas(dSPY - dMKTg, dSTK)
            for j in range(n_clusters):
                ResultTable[f"gamma{j+1}"] = Lambda[:, j]
            ResultTable["news_proxy"] = stk_news

            ResultTable["Rg_wt_intra"] = 0.0
            ResultTable["MKTg_wt_intra"] = 0.0
            ResultTable.loc[ResultTable.index[bot_news_stks], "Rg_wt_intra"] = Rg_intra_scaling / len(bot_news_stks)
            ResultTable.loc[ResultTable.index[bot_news_stks], "MKTg_wt_intra"] = orth_scaling_intra / len(bot_news_stks)

            intday_disp_tb = np.column_stack([
                np.nanmean(disps_spy_xs[:, top_news_stks], axis=1),
                np.nanmean(disps_spy_xs[:, bot_news_stks], axis=1),
                np.nanmean(disps_rg_xs[:, top_news_stks], axis=1),
                np.nanmean(disps_rg_xs[:, bot_news_stks], axis=1),
            ])
    else:
        n_clusters = 0

    msg = f"{ym}: N={N} T={T} | ranks AX/Pel/Kong/SF={ranks[0]}/{ranks[1]}/{ranks[2]}/{ranks[3]}, used={my_rank} | disp SPY {disps_spy[0]:.3f}->{disps_spy[-1]:.3f}"
    if n_clusters > 0:
        msg += f", Rg {disps_rg[0]:.3f}->{disps_rg[-1]:.3f}"
    msg += f" | {n_clusters} local factors"
    print("\r" + " " * 220, end="", flush=True)
    print("\r" + msg, end="", flush=True)

    summary_row = {
        "YM": int(ym.replace("-", "")),
        "rank_AX": ranks[0], "rank_Pel": ranks[1], "rank_Kong": ranks[2], "rank_SF": ranks[3],
        "nCluster": n_clusters, "N": N, "PC1_var_exp": PC1_var_prop,
        "PCbeta1": PCbetas[0], "PCbeta2": PCbetas[1], "PCbeta3": PCbetas[2],
        **{f"disp_spy_t{i+1}": disps_spy[i] for i in range(n_out)},
        **{f"disp_rg_t{i+1}": disps_rg[i] for i in range(n_out)},
    }

    return ResultTable, summary_row, intday_disp_tb, eta_vr_res


def _process_one_month(ym):
    """Process one month and return rows for summary/output tables."""
    ResultTable, summary_row, intday_disp_tb, eta_vr_res = get_PCA_results(
        ym, n=N_BARS_PER_DAY, kn=KN
    )
    date_int = int(ym.replace("-", ""))
    ResultTable = ResultTable.copy()
    ResultTable["YM"] = date_int
    disp_row = [date_int] + sum([intday_disp_tb[:, j].tolist() for j in range(4)], [])
    eta_row = [date_int] + eta_vr_res[:, 0].tolist() + eta_vr_res[:, 1].tolist()
    return (summary_row, disp_row, eta_row, ResultTable)


def run_monthly_analysis(ym_list):
    """Run get_PCA_results for each month, aggregate and save results."""
    rp = Path(result_path)
    rp.mkdir(parents=True, exist_ok=True)

    n, kn = N_BARS_PER_DAY, KN
    n_out = n - kn + 1

    summary_rows = []
    disp_tpbt_rows = []
    eta_vr_rows = []
    result_tables = []

    print(f"Simple for loop: {len(ym_list)} months")
    # serial: per-month is CPU-heavy (SVD + rotation + dispersion); outer-parallel causes BLAS oversubscription
    for ym in ym_list:
        summary_row, disp_row, eta_row, ResultTable = _process_one_month(ym)
        summary_rows.append(summary_row)
        disp_tpbt_rows.append(disp_row)
        eta_vr_rows.append(eta_row)
        result_tables.append(ResultTable)

    df_summary = pd.DataFrame(summary_rows)
    df_disp_tpbt = pd.DataFrame(disp_tpbt_rows, columns=["YM"] + [f"spy_tp_{i+1}" for i in range(n_out)]
        + [f"spy_bt_{i+1}" for i in range(n_out)] + [f"rg_tp_{i+1}" for i in range(n_out)]
        + [f"rg_bt_{i+1}" for i in range(n_out)])
    df_eta_vr = pd.DataFrame(eta_vr_rows, columns=["YM"]
        + [f"eta_{i+1}" for i in range(n_out)] + [f"vr_rg_spy_{i+1}" for i in range(n_out)])

    df_all_pcas = pd.concat(result_tables, ignore_index=True)

    df_summary.to_csv(rp / "monthly_disp.csv", index=False)
    df_disp_tpbt.to_csv(rp / "monthly_disp_tpbt.csv", index=False)
    df_eta_vr.to_csv(rp / "monthly_eta_vr.csv", index=False)
    df_all_pcas.to_csv(rp / "monthly_pcas.csv", index=False)
    print("Saved: monthly_disp.csv, monthly_disp_tpbt.csv, monthly_eta_vr.csv, monthly_pcas.csv")


def load_dispersion_data():
    """Load monthly_disp, monthly_eta_vr, monthly_disp_tpbt. Return dict."""
    rp = Path(result_path)
    n, kn = N_BARS_PER_DAY, KN
    n_out = n - kn + 1
    disp_tb = pd.read_csv(rp / "monthly_disp.csv")
    eta_vr_tb = pd.read_csv(rp / "monthly_eta_vr.csv")
    disp_tb_tpbt = pd.read_csv(rp / "monthly_disp_tpbt.csv")

    etas = eta_vr_tb[[f"eta_{i+1}" for i in range(n_out)]].values
    vr = eta_vr_tb[[f"vr_rg_spy_{i+1}" for i in range(n_out)]].values
    spy_disp = disp_tb[[f"disp_spy_t{i+1}" for i in range(n_out)]].values
    rg_disp = disp_tb[[f"disp_rg_t{i+1}" for i in range(n_out)]].values
    spy_tp = np.nanmean(disp_tb_tpbt[[f"spy_tp_{i+1}" for i in range(n_out)]].values, axis=0)
    spy_bt = np.nanmean(disp_tb_tpbt[[f"spy_bt_{i+1}" for i in range(n_out)]].values, axis=0)
    rg_tp = np.nanmean(disp_tb_tpbt[[f"rg_tp_{i+1}" for i in range(n_out)]].values, axis=0)
    rg_bt = np.nanmean(disp_tb_tpbt[[f"rg_bt_{i+1}" for i in range(n_out)]].values, axis=0)

    # bar 0 closes 09:40 (09:30->09:35 dropped); first kn-window ends at 09:40 + (kn-1)*FREQ, last at 16:00.
    start_min = 9 * 60 + 40 + (kn - 1) * FREQ
    t0 = datetime(2024, 1, 1, start_min // 60, start_min % 60, 0)
    intraday_time = [t0 + timedelta(minutes=FREQ * i) for i in range(n_out)]

    return {
        "disp_n": n_out,
        "intraday_index": np.linspace(0, 1, n_out),
        "intraday_time": intraday_time,
        "etas": etas, "vr": vr, "spy_disp": spy_disp, "rg_disp": rg_disp,
        "spy_disp_all": np.column_stack([np.nanmean(spy_disp, axis=0), spy_tp, spy_bt]),
        "rg_disp_all": np.column_stack([np.nanmean(rg_disp, axis=0), rg_tp, rg_bt]),
        "pcbetas": disp_tb[[f"PCbeta{i+1}" for i in range(3)]].values,
    }


def plot_eta_vr(data):
    """Plot eta and sigma_n^2/sigma_SPY^2."""
    out_path = Path(plot_path)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = data["intraday_time"]
    ax.plot(x, np.nanmean(data["etas"], axis=0), "-", lw=1.5, label=r"$\eta$")
    ax.plot(x, 1 - np.nanmean(((1 + data["etas"]) ** 2) * data["vr"], axis=0), "-", lw=1.5, label=r"$\sigma^2_n/\sigma^2_{SPY}$")
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.savefig(out_path / "eta_vr.pdf", bbox_inches="tight")
    plt.show()


_GROUP_NAME = {2: "half", 3: "tercile", 4: "quartile", 5: "quintile", 10: "decile"}


def _news_group_labels():
    """Legend labels for news-quantile groups; derived from PCA_GLOBAL_FACTOR_FOLD."""
    g = _GROUP_NAME.get(PCA_GLOBAL_FACTOR_FOLD, f"1/{PCA_GLOBAL_FACTOR_FOLD}")
    return ["all", f"top {g}", f"bottom {g}"]


def plot_intday_news(data):
    """Plot dispersion by news group."""
    out_path = Path(plot_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True, gridspec_kw={"wspace": 0.08}, constrained_layout=True)
    x = data["intraday_time"]
    ax1.plot(x, data["spy_disp_all"], lw=1.5)
    ax1.legend(_news_group_labels(), loc="best")
    ax1.set_title(r"dispersion of $\beta_{SPY}$")
    ax2.plot(x, data["rg_disp_all"], lw=1.5)
    ax2.set_title(r"dispersion of $\beta_g$")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.savefig(out_path / "intday_beta_disp_news.pdf", bbox_inches="tight")
    plt.show()


def plot_intday_news_tight(data):
    """Tight plot: dashed=SPY, solid=Rg. Same color per group."""
    out_path = Path(plot_path)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = data["intraday_time"]
    handles = []
    for i in range(3):
        c = f"C{i}"
        h = ax.plot(x, data["rg_disp_all"][:, i], "-", lw=1.5, color=c)
        handles.append(h[0])
        ax.plot(x, data["spy_disp_all"][:, i], "--", lw=1.5, color=c)
    lgd = ax.legend(handles, _news_group_labels(), loc="best")
    lgd.set_title("dashed: $\\beta_{SPY}$; solid: $\\beta_g$")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.savefig(out_path / "intday_beta_disp_news2.pdf", bbox_inches="tight")
    plt.show()


def plot_sbeta_by_pcbeta2(data, num_grp=5):
    """Plot intraday dispersion grouped by abs(PCbeta2)."""
    out_path = Path(plot_path)
    pcbeta2_abs = np.abs(data["pcbetas"][:, 1])
    valid = ~np.isnan(pcbeta2_abs)
    pcbeta2_abs_v = pcbeta2_abs[valid]
    spy_disp_v = data["spy_disp"][valid]
    q = np.nanquantile(pcbeta2_abs_v, np.linspace(0, 1, num_grp + 1))
    q[-1] = np.inf
    group_ids = np.digitize(pcbeta2_abs_v, q[1:-1], right=True)
    group_means = np.array([np.nanmean(spy_disp_v[group_ids == g], axis=0) for g in range(num_grp)])
    x = data["intraday_time"]

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    handles = []
    for i in range(num_grp):
        handles.append(ax.plot(x, group_means[i], lw=1.5)[0])
    handles.append(ax.plot(x, np.nanmean(data["spy_disp"], axis=0), "k--", lw=1.5)[0])
    labels = ["Low"] + [f"Group{i}" for i in range(2, num_grp)] + ["High", "Unconditional"]
    order = list(range(num_grp - 1, -1, -1)) + [num_grp]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.savefig(out_path / "intday_sbeta_disp_by_pcbeta2.pdf", bbox_inches="tight")
    plt.show()


def plot_implied_pattern(data, kn=271, n_stocks=500, rng_seed=1, smooth_s=0.003):
    """Plot implied beta pattern (calib_QE2021fig3): inputs, dispersion, spot-beta quantiles."""
    etas = data["etas"]
    vr = data["vr"]
    n_out = data["disp_n"]

    eta = np.nanmean(etas, axis=0)
    vr_p = np.nanmean(vr, axis=0)
    vr_n = 1 - np.nanmean(vr * (1 + etas) ** 2, axis=0)

    xgrid = np.arange(1, n_out + 1, dtype=float)
    xgrid_extend = np.linspace(xgrid[0], xgrid[-1], kn)

    eta = UnivariateSpline(xgrid, eta, s=smooth_s)(xgrid_extend)
    vr_p = UnivariateSpline(xgrid, vr_p, s=smooth_s)(xgrid_extend)
    vr_n = UnivariateSpline(xgrid, vr_n, s=smooth_s)(xgrid_extend)

    # First kn-window ends at 11:35 (bar 23 close); last at 16:00. 265 min span.
    t0 = datetime(2024, 1, 1, 11, 35, 0)
    total_sec = 265 * 60
    time_grid = [t0 + timedelta(seconds=i * total_sec / (kn - 1)) for i in range(kn)]

    def sbeta_grid(beta, gamma):
        return (beta + gamma * eta) * (1 + eta) * vr_p + gamma * vr_n

    rng = np.random.default_rng(rng_seed)
    betas = np.sort(1 + rng.standard_normal(n_stocks) * 0.3)
    gammas = np.maximum(3 * betas - 2.1, 0)
    sbetas = np.zeros((n_stocks, kn))
    for i in range(n_stocks):
        sbetas[i, :] = sbeta_grid(betas[i], gammas[i])

    out_path = Path(plot_path)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), gridspec_kw={"wspace": 0.08}, constrained_layout=True)

    ax1.plot(time_grid, eta, lw=1.5, label=r"$\eta$")
    ax1.plot(time_grid, vr_p, lw=1.5, label=r"$\frac{\sigma^2_g}{\sigma^2_{SPY}}$")
    ax1.plot(time_grid, vr_n, lw=1.5, label=r"$\frac{\sigma^2_n}{\sigma^2_{SPY}}$")
    ax1.set_title("inputs")
    ax1.legend(loc="center right", fontsize=16)

    ax2.plot(time_grid, np.nanmean((sbetas - 1) ** 2, axis=0), lw=1.5)
    ax2.set_title(r"implied dispersion of $\beta_{SPY}$")

    n_q = 5
    q_means = np.array(
        [blk.mean(axis=0) for blk in np.array_split(sbetas, n_q, axis=0)]
    )  # betas sorted ascending -> equal-size beta quintile means
    for q in range(n_q):
        ax3.axhline(q_means[q, 0], color="gray", alpha=0.4, linestyle="--", zorder=0)
    for q in range(n_q):
        ax3.plot(time_grid, q_means[q, :], lw=1.5, zorder=1)
    _pad = 0.05 * (q_means.max() - q_means.min())
    ax3.set_ylim(q_means.min() - _pad, q_means.max() + _pad)
    ax3.set_title(r"implied spot-beta across quantiles")

    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="both", labelsize=18)
        ax.title.set_fontsize(22)

    fig.savefig(out_path / "calib_QE2021fig3.pdf", bbox_inches="tight")
    plt.show()
