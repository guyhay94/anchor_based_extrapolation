import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, re
from pathlib import Path

from legendre_utils import fit_regular_ls, fit_ridge, evaluate_legendre_polynomial_with_coefs, \
    create_legendre_basis_functions_numpy
from utils import (
    calc_l2_approximation_error, calc_inner_extrapolation_condition_number,
    convert_basis_function_orthogonality_domain,
    calc_extrapolation_condition_number_improved
)
from theoretical_tests import (
    project_function_into_l2_ball,
    add_noise,
)

# =========================
# Real-data truth back-end
# =========================
_REAL_X = None
_REAL_Y = None
_REAL_M = None
_REAL_XMIN = None
_REAL_XMAX = None


def _pchip_slopes(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least two data points.")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")
    delta = np.diff(y) / h
    m = np.zeros(n, dtype=float)

    # interior slopes (Fritsch–Carlson)
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            m[k] = 0.0

    # endpoint slopes (one-sided, non-overshooting)
    if n == 2:
        m[0] = delta[0]
        m[1] = delta[0]
        return m

    m[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(m[0]) != np.sign(delta[0]):
        m[0] = 0.0
    elif (np.sign(delta[0]) != np.sign(delta[1])) and (abs(m[0]) > abs(3 * delta[0])):
        m[0] = 3 * delta[0]

    m[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(m[-1]) != np.sign(delta[-1]):
        m[-1] = 0.0
    elif (np.sign(delta[-1]) != np.sign(delta[-2])) and (abs(m[-1]) > abs(3 * delta[-1])):
        m[-1] = 3 * delta[-1]

    return m


def _pchip_eval(xq, x, y, m):
    xq = np.asarray(xq, dtype=float)
    idx = np.searchsorted(x, xq) - 1
    idx = np.clip(idx, 0, len(x) - 2)

    h = x[idx + 1] - x[idx]
    t = (xq - x[idx]) / h

    yk, yk1 = y[idx], y[idx + 1]
    mk, mk1 = m[idx], m[idx + 1]

    # Cubic Hermite basis
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)

    return h00 * yk + h10 * h * mk + h01 * yk1 + h11 * h * mk1


def _denormalize_from_minus_one_one(xn, xmin, xmax):
    return (xn + 1.0) * 0.5 * (xmax - xmin) + xmin


def _load_xy_csv(csv_path, x_col, y_col):
    xs, ys = [], []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
            except (KeyError, ValueError):
                continue
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    order = np.argsort(x)
    x = x[order];
    y = y[order]
    uniq = np.r_[True, np.diff(x) > 0]
    return x[uniq], y[uniq]


def _load_xy_txt_generic(filepath: str):
    """
    Parse refractiveindex.info style TXT (first numeric col=x, second=y).
    Ignores blank lines and lines starting with #, ;, //.
    Accepts whitespace- or comma-separated values.
    """
    xs, ys = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            parts = re.split(r"[,\s]+", line)
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if len(nums) >= 2:
                xs.append(nums[0]);
                ys.append(nums[1])

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        raise ValueError(f"No numeric (x,y) pairs parsed from TXT file: {filepath}")
    order = np.argsort(x)
    x = x[order];
    y = y[order]
    uniq = np.r_[True, np.diff(x) > 0]
    return x[uniq], y[uniq]


def init_real_truth(path, x_col="wavelength_nm", y_col="n"):
    """
    Install REAL dataset (x,y) as the ground truth.
    If `path` ends with .txt → parse as generic TXT (first col=x, second=y).
    Else → read CSV with the provided column names.
    """
    global _REAL_X, _REAL_Y, _REAL_M, _REAL_XMIN, _REAL_XMAX
    path = str(Path(path))
    if path.lower().endswith(".txt"):
        x, y = _load_xy_txt_generic(path)
    else:
        x, y = _load_xy_csv(path, x_col, y_col)

    if len(x) < 2:
        raise ValueError("Dataset must contain at least two distinct x values.")

    _REAL_X = x
    _REAL_Y = y
    _REAL_M = _pchip_slopes(x, y)
    _REAL_XMIN = float(np.min(x))
    _REAL_XMAX = float(np.max(x))
    print(f"[real-data] loaded {len(x)} points, x in [{_REAL_XMIN}, {_REAL_XMAX}]")


def map_domain(t, a, b):
    """Ignore (a,b); map normalized t ∈ [-1,1] to dataset [xmin, xmax]."""
    if _REAL_XMIN is None or _REAL_XMAX is None:
        raise RuntimeError("init_real_truth(...) must be called before using map_domain.")
    t = np.asarray(t, dtype=float)
    return _denormalize_from_minus_one_one(t, _REAL_XMIN, _REAL_XMAX)


def true_function(x):
    """
    If inputs lie in [-1.05,1.05] → treat as normalized, else original x-units.
    Interpolate on [xmin,xmax] with PCHIP; linear extrapolate outside.
    """
    if _REAL_X is None:
        raise RuntimeError("init_real_truth(...) must be called before using true_function.")
    x = np.asarray(x, dtype=float)
    if np.all((x >= -1.05) & (x <= 1.05)):
        xx = _denormalize_from_minus_one_one(x, _REAL_XMIN, _REAL_XMAX)
    else:
        xx = x
    yq = _pchip_eval(np.clip(xx, _REAL_XMIN, _REAL_XMAX), _REAL_X, _REAL_Y, _REAL_M)
    left = xx < _REAL_XMIN
    right = xx > _REAL_XMAX
    if np.any(left):
        yq[left] = _REAL_Y[0] + _REAL_M[0] * (xx[left] - _REAL_XMIN)
    if np.any(right):
        yq[right] = _REAL_Y[-1] + _REAL_M[-1] * (xx[right] - _REAL_XMAX)
    return yq


# =========================
# Experiments
# =========================
# ===== Modified run_ode_experiment that *returns True on success* + metrics =====
def run_two_fitted_anchors_ls_ridge_experiment(
        cutoff: float = 0.9,
        degree: int = 12,
        sigma: float = 0.1,
        lasso_alpha: float = 1e-3,
        *,
        plot: bool = False,
        verbose: bool = False,
        success_tol: float = 0.0,  # require at least this absolute improvement
        return_metrics: bool = False,  # if True, also return a metrics dict
):
    # Normalized domain
    T = np.linspace(-1, 1, 400)
    Omega = T[T <= cutoff]
    Xi = np.linspace(cutoff, 1, 400)

    # Map to real-data x-domain and evaluate truth
    t_full = map_domain(T, 0.0, 2.0)
    f_full = true_function(t_full)
    f_omega = f_full[:len(Omega)]
    f_xi = true_function(Xi)

    # Add noise on Omega
    y_omega_noisy = add_noise(f_omega, sigma, seed=0)

    basis = create_legendre_basis_functions_numpy(degree)

    # 1) LS / Ridge fits on Ω
    ls_coef = fit_regular_ls(x=Omega, y=y_omega_noisy, n=degree + 1)
    lasso_coef = fit_ridge(x=Omega, y=y_omega_noisy, n=degree + 1, alpha=lasso_alpha)

    # 2) In-domain error proxy Ē on Ω
    y_ls_omega = evaluate_legendre_polynomial_with_coefs(Omega, ls_coef)
    y_lasso_omega = evaluate_legendre_polynomial_with_coefs(Omega, lasso_coef)
    E_ls = calc_l2_approximation_error(y_true=y_omega_noisy, y_pred=y_ls_omega, x_samples=Omega)
    E_lasso = calc_l2_approximation_error(y_true=y_omega_noisy, y_pred=y_lasso_omega, x_samples=Omega)

    # 3) Extrapolation condition numbers κ
    basis_functions_omega = convert_basis_function_orthogonality_domain(
        basis, basis_orthogonality_a=-1, basis_orthogonality_b=1, a_1=-1, b_1=cutoff
    )
    kappa_1 = calc_extrapolation_condition_number_improved(
        phi_list=basis_functions_omega, domain_omega=(-1, cutoff), domain_xi=(cutoff, 1)
    )
    kappa_2 = calc_inner_extrapolation_condition_number(
        phi_list=basis, domain_xi=(cutoff, 1)
    )
    kappa = np.nanmin(np.array([kappa_1, kappa_2], dtype=float))

    # Guard rails: if κ or ℓ is nan, declare failure early
    if not np.isfinite(kappa):
        if verbose:
            print(f"[skip] non-finite kappa ({kappa})")
        out = (False, dict(
            cutoff=cutoff, degree=degree, sigma=sigma, alpha=lasso_alpha,
            kappa=float(kappa),
            reason="non_finite_kappa_or_ell"
        ))
        return out if return_metrics else out[0]

    # 5) Radii
    radius_ls = kappa * E_ls
    radius_lasso = kappa * E_lasso
    print(f"[skip] kappa ({kappa})")

    # 6) Compare LS vs Ridge on Ξ
    y_xi_ls = evaluate_legendre_polynomial_with_coefs(x=Xi, coefs=ls_coef)
    y_xi_lasso = evaluate_legendre_polynomial_with_coefs(x=Xi, coefs=lasso_coef)
    ls_lasso_dist = calc_l2_approximation_error(y_xi_ls, y_xi_lasso, x_samples=Xi)

    # Feasibility
    lasso_in = ls_lasso_dist <= radius_ls
    ls_in = ls_lasso_dist <= radius_lasso

    # if verbose:
    if not lasso_in:
        print("Ridge not in LS feasible area")
    if not ls_in:
        print("LS not in Ridge feasible area")

    # 7) Projections onto L2-balls on Ξ
    if not lasso_in:
        proj_lasso_coef = project_function_into_l2_ball(
            coeff_center=ls_coef, coeff_target=lasso_coef, radius=radius_ls, x_xi=Xi
        )
    else:
        proj_lasso_coef = lasso_coef

    if not ls_in:
        proj_ls_coef = project_function_into_l2_ball(
            coeff_center=lasso_coef, coeff_target=ls_coef, radius=radius_lasso, x_xi=Xi
        )
    else:
        proj_ls_coef = ls_coef

    # 8) Distances to true on Ξ
    ls_fn = lambda t: evaluate_legendre_polynomial_with_coefs(t, ls_coef)
    lasso_fn = lambda t: evaluate_legendre_polynomial_with_coefs(t, lasso_coef)
    proj_ls_fn = lambda t: evaluate_legendre_polynomial_with_coefs(t, proj_ls_coef)
    proj_las_fn = lambda t: evaluate_legendre_polynomial_with_coefs(t, proj_lasso_coef)

    dist_ls = calc_l2_approximation_error(y_true=f_xi, y_pred=ls_fn(Xi), x_samples=Xi)
    dist_lasso = calc_l2_approximation_error(y_true=f_xi, y_pred=lasso_fn(Xi), x_samples=Xi)
    dist_proj_ls = calc_l2_approximation_error(y_true=f_xi, y_pred=proj_ls_fn(Xi), x_samples=Xi)
    dist_proj_lasso = calc_l2_approximation_error(y_true=f_xi, y_pred=proj_las_fn(Xi), x_samples=Xi)

    delta_for_ls_improvement_bounds = calc_l2_approximation_error(y_true=ls_fn(Xi), y_pred=proj_ls_fn(Xi), x_samples=Xi) ** 0.5
    delta_for_lasso_improvement_bounds = calc_l2_approximation_error(y_true=lasso_fn(Xi), y_pred=proj_las_fn(Xi), x_samples=Xi) ** 0.5
    lower_bound_for_ls_improvement_bound = delta_for_ls_improvement_bounds * (delta_for_lasso_improvement_bounds / (radius_lasso ** 0.5 + delta_for_ls_improvement_bounds)) ** 0.5
    lower_bound_for_lasso_improvement_bound = delta_for_lasso_improvement_bounds * (delta_for_lasso_improvement_bounds / (radius_ls ** 0.5 + delta_for_lasso_improvement_bounds)) ** 0.5


    baseline = min(dist_ls, dist_lasso)
    projected = min(dist_proj_ls, dist_proj_lasso)
    improvement = baseline - projected
    success = improvement > success_tol

    if verbose:
        if dist_ls <= radius_ls:
            print(f"True in LS feasible space")

        if dist_lasso <= radius_lasso:
            print(f"True in Ridge feasible space")

    if verbose:
        print("[per-fit stats]")
        print(
            f"\tLS:    Ω error = {E_ls:.3e}, "
            f"Ξ error (vs True) = {dist_ls:.3e}, "
            f"feasible radius r = κ·EΩ = {radius_ls:.3e}, "
            f"projection Ξ error: {dist_proj_ls:.3e}"
        )
        print(
            f"\tRidge: Ω error = {E_lasso:.3e}, "
            f"Ξ error (vs True) = {dist_lasso:.3e}, "
            f"feasible radius r = κ·EΩ = {radius_lasso:.3e}, "
            f"projection Ξ error: {dist_proj_lasso:.3e}"
        )
        print("Sqrt for paper:")
        print(
            f"\tLS:    Ω error = {E_ls ** 0.5:.3e}, "
            f"Ξ error (vs True) = {dist_ls ** 0.5:.3e}, "
            f"feasible radius r = κ·EΩ = {radius_ls:.3e}, "
            f"projection Ξ error: {dist_proj_ls ** 0.5:.3e}"
        )
        print(f"Projection Improvement: {dist_ls ** 0.5 - dist_proj_ls ** 0.5:.3e}"
              f" {delta_for_ls_improvement_bounds:.3e}"
              f" {lower_bound_for_ls_improvement_bound:.3e}")

        print(
            f"\tRidge: Ω error = {E_lasso ** 0.5:.3e}, "
            f"Ξ error (vs True) = {dist_lasso ** 0.5:.3e}, "
            f"feasible radius r = κ·EΩ = {radius_lasso:.3e}, "
            f"projection Ξ error: {dist_proj_lasso ** 0.5:.3e}"
        )
        print(f"Projection Improvement: {dist_lasso ** 0.5 - dist_proj_lasso ** 0.5:.3e}"
              f" {delta_for_lasso_improvement_bounds:.3e}"
              f" {lower_bound_for_lasso_improvement_bound:.3e}")

        print(f"[cutoff={cutoff:.2f}, deg={degree}, sigma={sigma:.3g}, alpha={lasso_alpha:.2e}] "
              f"baseline={baseline:.4e}, projected={projected:.4e}, Δ={improvement:.4e}, "
              f"κ={kappa:.3g}, success={success}")

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(T, f_full, color='black', label="True", linewidth=2)
        line_ls, = plt.plot(T, ls_fn(T), label="LS Fit")
        plt.plot(T, proj_ls_fn(T), linestyle='--', color=line_ls.get_color(), label="Proj LS")
        line_lasso, = plt.plot(T, lasso_fn(T), label="Ridge Fit")
        plt.plot(T, proj_las_fn(T), linestyle='--', color=line_lasso.get_color(), label="Proj Ridge")
        # plt.plot(T, proj_las_fn(T), linestyle='--', color="orange", label="Proj Ridge")
        plt.axvline(cutoff, color='gray', linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.savefig(r"./results/geo_ls_and_lasso_projection_improvement.png")
        # plt.show()

        # --- zoomed-in view ---
        zoom_xlim = (0.6, 1)  # set these to your desired x-range
        zoom_ylim = (25_000, 250_000)  # set these to your desired y-range

        plt.figure(figsize=(8, 4))
        plt.plot(T, f_full, color='black', label="True", linewidth=2)
        # reuse colors from the first plot via line_ls / line_lasso
        plt.plot(T, ls_fn(T), label="LS Fit", color=line_ls.get_color())
        plt.plot(T, proj_ls_fn(T), linestyle='--', color=line_ls.get_color(), label="Proj LS")
        plt.plot(T, lasso_fn(T), label="Ridge Fit", color=line_lasso.get_color())
        plt.plot(T, proj_las_fn(T), linestyle='--', color=line_lasso.get_color(), label="Proj Ridge")
        plt.axvline(cutoff, color='gray', linestyle=':')
        plt.xlim(*zoom_xlim)
        plt.ylim(*zoom_ylim)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./results/geo_ls_and_lasso_projection_improvement_zoom.png", dpi=300)
        # plt.show()

    if return_metrics:
        metrics = dict(
            cutoff=float(cutoff), degree=int(degree), sigma=float(sigma), alpha=float(lasso_alpha),
            kappa=float(kappa),
            dist_ls=float(dist_ls), dist_lasso=float(dist_lasso),
            dist_proj_ls=float(dist_proj_ls), dist_proj_lasso=float(dist_proj_lasso),
            baseline=float(baseline), projected=float(projected),
            improvement=float(improvement),
            success=bool(success),
            best_baseline=("LS" if dist_ls <= dist_lasso else "Ridge"),
            best_projected=("Proj LS" if dist_proj_ls <= dist_proj_lasso else "Proj Ridge"),
        )
        return success, metrics

    return success


# ========= Initialize REAL DATA (REQUIRED) =========
DATA_FILE = r"./data/geomag_br_mu_from_user_wmm.txt"
init_real_truth(DATA_FILE)  # .txt auto-parsed: first column = wavelength, second = n
# ===================================================


if __name__ == "__main__":

    run_two_fitted_anchors_ls_ridge_experiment(
        cutoff=0.8,
        degree=8,
        sigma=0.0,
        lasso_alpha=1e-1,
        plot=True,
        verbose=True,
        success_tol=0.0,
        return_metrics=False,
    )