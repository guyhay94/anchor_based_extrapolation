import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

from legendre_utils import (
    fit_regular_ls,
    evaluate_legendre_polynomial_with_coefs,
    create_legendre_basis_functions_numpy,
    fit_lasso
)
from utils import (
    calc_l2_approximation_error, calc_inner_extrapolation_condition_number,
    convert_basis_function_orthogonality_domain,
    calc_extrapolation_condition_number_improved
)
from theoretical_tests import add_noise, project_function_into_l2_ball


# -- Core true functions --
def damped_oscillator(t: np.ndarray, zeta: float = 0.3, omega0: float = 2 * np.pi) -> np.ndarray:
    omega1 = omega0 * np.sqrt(1 - zeta ** 2)
    return np.exp(-zeta * omega0 * t) * (
        np.cos(omega1 * t) + (zeta / np.sqrt(1 - zeta ** 2)) * np.sin(omega1 * t)
    )

# -- Domain mapping --
def map_domain(t: np.ndarray, a: float, b: float) -> np.ndarray:
    # maps normalized t in [-1,1] to [a,b]
    return a + (t + 1) / 2 * (b - a)

# =========================
# Single-fit + constant-anchor (NO projections)
# =========================
def run_single_anchor_ls_or_lasso_experiment(
    *,
    cutoff: float = 0.9,
    degree: int = 12,
    sigma: float = 0.1,
    method: str = "ls",        # "ls" or "lasso"
    lasso_alpha: float = 1e-3, # used only if method="lasso"
    anchor_constant: float = 0.0,
    max_radius: float = None,  # cap applied to anchor radius
    plot: bool = False,
    verbose: bool = True,
    plot_ls: bool = False,
    max_around_anchor: float = 0
):
    """
    One predictive fit (LS or LASSO) vs. one constant anchor.
    No projections. We compute EΩ and Ξ errors for both, plus κ and the anchor's feasible radius.

    Printed fields (exact order/labels):
    ls/lasso fit:
    Omega error:
    Xi error:

    anchor function:
    Omega error:
    Xi error:
    kappa:
    radius:
    max_radius:
    max_radius_used:

    anchor_function error vs LS/LASSO in Xi
    """

    # Normalized domain
    T = np.linspace(-1, 1, 400)
    Omega = np.linspace(-1, cutoff, 100)
    Xi = np.linspace(cutoff, 1, 400)

    f_omega = damped_oscillator(Omega)
    f_xi = damped_oscillator(Xi)

    # Noise on Ω
    y_omega_noisy = add_noise(f_omega, sigma, seed=0)

    # Fit (LS or LASSO) on Ω
    ncoef = degree + 1
    if method.lower() == "ls":
        fit_coef = fit_regular_ls(x=Omega, y=y_omega_noisy, n=ncoef)
    elif method.lower() == "lasso":
        fit_coef = fit_lasso(x=Omega, y=y_omega_noisy, n=ncoef, alpha=lasso_alpha)
    else:
        raise ValueError("method must be 'ls' or 'lasso'.")

    # Constant anchor -> fit polynomial (same pipeline) to constant values on Ω
    y_anchor_omega = np.full_like(Omega, fill_value=anchor_constant, dtype=float)
    anchor_coef = np.round(fit_regular_ls(x=Omega, y=y_anchor_omega, n=ncoef), 10)

    # In-domain Ω errors EΩ
    y_fit_omega = evaluate_legendre_polynomial_with_coefs(Omega, fit_coef)
    y_anchor_fit_omega = evaluate_legendre_polynomial_with_coefs(Omega, anchor_coef)
    E_fit_omega = calc_l2_approximation_error(y_true=y_omega_noisy, y_pred=y_fit_omega, x_samples=Omega)
    E_anchor_omega = calc_l2_approximation_error(y_true=y_omega_noisy, y_pred=y_anchor_fit_omega, x_samples=Omega)

    # Ξ errors vs truth
    y_xi_fit    = evaluate_legendre_polynomial_with_coefs(Xi, fit_coef)
    y_xi_anchor = evaluate_legendre_polynomial_with_coefs(Xi, anchor_coef)
    Xi_err_fit    = calc_l2_approximation_error(y_true=f_xi, y_pred=y_xi_fit, x_samples=Xi)
    Xi_err_anchor = calc_l2_approximation_error(y_true=f_xi, y_pred=y_xi_anchor, x_samples=Xi)

    # κ and radius (radius is based on ANCHOR EΩ, like any other anchor)
    basis = create_legendre_basis_functions_numpy(degree)
    basis_functions_omega = convert_basis_function_orthogonality_domain(
        basis, basis_orthogonality_a=-1, basis_orthogonality_b=1, a_1=-1, b_1=cutoff
    )
    kappa = calc_extrapolation_condition_number_improved(
        phi_list=basis_functions_omega, domain_omega=(-1, cutoff), domain_xi=(cutoff, 1)
    )

    radius_unclamped = kappa * E_anchor_omega
    if max_radius is None:
        radius_used = radius_unclamped
    else:
        radius_used = min(radius_unclamped, max_radius)

    # Anchor vs Fit distance on Ξ
    Xi_err_anchor_vs_fit = calc_l2_approximation_error(y_xi_anchor, y_xi_fit, x_samples=Xi)

    fit_in_feasible_space = Xi_err_anchor_vs_fit <= radius_used

    if not fit_in_feasible_space:
        proj_coef = project_function_into_l2_ball(
            coeff_center=anchor_coef, coeff_target=fit_coef, radius=radius_used, x_xi=Xi
        )
    else:
        proj_coef = fit_coef

    y_omega_proj = evaluate_legendre_polynomial_with_coefs(Omega, proj_coef)
    y_xi_proj = evaluate_legendre_polynomial_with_coefs(Xi, proj_coef)

    proj_omega_error = calc_l2_approximation_error(y_true=f_omega, y_pred=y_omega_proj, x_samples=Omega)
    proj_xi_error = calc_l2_approximation_error(y_true=f_xi, y_pred=y_xi_proj, x_samples=Xi)

    # Relative errors on Ξ: divide reported errors by the true-function "error" from 0
    # i.e., E_Ξ(f, 0) over the same domain with the same metric.
    true_xi_from0 = calc_l2_approximation_error(
        y_true=f_xi, y_pred=np.zeros_like(f_xi, dtype=float), x_samples=Xi
    )
    denom_xi_root = (true_xi_from0 ** 0.5) if true_xi_from0 > 0 else np.nan

    rel_true_xi_error_root = (true_xi_from0 ** 0.5) / denom_xi_root
    rel_fit_xi_error_root = (Xi_err_fit ** 0.5) / denom_xi_root
    rel_proj_xi_error_root = (proj_xi_error ** 0.5) / denom_xi_root

    delta = calc_l2_approximation_error(y_true=y_xi_fit, y_pred=y_xi_proj, x_samples=Xi) ** 0.5
    lower_bound = delta * (delta / (radius_used ** 0.5 + delta)) ** 0.5

    # ====== PRINTS (exact labels/order) ======
    if verbose:
        print("ls/lasso fit:")
        print(f"\tOmega error: {E_fit_omega:.6e}")
        print(f"\tXi error: {Xi_err_fit:.6e}\n")

        print("anchor function:")
        print(f"\tOmega error: {E_anchor_omega:.6e}")
        print(f"\tXi error: {Xi_err_anchor:.6e}")
        print(f"\tkappa: {kappa:.6e}")
        print(f"\tradius: {radius_unclamped:.6e}")
        print(f"\tmax_radius: {('None' if max_radius is None else f'{max_radius:.6e}')}")
        print(f"\tmax_radius_used: {radius_used:.6e}")
        print(f"\tTrue in feasible space: {Xi_err_anchor <= radius_used}")

        print("anchor_function error vs LS/LASSO in Xi")
        print(f"\tls/lasso in anchor feasible space: {fit_in_feasible_space}")
        print(f"\tDist from anchor: {Xi_err_anchor_vs_fit:.6e}")

        print("Projection:")
        print(f"\tOmega error: {proj_omega_error:.6e}")
        print(f"\tXi error: {proj_xi_error:.6e}")
        print(f"Is proj better: {proj_xi_error < Xi_err_fit}")

        print("Relative errors in Xi on root-metric (divide by sqrt(true Xi error-from-0)):")
        print(f"\tTrue Xi error from 0: {true_xi_from0:.6e}")
        print(f"\tSqrt(True Xi error from 0): {true_xi_from0 ** 0.5:.6e}")
        print(f"\tTrue Xi sqrt-error (rel): {rel_true_xi_error_root:.6e}")
        print(f"\tLS Xi sqrt-error (rel): {rel_fit_xi_error_root:.6e}")
        print(f"\tProj Xi sqrt-error (rel): {rel_proj_xi_error_root:.6e}")

        print("Sqrt for paper:")
        print(f"\tLS Omega error: {E_fit_omega ** 0.5:.6e}")
        print(f"\tLS Xi error: {Xi_err_fit ** 0.5:.6e}\n")
        print(f"\tProj Omega error: {proj_omega_error ** 0.5:.6e}")
        print(f"\tProj Xi error: {proj_xi_error ** 0.5:.6e}")
        print(f"\tError reduction: {Xi_err_fit ** 0.5 - proj_xi_error ** 0.5:.6e}")
        print(f"\tError bounds: {lower_bound:.6e} - {delta:.6e}")

    if plot:
        T = np.linspace(-1, 1, 400)
        plt.figure(figsize=(8, 4))
        plt.plot(T, damped_oscillator(T), color='black', label="True", linewidth=2)
        if plot_ls:
            plt.plot(T, evaluate_legendre_polynomial_with_coefs(T, fit_coef), label=f"{method.upper()} Fit", color="orange")
        if not fit_in_feasible_space:
            plt.plot(T, evaluate_legendre_polynomial_with_coefs(T, proj_coef), label=f"Projection Fit", color="orange", linestyle="--")
        if max_around_anchor > 0:
            plt.plot(T, np.full_like(T, fill_value=anchor_constant + max_around_anchor, dtype=float),
                     label=f"Anchor upper bound", linestyle="--", color="blue")
            plt.plot(T, np.full_like(T, fill_value=anchor_constant - max_around_anchor, dtype=float),
                     label=f"Anchor lower bound", linestyle="-.", color="blue")

        plt.axvline(cutoff, color='gray', linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(r"./results/damped_oscilator_constant_anchors.png")

    return proj_xi_error < Xi_err_fit

if __name__ == "__main__":
    # example run
    max_around_anchor = 1.2
    _ = run_single_anchor_ls_or_lasso_experiment(
        cutoff=0.0,
        degree=12,
        sigma=0.5,
        method="lasso",        # "ls" or "lasso"
        lasso_alpha=1e-2,
        anchor_constant=0.0,
        # max_radius=(0.75*0.5) ** 2,
        max_radius=max_around_anchor ** 2,
        max_around_anchor=max_around_anchor,
        plot=True,
        verbose=True,
        plot_ls=True,
    )
