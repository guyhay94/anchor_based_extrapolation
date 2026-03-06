import random

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.optimize import minimize

from legendre_utils import evaluate_legendre_polynomial_with_coefs, fit_regular_ls, fit_lasso, \
    create_legendre_basis_functions_numpy
from utils import calc_l2_approximation_error, calc_extrapolation_condition_number, \
    convert_basis_function_orthogonality_domain, \
    calc_inner_extrapolation_condition_number, \
    calc_extrapolation_condition_number_improved, ell_bounds_known_sigma


def generate_sample_points(num_points: int, a: float = -1, b: float = 1) -> np.ndarray:
    """
    Generate sample points evenly spaced in the interval [-1, 1].
    """
    return np.linspace(a, b, num_points)


def add_noise(data: np.ndarray, sigma: float, seed: int = None) -> np.ndarray:
    """
    Add Gaussian noise with standard deviation sigma to the data.
    """
    np.random.seed(seed)
    noise: np.ndarray = np.random.normal(0, sigma, size=data.shape)
    return data + noise


def calc_coefficient_orientation(true_coefs: np.ndarray, estimated_coefs: np.ndarray) -> float:
    """
    Calculate the angle in degrees between true and estimated coefficient vectors.
    This version accounts for full 360° rotation (both upper and lower halves of the circle).
    """
    vec = estimated_coefs - true_coefs
    ref_vector = np.zeros_like(vec)
    ref_vector[0] = 1  # x-axis direction (1, 0, 0, ..., 0)

    # Reference vector along the x-axis
    ref_vector = np.zeros_like(vec)
    ref_vector[0] = 1  # Unit vector along x-axis (1, 0, 0, ..., 0)

    # Compute dot product and magnitudes
    dot_product = np.dot(vec, ref_vector)
    magnitude_vec = np.linalg.norm(vec)

    if magnitude_vec == 0:
        raise ValueError("Zero vector has no defined angle.")

    # Compute cosine of the angle
    cos_angle = dot_product / magnitude_vec
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid floating-point errors

    # Compute angle in radians
    angle_rad = np.arccos(cos_angle)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    # Adjust for full 360-degree range using atan2 (only for 2D or more)
    if len(vec) >= 2:
        angle_deg = np.degrees(np.arctan2(vec[1], vec[0]))

    # Ensure the angle is in [0, 360]
    angle_deg = angle_deg % 360

    return angle_deg


def project_function_into_l2_ball(
        coeff_center: np.ndarray,
        coeff_target: np.ndarray,
        radius: float,
        x_xi: np.ndarray
) -> np.ndarray:
    def objective(a):
        y_target = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                           coefs=coeff_target)
        y_pred = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                         coefs=a)
        return calc_l2_approximation_error(y_true=y_target, y_pred=y_pred, x_samples=x_xi)

    def constraint(a):
        y_center = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                           coefs=coeff_center)
        y_pred = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                         coefs=a)
        return radius - calc_l2_approximation_error(y_true=y_center, y_pred=y_pred, x_samples=x_xi)

    result = minimize(
        fun=objective,
        x0=coeff_target.copy(),
        constraints={'type': 'ineq', 'fun': constraint}
    )

    return result.x


def main_test_theorem_projection_is_always_optimal_to_xi(sigma_noise: float,
                                                         num_basis: int,
                                                         num_omega_points: int,
                                                         b: float,
                                                         alpha: float,
                                                         print_result: bool = True,
                                                         seed: int = None) -> Tuple[
    bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    true_coefs: np.ndarray = np.ones((num_basis,))

    x_omega = generate_sample_points(num_omega_points, a=-1, b=b)
    y_omega = evaluate_legendre_polynomial_with_coefs(x=x_omega,
                                                      coefs=true_coefs)
    x_xi = generate_sample_points(1000, a=b, b=1)
    y_omega_noisy = add_noise(y_omega, sigma_noise, seed=seed)

    ls_model_coefs = fit_regular_ls(x_omega, y_omega_noisy, num_basis)

    lasso_model_coefs = fit_lasso(x_omega, y_omega_noisy, num_basis, alpha=alpha)

    y_omega_ls = evaluate_legendre_polynomial_with_coefs(x=x_omega,
                                                         coefs=ls_model_coefs)

    basis_functions = create_legendre_basis_functions_numpy(num_basis)
    basis_functions_omega = convert_basis_function_orthogonality_domain(basis_functions,
                                                                        basis_orthogonality_a=-1,
                                                                        basis_orthogonality_b=1,
                                                                        a_1=-1,
                                                                        b_1=b)
    # convert to basis
    kappa_1 = calc_extrapolation_condition_number_improved(
        phi_list=basis_functions_omega,
        domain_omega=(-1, b),
        domain_xi=(b, 1)
    )

    kappa_2 = calc_inner_extrapolation_condition_number(
        phi_list=basis_functions,
        domain_xi=(b, 1)
    )

    kappa = np.nanmin(np.array([kappa_1, kappa_2], dtype=float))

    E_tilde = calc_l2_approximation_error(y_omega_noisy, y_omega_ls, x_omega)

    x_omega_many = generate_sample_points(1000, -1, b=b)
    y_omega_many = evaluate_legendre_polynomial_with_coefs(x=x_omega_many, coefs=true_coefs)
    y_omega_ls_many = evaluate_legendre_polynomial_with_coefs(x=x_omega_many, coefs=ls_model_coefs)
    E_omega_2 = calc_l2_approximation_error(y_omega_many, y_omega_ls_many, x_omega_many)
    E_omega_1 = calc_l2_approximation_error(y_omega, y_omega_ls, x_omega)
    good_omega_calc = abs(E_omega_1 - E_omega_2) / E_omega_2 <= 0.1

    good_e_tilde = (E_omega_2 - E_tilde) <= 0
    if not good_omega_calc:
        print("Bad calc")
    if not good_e_tilde:
        print("E_tilde is smaller")

    radius = kappa * E_tilde

    y_xi_true = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                        coefs=true_coefs)
    y_xi_ls = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                      coefs=ls_model_coefs)
    y_xi_lasso = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                         coefs=lasso_model_coefs)

    ls_real_dist = calc_l2_approximation_error(y_xi_ls, y_xi_true, x_samples=x_xi)
    real_in_search_space = ls_real_dist <= radius

    ls_lasso_dist = calc_l2_approximation_error(y_xi_ls, y_xi_lasso, x_samples=x_xi)
    lasso_in_search_space = ls_lasso_dist <= radius

    if lasso_in_search_space:
        ours_coefs = lasso_model_coefs
    else:
        ours_coefs = project_function_into_l2_ball(
            coeff_center=ls_model_coefs,
            coeff_target=lasso_model_coefs,
            radius=radius,
            x_xi=x_xi
        )

    y_xi_ours_coefs = evaluate_legendre_polynomial_with_coefs(x_xi,
                                                              coefs=ours_coefs)

    lasso_extrapolation_error = calc_l2_approximation_error(y_xi_true, y_xi_lasso, x_samples=x_xi)
    ls_extrapolation_error = calc_l2_approximation_error(y_xi_true, y_xi_ls, x_samples=x_xi)
    ours_extrapolation_error = calc_l2_approximation_error(y_xi_true, y_xi_ours_coefs, x_samples=x_xi)

    if print_result:
        print(f"Improved kappa: {kappa_1}")
        print(f"Inner kappa: {kappa_2}")
        print(f"E_Omega measured: {E_tilde}, E_Omega real: {E_omega_2}")
        print(f"Search space radius: {radius}")
        print(f"Lasso dist from LS: {ls_lasso_dist}, Radius: {radius}")
        print(f"Real dist from LS: {ls_real_dist}, Radius: {radius}")
        print(f"\tLasso is in search space: {lasso_in_search_space}")
        print(f"\tReal is in search space: {real_in_search_space}")
        print(f"\tLS error: {ls_extrapolation_error}, sqrt: {ls_extrapolation_error ** 0.5}")
        print(f"\tLasso error: {lasso_extrapolation_error}, sqrt: {lasso_extrapolation_error ** 0.5}")
        print(f"\tOurs functions error: {ours_extrapolation_error}, sqrt: {ours_extrapolation_error ** 0.5}")

    return np.all([not lasso_in_search_space,
                   real_in_search_space,
                   lasso_extrapolation_error < ls_extrapolation_error]), ls_model_coefs, lasso_model_coefs, ours_coefs, true_coefs, radius


def plot_ls_lasso_projection_graph(
        distances: List[float],
        orientations: List[float],
        names: List[str],
        colors: List[str],
        radius: float
) -> None:
    """
    Plots points in polar coordinates based on distances and orientations.

    Args:
        distances (List[float]): List of distances for each point.
        orientations (List[float]): List of orientations (angles in degrees).
        names (List[str]): List of labels for each point.
        colors (List[str]): List of colors for each point.
        radius (float): The radius to normalize distances.
    """
    if not (len(distances) == len(orientations) == len(names) == len(colors)):
        raise ValueError("The lengths of distances, orientations, names, and colors must be equal.")

    normalized_distances = [d / radius for d in distances]
    angles = [np.deg2rad(angle) for angle in orientations]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for r, theta, label, color in zip(normalized_distances, angles, names, colors):
        ax.plot(theta, r, 'o', label=label, color=color)
        ax.text(theta, r * 1.02, label, fontsize=14, ha='center', va='bottom')

    circle_angles = np.linspace(0, 2 * np.pi, 500)
    ax.plot(circle_angles, np.ones_like(circle_angles), linestyle='--', color='gray', linewidth=1)

    # Reduce whitespace
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(
        r"./results/ls_lasso_projection_graph_2.png",
        bbox_inches='tight', pad_inches=0.01, dpi=300
    )
    plt.close(fig)


def create_ls_lasso_projection_graph(sigma_noise: float,
                                     num_basis: int,
                                     num_omega_points: int,
                                     b: float,
                                     alpha: float,
                                     print_result: bool = True,
                                     seed: int = None) -> None:
    _, ls_coefs, lasso_coefs, ours_coefs, true_coefs, search_space_radius = main_test_theorem_projection_is_always_optimal_to_xi(
        sigma_noise=sigma_noise,
        num_basis=num_basis,
        num_omega_points=num_omega_points,
        b=b,
        alpha=alpha,
        print_result=print_result,
        seed=seed
    )
    lasso_orientation = calc_coefficient_orientation(true_coefs=ls_coefs, estimated_coefs=lasso_coefs)
    true_orientation = calc_coefficient_orientation(true_coefs=ls_coefs, estimated_coefs=true_coefs)
    ours_orientation = calc_coefficient_orientation(true_coefs=ls_coefs, estimated_coefs=ours_coefs)

    print(f"lasso orientation: {lasso_orientation}")
    print(f"ours orientation: {ours_orientation}")
    print(f"true orientation: {true_orientation}")

    print(f"lasso coefs: \n {lasso_coefs}")
    print(f"ls coefs: \n {ls_coefs}")
    print(f"true coefs: \n {true_coefs}")
    x_xi = generate_sample_points(1000, a=b, b=1)
    y_xi_true = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                        coefs=true_coefs)
    y_xi_ls = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                      coefs=ls_coefs)
    y_xi_lasso = evaluate_legendre_polynomial_with_coefs(x=x_xi,
                                                         coefs=lasso_coefs)

    y_xi_ours_coefs = evaluate_legendre_polynomial_with_coefs(x_xi,
                                                              coefs=ours_coefs)

    true_ls_dist = calc_l2_approximation_error(y_xi_ls, y_xi_true, x_samples=x_xi)
    lasso_ls_dist = calc_l2_approximation_error(y_xi_ls, y_xi_lasso, x_samples=x_xi)
    ours_ls_dist = calc_l2_approximation_error(y_xi_ls, y_xi_ours_coefs, x_samples=x_xi)

    delta = calc_l2_approximation_error(y_xi_ours_coefs, y_xi_lasso, x_samples=x_xi) ** 0.5
    upper_bound_for_error_reduction = delta
    lower_bound_for_error_reduction = delta * (delta / (search_space_radius + delta)) ** 0.5
    plot_ls_lasso_projection_graph(
        distances=[0, true_ls_dist, lasso_ls_dist, ours_ls_dist],
        orientations=[0, true_orientation, lasso_orientation, ours_orientation],
        names=["LS", "True", "Lasso", "Projection"],
        colors=["red", "black", "orange", "green"],
        radius=search_space_radius
    )

    plot_extrapolation_function_and_predictions_chebyshev(
        extrapolation_function_coefs=true_coefs,
        prediction_function_coefs_list=[ls_coefs, lasso_coefs, ours_coefs],
        prediction_function_name_list=["LS", "Lasso", "Projection"],
        colors=["red", "orange", "green"],
        xi_range=(b, 1.0),
        save_path=r"./results/ls_lasso_projection_graph_full_function_2.png"
    )


def plot_extrapolation_function_and_predictions_chebyshev(
        extrapolation_function_coefs: np.ndarray,
        prediction_function_coefs_list: List[np.ndarray],
        prediction_function_name_list: List[str],
        xi_range: Tuple[float, float],
        save_path: str,
        colors: List[str],
        plot_extended_range: bool = False,
) -> None:
    """
    Plot the extrapolation function and a list of prediction functions (in Chebyshev basis)
    over the specified xi range, with optional extension to [-1, b], and save the plot.

    Args:
        extrapolation_function_coefs (np.ndarray): Coefficients of the true extrapolation function.
        prediction_function_coefs_list (List[np.ndarray]): List of coefficients for prediction functions.
        prediction_function_name_list (List[str]): List of names for prediction functions.
        xi_range (Tuple[float, float]): Range of xi to evaluate (start, end).
        save_path (str): Path to save the plot.
        plot_extended_range (bool): Whether to also plot the function on [-1, b] using same colors.
    """

    a, b = xi_range
    fig, ax = plt.subplots(figsize=(8, 6))

    if plot_extended_range:
        # Extended range
        x_ext = generate_sample_points(1000, a=-1.0, b=b)
        y_true_ext = evaluate_legendre_polynomial_with_coefs(x_ext, coefs=extrapolation_function_coefs)
        ax.plot(x_ext, y_true_ext, label='True', color='black', linewidth=2)

        for coefs, label, color in zip(prediction_function_coefs_list, prediction_function_name_list, colors):
            y_ext = evaluate_legendre_polynomial_with_coefs(x_ext, coefs=coefs)
            ax.plot(x_ext, y_ext, color=color, label=label, linewidth=2)

    else:
        # Main range only
        x_main = generate_sample_points(1000, a=a, b=b)
        y_true_main = evaluate_legendre_polynomial_with_coefs(x_main, coefs=extrapolation_function_coefs)
        ax.plot(x_main, y_true_main, label='True', color='black', linewidth=2)

        for coefs, label, color in zip(prediction_function_coefs_list, prediction_function_name_list, colors):
            y_main = evaluate_legendre_polynomial_with_coefs(x_main, coefs=coefs)
            ax.plot(x_main, y_main, label=label, linewidth=2, color=color)

    # Add vertical dashed line at xi = b
    # ax.axvline(x=b, color='gray', linestyle='--', linewidth=1.5, label=r'$b$ (end of main range)')

    ax.set_xlabel(r'$\xi$', fontsize=18)
    ax.set_ylabel('Function value', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=23)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_kappa_comparison_with_improved(num_basis: int, b_values: np.ndarray) -> None:
    kappa_1_list = []
    kappa_2_list = []
    valid_b_values_kappa_2 = []

    basis_functions = create_legendre_basis_functions_numpy(num_basis)
    for b in b_values:
        basis_functions_omega = convert_basis_function_orthogonality_domain(
            basis_functions,
            basis_orthogonality_a=-1,
            basis_orthogonality_b=1,
            a_1=-1,
            b_1=b
        )

        kappa_2 = calc_extrapolation_condition_number(
            phi_list=basis_functions_omega,
            domain_omega=(-1, b),
            domain_xi=(b, 1)
        )
        kappa_1 = calc_extrapolation_condition_number_improved(
            phi_list=basis_functions_omega,
            domain_omega=(-1, b),
            domain_xi=(b, 1)
        )

        kappa_1_list.append(kappa_1)
        if kappa_2 is not None:
            kappa_2_list.append(kappa_2)
            valid_b_values_kappa_2.append(b)

    plt.figure(figsize=(8, 5))
    plt.plot(b_values, kappa_1_list, label=r'$\kappa_{spec}$', marker='o', color="orange")
    plt.plot(valid_b_values_kappa_2, kappa_2_list, label=r'$\kappa$', marker='+', linestyle='--', color="green")
    plt.yscale("log")
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fr"./results/kappa_comparison_vs_improved_{num_basis}")


def main_graph_for_lemma_explanation():
    radius = 5
    B = np.array([1, 4])  # Point inside the circle
    P = np.array([9, 0])  # Point outside on x-axis
    Z = np.array([0, 0])  # Extra reference point for visual balance

    Q = np.array([5, 0])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(False)

    circle = plt.Circle((0, 0), radius=radius, color='black', fill=False, lw=1.5)
    ax.add_artist(circle)

    ax.plot([-6, P[0]], [0, P[1]], 'k-', lw=1)  # x-axis line
    ax.plot([B[0], Q[0]], [B[1], Q[1]], 'r-', lw=2)  # BQ in red
    ax.plot([B[0], P[0]], [B[1], P[1]], 'r-', lw=2)  # BP in red

    for point, label, color, valign in [
        (Z, 'Z', 'black', 'center'),
        (B, 'B', 'blue', 'center'),
        (Q, 'Q', 'red', 'top'),
        (P, 'P', 'purple', 'top')
    ]:
        ax.scatter(*point, color=color, zorder=5)
        ax.text(point[0], point[1], f'  {label}', fontsize=12, va=valign)

    plt.ylim(-6, 6)
    plt.axis('off')
    # plt.show()
    plt.savefig(r"./results/theorem_explanation_graph.png")


if __name__ == "__main__":
    main_graph_for_lemma_explanation()

    # # b values to test (you can increase the resolution if needed)
    b_vals = np.linspace(0.90, 0.99, 50)
    #
    # # Generate plots
    plot_kappa_comparison_with_improved(num_basis=15, b_values=b_vals)
    plot_kappa_comparison_with_improved(num_basis=10, b_values=b_vals)
    plot_kappa_comparison_with_improved(num_basis=5, b_values=b_vals)


    param = {'sigma_noise': 0.01,
             'num_basis': 20,
             'num_omega_points': 50,
             'b': 0.90,
             'alpha': 0.001,
             'seed': 0}

    create_ls_lasso_projection_graph(
        sigma_noise=param['sigma_noise'],
        num_basis=param['num_basis'],
        num_omega_points=param['num_omega_points'],
        b=param['b'],
        alpha=param['alpha'],
        print_result=True,
        seed=param['seed']
    )

