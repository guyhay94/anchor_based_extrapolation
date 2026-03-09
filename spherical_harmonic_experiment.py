import numpy as np
import pandas as pd
from typing import Tuple
from scipy.special import sph_harm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from find_simple_baselines import find_best_coefs_for_specific_basis_functions
from utils import fit_basis_functions
from scipy.integrate import simpson
from scipy.optimize import minimize


def calc_l2_inner_product_on_sphere(
        f: np.ndarray,
        g: np.ndarray,
        points: np.ndarray,
) -> float:
    """
    Compute L2 inner product on sphere surface: <f, g> = ∫∫ f(θ,φ) g(θ,φ) sin(φ) dφ dθ.

    Parameters
    ----------
    f : np.ndarray, shape (N,)
        First function values evaluated at points.
    g : np.ndarray, shape (N,)
        Second function values evaluated at points.
    points : np.ndarray, shape (N, 2)
        Points on sphere, columns [theta, phi]. Assumed to lie on a tensor grid
        with n_phi unique phi values and n_theta unique theta values.

    Returns
    -------
    float
        Inner product <f, g>.
    """
    if f.shape != g.shape or points.shape[0] != f.shape[0]:
        raise ValueError("Shapes of f, g, and points are inconsistent.")

    theta = points[:, 0]
    phi = points[:, 1]

    # Recover grid sizes (assumes tensor product grid)
    theta_vals = np.unique(theta)
    phi_vals = np.unique(phi)
    n_theta = theta_vals.size
    n_phi = phi_vals.size

    if n_theta * n_phi != points.shape[0]:
        raise ValueError("Points are not on a regular tensor grid (phi, theta).")

    # Reshape to (n_phi, n_theta) grid
    f_grid = f.reshape(n_phi, n_theta)
    g_grid = g.reshape(n_phi, n_theta)
    phi_grid = phi.reshape(n_phi, n_theta)

    # Integrand: f * g * sin(phi)
    integrand = f_grid * g_grid * np.sin(phi_grid)

    # Integrate over phi (axis=0), then over theta
    # phi_vals and theta_vals must be sorted; np.unique returns sorted arrays.
    tmp = simpson(integrand, x=phi_vals, axis=0)  # shape (n_theta,)
    inner_product = simpson(tmp, x=theta_vals)  # scalar

    # Clip tiny negative values due to roundoff (for squared norms)
    if inner_product < 0:
        if inner_product > -1e-12 * abs(inner_product):
            inner_product = 0.0

    return float(inner_product)


def calc_gram_matrix_on_sphere(
        basis_functions,
        domain_phi: Tuple[float, float],
        n_quad: int = 1000,
):
    """
    Calculate Gram matrix on sphere surface for given basis functions.

    The Gram matrix G[i,j] = <phi_i, phi_j>_domain, where the inner product
    is computed over the spherical region with surface element sin(phi) dphi dtheta.

    Parameters
    ----------
    basis_functions : list of callables
        Each basis function takes an array of points of shape (N, 2) with
        columns (theta, phi), and returns an array of length N.
    domain_phi : (float, float)
        Phi range for the domain: (start_phi, end_phi).
    n_quad : int
        Number of quadrature points in each angular direction.

    Returns
    -------
    np.ndarray, shape (d, d)
        Gram matrix G where G[i,j] = <phi_i, phi_j>_domain.
    """
    phi_start, phi_end = domain_phi

    # 1D quadrature points
    theta_points = np.linspace(0.0, 2.0 * np.pi, n_quad)
    phi_points = np.linspace(phi_start, phi_end, n_quad)

    # 2D grids: shape (n_quad, n_quad), rows = phi, cols = theta
    theta_grid, phi_grid = np.meshgrid(theta_points, phi_points)

    # Flattened points for evaluation: (theta, phi)
    points = np.column_stack([theta_grid.ravel(), phi_grid.ravel()])

    d = len(basis_functions)

    # Precompute basis values on the grid for efficiency
    Phi = np.empty((d, n_quad, n_quad))
    for k, bf in enumerate(basis_functions):
        Phi[k] = bf(points).reshape(n_quad, n_quad)

    # Calculate Gram matrix G[i,j] = <phi_i, phi_j>_domain using inner product function
    G = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            # Flatten the grids to vectors for inner product calculation
            phi_i_flat = Phi[i].ravel()
            phi_j_flat = Phi[j].ravel()
            G[i, j] = calc_l2_inner_product_on_sphere(phi_i_flat, phi_j_flat, points)
            G[j, i] = G[i, j]  # Symmetric matrix

    return G


# Spherical harmonics functions
def real_spherical_harmonics(x: np.ndarray, l: int, m: int):
    """Compute real spherical harmonics Y_l^m."""
    if m == 0:
        output = sph_harm(m, l, x[:, 0], x[:, 1])
    elif m < 0:
        output = 1/(1j * 2 ** 0.5) * (sph_harm(-m, l, x[:, 0], x[:, 1]) - (-1)**m * sph_harm(m, l, x[:, 0], x[:, 1]))
    else:
        output = 1 / (2 ** 0.5) * (sph_harm(m, l, x[:, 0], x[:, 1]) + (-1) ** m * sph_harm(-m, l, x[:, 0], x[:, 1]))
    return np.real(output)


def generate_spherical_harmonics_l_from_degree(degree: int, deviation: int = 3) -> int:
    """Generate l (angular momentum quantum number) from degree."""
    if degree <= 0:
        return 0
    if 1 <= degree <= 3:
        return 1
    return generate_spherical_harmonics_l_from_degree(degree - deviation, deviation+2) + 1


def generate_spherical_harmonics_m_from_degree_and_l(degree: int, l: int) -> int:
    """Generate m (magnetic quantum number) from degree and l."""
    starting_degree = l**2  # easy proof by injunction
    degree_in_degree = degree - starting_degree
    return l - degree_in_degree


def generate_spherical_harmonics_l_m_from_degree(degree: int) -> Tuple[int, int]:
    """Generate (l, m) quantum numbers from degree."""
    l = generate_spherical_harmonics_l_from_degree(degree)
    m = generate_spherical_harmonics_m_from_degree_and_l(degree, l)
    return l, m


def create_spherical_harmonics_basis_element(degree: int = 7):
    """Create a basis element function for a given degree."""
    l, m = generate_spherical_harmonics_l_m_from_degree(degree)
    def basis_func(x):
        # Ensure x is numpy array
        x = np.asarray(x, dtype=np.float64)
        return real_spherical_harmonics(x, l, m)
    return basis_func


def create_spherical_harmonics_functions(deg: int = 7):
    """Create list of spherical harmonics basis functions up to degree deg."""
    return [create_spherical_harmonics_basis_element(d) for d in range(deg)]


def create_spherical_harmonics_points(num_input_points: int = 100,
                                     omega_size: float = np.pi / 4):
    """
    Create training (Omega) and extrapolation (Xi) points for spherical harmonics.
    
    Xi (extrapolation) is always the upper half of the sphere: phi from 0 to pi/2
    Omega (training) is part of the bottom half: phi from (pi - omega_size) to pi
    
    Parameters:
    -----------
    num_input_points : int
        Number of training points (will create sqrt(num_input_points) x sqrt(num_input_points) grid)
    omega_size : float
        Size of Omega region in radians. For example, pi/4 means half of the bottom half.
        Omega will be phi from (pi - omega_size) to pi.
        
    Returns:
    --------
    training_points : np.ndarray
        Training points (Omega region), shape (N, 2) where columns are [theta, phi]
    extrapolation_points : np.ndarray
        Extrapolation points (Xi region), shape (M, 2) where columns are [theta, phi]
    """
    # Xi (extrapolation): always upper half of sphere, phi from 0 to pi/2
    start_theta_xi = 0
    end_theta_xi = 2 * np.pi
    start_phi_xi = 0
    end_phi_xi = np.pi / 2
    
    # Omega (training): bottom part of sphere, phi from (pi - omega_size) to pi
    start_theta_omega = 0
    end_theta_omega = 2 * np.pi
    start_phi_omega = np.pi - omega_size
    end_phi_omega = np.pi
    
    # Create training grid (Omega)
    n_train = int(num_input_points ** 0.5)
    theta_train_points = np.linspace(start_theta_omega, end_theta_omega, n_train)
    phi_train_points = np.linspace(start_phi_omega, end_phi_omega, n_train)
    theta_train_grid, phi_train_grid = np.meshgrid(theta_train_points, phi_train_points)
    training_points = np.column_stack([theta_train_grid.reshape(-1), phi_train_grid.reshape(-1)])
    
    # Create extrapolation grid (Xi) - 100x100
    theta_ext_points = np.linspace(start_theta_xi, end_theta_xi, 100)
    phi_ext_points = np.linspace(start_phi_xi, end_phi_xi, 100)
    theta_ext_grid, phi_ext_grid = np.meshgrid(theta_ext_points, phi_ext_points)
    extrapolation_points = np.column_stack([theta_ext_grid.reshape(-1), phi_ext_grid.reshape(-1)])
    
    return training_points, extrapolation_points


def evaluate_spherical_harmonics(coefs: np.ndarray, points: np.ndarray, basis_functions):
    """
    Evaluate spherical harmonics at given points using given coefficients and basis functions.
    
    Parameters:
    -----------
    coefs : np.ndarray
        Coefficients for basis functions, shape (deg,)
    points : np.ndarray
        Points to evaluate at, shape (N, 2) where columns are [theta, phi]
    basis_functions : list
        List of basis functions (can be TensorFlow-compatible or numpy-compatible)
        
    Returns:
    --------
    np.ndarray : Evaluated function values, shape (N,)
    """
    # Evaluate each basis function and sum with coefficients
    result = np.zeros(points.shape[0])
    for i, basis_func in enumerate(basis_functions):
        if i < len(coefs):
            basis_values = basis_func(points)
            result = result + coefs[i] * basis_values
    return result


def create_anchor_functions_with_bounds(extrapolation_points: np.ndarray,
                                        y_ext_true: np.ndarray):
    """
    Create anchor function with bounds:
    - Anchor constant = (true_max + true_min) / 2
    - Error radius = (Xi_size * (2 - 1))^2 = Xi_size^2
    where Xi_size is the area of the extrapolation region
    
    Parameters:
    -----------
    extrapolation_points : np.ndarray
        Extrapolation points, shape (M, 2) where columns are [theta, phi]
    y_train_true : np.ndarray
        True training data (without noise) - used to compute true_min and true_max
        
    Returns:
    --------
    dict : Dictionary containing anchor function results
    """
    # Compute true value range
    true_min = np.min(y_ext_true)
    true_max = np.max(y_ext_true)
    
    # Anchor function: constant value = (true_max + true_min) / 2
    anchor_constant = (true_max + true_min) / 2
    
    # Create constant anchor function on extrapolation points
    y_anchor_ext = np.full(len(extrapolation_points), fill_value=anchor_constant)
    
    # Compute Xi size (area of extrapolation region on sphere)
    theta_ext = extrapolation_points[:, 0]
    phi_ext = extrapolation_points[:, 1]
    n_theta_ext = len(np.unique(theta_ext))
    n_phi_ext = len(np.unique(phi_ext))
    phi_grid = phi_ext.reshape(n_phi_ext, n_theta_ext)
    phi_values = phi_grid[:, 0]
    theta_values = theta_ext.reshape(n_phi_ext, n_theta_ext)[0, :]
    area_elements = np.sin(phi_grid)
    
    # Ensure phi_values are in ascending order for integration
    # (phi may be in descending order, which would give negative area)
    phi_sorted_indices = np.argsort(phi_values)
    phi_values_sorted = phi_values[phi_sorted_indices]
    area_elements_sorted = area_elements[phi_sorted_indices, :]
    
    integrated_area = np.array([simpson(area_elements_sorted[:, k], phi_values_sorted) for k in range(n_theta_ext)])
    xi_size = simpson(integrated_area, theta_values)
    
    # Ensure Xi_size is positive (area should always be positive)
    if xi_size < 0:
        print(f"Warning: Xi_size is negative ({xi_size:.6f}), taking absolute value")
        xi_size = abs(xi_size)

    radius = xi_size * (true_max - anchor_constant) ** 2

    
    return {
        'anchor_constant': anchor_constant,
        'y_anchor_ext': y_anchor_ext,
        'radius': radius,
        'xi_size': xi_size,
        'true_min': true_min,
        'true_max': true_max
    }


def project_spherical_harmonics_into_l2_ball(
        coeff_anchor: np.ndarray,
        coeff_target: np.ndarray,
        radius: float,
        extrapolation_points: np.ndarray,
        basis_functions
) -> np.ndarray:

    def objective(coeff_proj):
        y_target = evaluate_spherical_harmonics(coeff_target, extrapolation_points, basis_functions)
        y_proj = evaluate_spherical_harmonics(coeff_proj, extrapolation_points, basis_functions)
        return calc_l2_error_on_sphere(y_target, y_proj, extrapolation_points)

    def constraint(coeff_proj):
        y_anchor = evaluate_spherical_harmonics(coeff_anchor, extrapolation_points, basis_functions)
        y_proj = evaluate_spherical_harmonics(coeff_proj, extrapolation_points, basis_functions)
        error = calc_l2_error_on_sphere(y_anchor, y_proj, extrapolation_points)
        return radius - error   # want >= 0

    result = minimize(
        fun=objective,
        x0=coeff_target.copy(),
        constraints={'type': 'ineq', 'fun': constraint},
        method='SLSQP',
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    # Numerical tolerance for constraint satisfaction
    tol = 1e-6   # for example

    constraint_value = constraint(result.x)
    if constraint_value < -tol:
        # real violation
        print(f"Warning: Projection constraint not satisfied. "
              f"Constraint value: {constraint_value:.6e}, radius: {radius:.6e}")
        return coeff_anchor.copy()
    else:
        # treat tiny negative as zero
        return result.x



def calc_l2_error_on_sphere(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    points: np.ndarray,
) -> float:
    """
    Compute L2 error on sphere surface with surface element sin(phi) dphi dtheta.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        True function values.
    y_pred : np.ndarray, shape (N,)
        Predicted function values.
    points : np.ndarray, shape (N, 2)
        Points on sphere, columns [theta, phi]. Assumed to lie on a tensor grid
        with n_phi unique phi values and n_theta unique theta values.

    Returns
    -------
    float
        L2 error squared (integral of squared difference over the sphere region).
    """
    if y_true.shape != y_pred.shape or points.shape[0] != y_true.shape[0]:
        raise ValueError("Shapes of y_true, y_pred, and points are inconsistent.")

    diff = y_true - y_pred
    theta = points[:, 0]
    phi = points[:, 1]

    # Recover grid sizes (assumes tensor product grid)
    theta_vals = np.unique(theta)
    phi_vals = np.unique(phi)
    n_theta = theta_vals.size
    n_phi = phi_vals.size

    if n_theta * n_phi != points.shape[0]:
        raise ValueError("Points are not on a regular tensor grid (phi, theta).")

    # Reshape to (n_phi, n_theta) grid
    # Assumes ordering consistent with reshape; if you built points with meshgrid
    # from (phi_vals, theta_vals) in that order, this is correct.
    diff_grid = diff.reshape(n_phi, n_theta)
    phi_grid = phi.reshape(n_phi, n_theta)

    # Integrand: (difference)^2 * sin(phi)
    integrand = diff_grid**2 * np.sin(phi_grid)

    # Integrate over phi (axis=0), then over theta
    # phi_vals and theta_vals must be sorted; np.unique returns sorted arrays.
    tmp = simpson(integrand, x=phi_vals, axis=0)      # shape (n_theta,)
    l2_error_squared = simpson(tmp, x=theta_vals)     # scalar

    # Clip tiny negative values due to roundoff
    if l2_error_squared < 0:
        if l2_error_squared > -1e-12 * abs(l2_error_squared):
            l2_error_squared = 0.0
        else:
            # Large negative -> something is wrong with grid or integrand
            raise RuntimeError(
                f"Computed negative L2 error squared: {l2_error_squared}"
            )

    return float(l2_error_squared)



def plot_spherical_harmonics_error(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   extrapolation_points: np.ndarray,
                                   error_min: float = None,
                                   error_max: float = None,
                                   name: str = "Unknown",
                                   not_error: bool = False,
                                   save_path: str = None):
    tick_size = 13.5
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    
    # Reshape points and errors for plotting
    theta = extrapolation_points[:, 0].reshape((100, -1))
    phi = extrapolation_points[:, 1].reshape((100, -1))
    
    if not not_error:
        errors = np.abs(y_true - y_pred).reshape((100, -1))
    else:
        errors = y_pred.reshape((100, -1))
    
    error_max = errors.max() if error_max is None else error_max
    error_min = errors.min() if error_min is None else error_min
    
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)

    cmap = cm.jet

    norm = plt.Normalize(error_min, error_max)
    rgba = cmap(norm(errors))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, facecolors=rgba,
        linewidth=0, antialiased=False, alpha=0.5)

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    ax.view_init(elev=25, azim=130)
    ax.set_zlim(0, 2)

    if save_path is None:
        if not_error:
            save_path = f"./results/spherical_harmonics/{name}_true_values.png"
        else:
            save_path = f"./results/spherical_harmonics/{name}_error_values.png"
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    return errors.min(), errors.max()


def test_ls_and_lasso_with_given_coefs(coefs: np.ndarray,
                                       deg: int = 9,
                                       method: str = "ls",
                                       lasso_alpha: float = 0.01,
                                       omega_size: float = np.pi / 4,
                                       num_input_points: int = 100,
                                       add_noise: bool = True,
                                       snr: float = 35.0):
    """
    Test LS or LASSO on spherical harmonics with given coefficients.
    
    Parameters:
    -----------
    coefs : np.ndarray
        True coefficients to use, shape (deg,). If None, uses ones.
    deg : int
        Degree of spherical harmonics
    method : str
        Method to use: "ls" for Least Squares or "lasso" for LASSO
    lasso_alpha : float
        LASSO regularization parameter (only used if method="lasso")
    omega_size : float
        Size of Omega (training) region in radians. Omega is phi from (pi - omega_size) to pi.
        Xi (extrapolation) is always the upper half: phi from 0 to pi/2.
    num_input_points : int
        Number of training points (will create sqrt(num_input_points) x sqrt(num_input_points) grid)
    add_noise : bool
        Whether to add noise to training data
    snr : float
        Signal-to-noise ratio if adding noise
    """
    # Use ones if coefs not provided
    if coefs is None:
        coefs = np.ones(deg)
    elif len(coefs) != deg:
        raise ValueError(f"coefs length ({len(coefs)}) must match deg ({deg})")
    
    # Create points
    training_points, extrapolation_points = create_spherical_harmonics_points(
        num_input_points=num_input_points,
        omega_size=omega_size
    )
    
    # Create basis functions
    basis_functions = create_spherical_harmonics_functions(deg)
    
    # Evaluate true function
    y_train_true = evaluate_spherical_harmonics(coefs, training_points, basis_functions)
    y_ext_true = evaluate_spherical_harmonics(coefs, extrapolation_points, basis_functions)
    
    # Add noise if requested
    if add_noise:
        # Calculate noise level from SNR
        signal_power = np.mean(y_train_true ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), y_train_true.shape)
        y_train = y_train_true + noise
    else:
        y_train = y_train_true
    
    # Fit chosen method
    if method.lower() == "ls":
        pred_coefs = find_best_coefs_for_specific_basis_functions(
            training_points, y_train, basis_functions
        )
    elif method.lower() == "lasso":
        pred_coefs = fit_basis_functions(
            basis_functions=basis_functions,
            x_train=training_points,
            y_train=y_train,
            method="lasso",
            alpha=lasso_alpha
        )
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'ls' or 'lasso'")
    
    # Create anchor function with bounds
    anchor_results = create_anchor_functions_with_bounds(
        extrapolation_points=extrapolation_points,
        y_ext_true=y_ext_true
    )
    
    y_anchor_ext = anchor_results['y_anchor_ext']
    radius = anchor_results['radius']
    anchor_constant = anchor_results['anchor_constant']
    
    # Create anchor coefficients: [anchor_constant, 0, 0, ..., 0]
    anchor_coefs = np.zeros(deg)
    anchor_coefs[0] = anchor_constant
    
    # Evaluate predictions
    y_train_pred = evaluate_spherical_harmonics(pred_coefs, training_points, basis_functions)
    y_ext_pred = evaluate_spherical_harmonics(pred_coefs, extrapolation_points, basis_functions)

    Xi_err_fit = calc_l2_error_on_sphere(y_ext_true, y_ext_pred, extrapolation_points) ** 0.5
    
    # Compute distance between prediction and anchor on extrapolation region
    Xi_err_anchor_vs_fit = calc_l2_error_on_sphere(y_ext_pred, y_anchor_ext, extrapolation_points) ** 0.5
    
    # Compute anchor function's error to true function
    Xi_err_anchor_vs_true = calc_l2_error_on_sphere(y_ext_true, y_anchor_ext, extrapolation_points) ** 0.5
    
    # Project LS/LASSO prediction onto feasible space
    proj_coefs = project_spherical_harmonics_into_l2_ball(
        coeff_anchor=anchor_coefs,
        coeff_target=pred_coefs,
        radius=radius,
        extrapolation_points=extrapolation_points,
        basis_functions=basis_functions
    )
    
    # Evaluate projected function
    y_ext_proj = evaluate_spherical_harmonics(proj_coefs, extrapolation_points, basis_functions)
    
    # Compute errors for projected function
    Xi_err_proj = calc_l2_error_on_sphere(y_ext_true, y_ext_proj, extrapolation_points) ** 0.5
    Xi_err_anchor_vs_proj = calc_l2_error_on_sphere(y_ext_proj, y_anchor_ext, extrapolation_points) ** 0.5
    
    # Compute metrics
    metrics = {
        method.upper(): {
            'Xi_err': Xi_err_fit,
            'Xi_err_anchor_vs_fit': Xi_err_anchor_vs_fit,
            'coefs': pred_coefs
        },
        f'{method.upper()}_proj': {
            'Xi_err': Xi_err_proj,
            'Xi_err_anchor_vs_fit': Xi_err_anchor_vs_proj,
            'coefs': proj_coefs
        },
        'True': {
            'coefs': coefs,
            'Xi_err_anchor_vs_true': Xi_err_anchor_vs_true
        }
    }
    
    # Store method name for easier access
    method_key = method.upper()
    
    return {
        'training_points': training_points,
        'extrapolation_points': extrapolation_points,
        'y_train_true': y_train_true,
        'y_ext_true': y_ext_true,
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'y_ext_pred': y_ext_pred,
        'y_ext_proj': y_ext_proj,
        'y_anchor_ext': anchor_results['y_anchor_ext'],
        'radius': anchor_results['radius'],
        'xi_size': anchor_results['xi_size'],
        'true_min': anchor_results['true_min'],
        'true_max': anchor_results['true_max'],
        'anchor_constant': anchor_results['anchor_constant'],
        'method': method.upper(),
        'metrics': metrics
    }


def plot_results(results: dict, name_prefix: str = "test"):
    """
    Plot results for the chosen method (LS or LASSO).
    """
    training_points = results['training_points']
    extrapolation_points = results['extrapolation_points']
    y_ext_true = results['y_ext_true']
    y_ext_pred = results['y_ext_pred']
    method = results.get('method', 'LS')
    
    # Compute min/max values for scaling (without plotting)
    # True function values
    true_values = y_ext_true.reshape((100, -1))
    value_min_tr = true_values.min()
    value_max_tr = true_values.max()
    
    # Error values
    errors = np.abs(y_ext_true - y_ext_pred).reshape((100, -1))
    error_min = errors.min()
    error_max = errors.max()
    
    # Function values
    pred_values = y_ext_pred.reshape((100, -1))
    value_min_pred = pred_values.min()
    value_max_pred = pred_values.max()
    
    # Plot error values
    print("Plotting scaled error values...")
    
    plot_spherical_harmonics_error(
        y_ext_true, y_ext_pred, extrapolation_points,
        name=f"{name_prefix}_{method}_error_values_scaled",
        not_error=False,
        error_min=error_min,
        error_max=error_max
    )
    
    # Plot function values with common scale
    print("Plotting scaled function values...")
    
    # Include anchor function in value range
    value_min = value_min_tr
    value_max = value_max_tr
    if 'y_anchor_ext' in results:
        anchor_values = results['y_anchor_ext'].reshape((100, -1))
        value_min = min(value_min, value_min_pred, anchor_values.min())
        value_max = max(value_max, value_max_pred, anchor_values.max())
    else:
        value_min = min(value_min, value_min_pred)
        value_max = max(value_max, value_max_pred)
    
    # Plot true function values
    plot_spherical_harmonics_error(
        y_ext_true, y_ext_true, extrapolation_points,
        name=f"{name_prefix}_true_true_values_scaled",
        not_error=True,
        error_min=value_min,
        error_max=value_max
    )
    
    # Plot prediction
    plot_spherical_harmonics_error(
        y_ext_true, y_ext_pred, extrapolation_points,
        name=f"{name_prefix}_{method}_true_values_scaled",
        not_error=True,
        error_min=value_min,
        error_max=value_max
    )
    
    # Plot anchor function
    if 'y_anchor_ext' in results:
        plot_spherical_harmonics_error(
            results['y_anchor_ext'], results['y_anchor_ext'], extrapolation_points,
            name=f"{name_prefix}_anchor_true_values_scaled",
            not_error=True,
            error_min=value_min,
            error_max=value_max
        )


def main():
    # ========== Parameters ==========
    deg = 16
    method = "ls"  # "ls"
    lasso_alpha = 1.0
    omega_size = np.pi / 4  # Size of Omega region (training)
    num_input_points_list = [30, 50, 100, 200, 500, 1000, 2000, 5000]  # List of input points to test
    add_noise = True
    snr = 30.0
    random_seed = 42  # Base random seed for reproducibility
    num_seeds = 10
    
    # Use fixed coefficients: 1 * ones
    coefs = 1 * np.ones(deg)
    
    # Storage for results
    input_points_results = []
    ls_errors = []
    projected_errors = []
    ls_errors_std = []
    projected_errors_std = []
    lower_bounds = []
    upper_bounds = []
    error_improvements = []
    
    print("=" * 80)
    print(f"Testing with {method.upper()} and coefficients: {coefs}")
    print(f"Testing with input points: {num_input_points_list}")
    print(f"Testing with seeds: {[random_seed + i for i in range(num_seeds)]}")
    print("=" * 80)
    
    # Loop over different numbers of input points
    for num_input_points in num_input_points_list:
        print(f"\n{'=' * 80}")
        print(f"Running with num_input_points = {num_input_points}")
        print(f"{'=' * 80}")

        per_seed_ls_errors = []
        per_seed_projected_errors = []
        per_seed_lower_bounds = []
        per_seed_upper_bounds = []
        per_seed_error_improvements = []

        for seed_idx in range(num_seeds):
            current_seed = random_seed + seed_idx
            np.random.seed(current_seed)
            print(f"  Seed {current_seed}...")

            # Run test
            results = test_ls_and_lasso_with_given_coefs(
                coefs=coefs,
                deg=deg,
                method=method,
                lasso_alpha=lasso_alpha,
                omega_size=omega_size,
                num_input_points=num_input_points,
                add_noise=add_noise,
                snr=snr
            )

            # Check if method is in feasible space
            method_key = results['method']
            method_proj_key = f'{method_key}_proj'
            Xi_err_anchor_vs_fit = results['metrics'][method_key]['Xi_err_anchor_vs_fit']
            radius = results['radius']
            in_feasible = Xi_err_anchor_vs_fit <= radius ** 0.5

            # Get metrics for original method
            true_min = results.get('true_min', None)
            true_max = results.get('true_max', None)
            anchor_constant = results.get('anchor_constant', None)
            xi_size = results.get('xi_size', None)
            Xi_err = results['metrics'][method_key]['Xi_err']
            Xi_err_anchor_vs_true = results['metrics']['True']['Xi_err_anchor_vs_true']

            # Get metrics for projected method
            Xi_err_proj = results['metrics'][method_proj_key]['Xi_err']
            Xi_err_anchor_vs_proj = results['metrics'][method_proj_key]['Xi_err_anchor_vs_fit']
            in_feasible_proj = Xi_err_anchor_vs_proj <= radius ** 0.5

            # Calculate distance between prediction and projection (upper bound)
            y_ext_pred = results['y_ext_pred']
            y_ext_proj = results['y_ext_proj']
            extrapolation_points = results['extrapolation_points']
            delta = calc_l2_error_on_sphere(y_ext_pred, y_ext_proj, extrapolation_points) ** 0.5
            upper_bound = delta
            # radius is L2, therefore we need to sqrt it
            lower_bound = delta * (delta / (delta + (radius ** 0.5))) ** 0.5

            # Calculate error improvement
            error_improvement = Xi_err - Xi_err_proj

            per_seed_ls_errors.append(Xi_err)
            per_seed_projected_errors.append(Xi_err_proj)
            per_seed_lower_bounds.append(lower_bound)
            per_seed_upper_bounds.append(upper_bound)
            per_seed_error_improvements.append(error_improvement)

            # Print metrics
            print(f"\n{'*' * 80}")
            print(f"Results - {method_key} (num_input_points={num_input_points}, seed={current_seed})")
            print(f"{'*' * 80}")
            print(f"1. Anchor radius: {radius ** 0.5:.6e}")
            print(f"2. Xi_size: {xi_size:.6e}")
            print(f"3. True value range: min={true_min:.6f}, max={true_max:.6f}, anchor_constant={anchor_constant:.6f}")
            print(f"4. {method_key} error from true (Xi_err): {Xi_err:.6e}")
            print(f"5. {method_key} error from anchor (Xi_err_anchor_vs_fit): {Xi_err_anchor_vs_fit:.6e}")
            print(f"6. In feasible space: {in_feasible}")
            print(f"\n{'*' * 80}")
            print(f"Results - {method_proj_key} (Projected) (num_input_points={num_input_points}, seed={current_seed})")
            print(f"{'*' * 80}")
            print(f"1. Anchor radius: {radius ** 0.5:.6e}")
            print(f"2. Xi_size: {xi_size:.6e}")
            print(f"3. True value range: min={true_min:.6f}, max={true_max:.6f}, anchor_constant={anchor_constant:.6f}")
            print(f"4. {method_proj_key} error from true (Xi_err): {Xi_err_proj:.6e}")
            print(f"5. {method_proj_key} error from anchor (Xi_err_anchor_vs_fit): {Xi_err_anchor_vs_proj:.6e}")
            print(f"6. In feasible space: {in_feasible_proj}")
            print(f"\n{'*' * 80}")
            print(f"Comparison (num_input_points={num_input_points}, seed={current_seed}):")
            print(f"{'*' * 80}")
            print(f"Regular {method_key} error from true: {Xi_err:.6e}")
            print(f"Projected {method_proj_key} error from true: {Xi_err_proj:.6e}")
            print(f"Error improvement: {Xi_err - Xi_err_proj:.6e} ({((Xi_err - Xi_err_proj) / Xi_err * 100):.2f}% reduction)")
            print(f"\nError to anchor function:")
            print(f"  Anchor function error to true function: {Xi_err_anchor_vs_true:.6e}")
            print(f"  Regular {method_key} error from anchor: {Xi_err_anchor_vs_fit:.6e}")
            print(f"  Projected {method_proj_key} error from anchor: {Xi_err_anchor_vs_proj:.6e}")
            print(f"  Anchor radius (feasible space bound): r={radius ** 0.5:.6e}")
            print(f"  Regular {method_key} within feasible space: {in_feasible} (error {'<=' if in_feasible else '>'} radius)")
            print(f"  Projected {method_proj_key} within feasible space: {in_feasible_proj} (error {'<=' if in_feasible_proj else '>'} radius)")
            print(f"\nBounds:")
            print(f"Upper bound (distance between prediction and projection): {upper_bound:.6e}")
            print(f"Lower bound: {lower_bound:.6e}")
            print(f"{'*' * 80}")

        ls_error_mean = np.mean(per_seed_ls_errors)
        ls_error_std = np.std(per_seed_ls_errors)
        projected_error_mean = np.mean(per_seed_projected_errors)
        projected_error_std = np.std(per_seed_projected_errors)
        lower_bound_mean = np.mean(per_seed_lower_bounds)
        upper_bound_mean = np.mean(per_seed_upper_bounds)
        error_improvement_mean = np.mean(per_seed_error_improvements)

        # Clear summary print
        print(f"\n{'=' * 80}")
        print(f"SUMMARY (num_input_points={num_input_points}, averaged over {num_seeds} seeds):")
        print(f"{'=' * 80}")
        print(f"{method.upper()} error:        {ls_error_mean:.6e} ± {ls_error_std:.6e}")
        print(f"Projection error:   {projected_error_mean:.6e} ± {projected_error_std:.6e}")
        print(f"Lower bound mean:   {lower_bound_mean:.6e}")
        print(f"Error improvement:  {error_improvement_mean:.6e}")
        print(f"Upper bound mean:   {upper_bound_mean:.6e}")
        print(f"{'=' * 80}\n")

        # Store mean/std errors for plotting
        input_points_results.append(num_input_points)
        ls_errors.append(ls_error_mean)
        projected_errors.append(projected_error_mean)
        ls_errors_std.append(ls_error_std)
        projected_errors_std.append(projected_error_std)
        lower_bounds.append(lower_bound_mean)
        upper_bounds.append(upper_bound_mean)
        error_improvements.append(error_improvement_mean)
    
    # Create comparison plot
    print("\n" + "=" * 80)
    print("Creating comparison plot...")
    print("=" * 80)
    
    tick_size = 25
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    plt.rc('axes', labelsize=tick_size, titlesize=tick_size)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(input_points_results, ls_errors, 'o-', label=f'{method.upper()}', linewidth=2, markersize=8)
    ax.plot(input_points_results, projected_errors, 's-', label='Projected', linewidth=2, markersize=8)
    ax.fill_between(
        input_points_results,
        np.array(ls_errors) - np.array(ls_errors_std),
        np.array(ls_errors) + np.array(ls_errors_std),
        alpha=0.2
    )
    ax.fill_between(
        input_points_results,
        np.array(projected_errors) - np.array(projected_errors_std),
        np.array(projected_errors) + np.array(projected_errors_std),
        alpha=0.2
    )
    
    ax.set_xlabel('Number of Input Points', fontsize=tick_size)
    ax.set_ylabel('Error from True Function', fontsize=tick_size)
    # ax.set_title(f'Error Comparison: {method.upper()} vs Projected (Increasing Input Points)', fontsize=tick_size)
    ax.legend(fontsize=tick_size)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    
    # Save the comparison plot
    import os
    os.makedirs('./results/spherical_harmonics', exist_ok=True)
    plt.savefig(f'./results/spherical_harmonics/{method.lower()}_error_comparison_vs_input_points.png', dpi=300)
    print("Saved comparison plot to: ./results/spherical_harmonics/error_comparison_vs_input_points.png")
    plt.close()
    
    # Final summary print
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY (All Runs):")
    print(f"{'=' * 80}")
    for i, num_points in enumerate(input_points_results):
        print(f"\nnum_input_points={num_points}:")
        print(f"  {method.upper()} error:        {ls_errors[i]:.6e} ± {ls_errors_std[i]:.6e}")
        print(f"  Projection error:   {projected_errors[i]:.6e} ± {projected_errors_std[i]:.6e}")
        print(f"  Lower bound:        {lower_bounds[i]:.6e}")
        print(f"  Error improvement:  {error_improvements[i]:.6e}")
        print(f"  Upper bound:        {upper_bounds[i]:.6e}")
    print(f"{'=' * 80}\n")


if "__main__" == __name__:
    main()
