import random
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import scipy.stats as stats
from scipy.stats import chi2
from typing import List, Callable, Union, Tuple, Any, Optional
from typing_extensions import Literal
from find_simple_baselines import find_best_coefs_for_specific_basis_functions, create_func_from_basis_and_coefficients


def inner_product_y_samples_and_function(sample_x, sample_y, basis_function):
    basis_values = basis_function(sample_x)
    inner_prod = simpson(sample_y * basis_values, sample_x)
    return inner_prod


def inner_product_between_functions(func_a: Callable,
                                    func_b: Callable,
                                    start_point: float,
                                    end_point: float,
                                    num_samples: int = 1000) -> float:
    x_samples = np.linspace(start_point, end_point, num_samples)
    y_samples_a = func_a(x_samples)
    y_samples_b = func_b(x_samples)
    return simpson(y_samples_a * y_samples_b, x_samples)


def inner_product_matrix(num_terms: int, a: float, b: float, num_samples: int = 1000) -> np.ndarray:
    """
    Compute the matrix of inner products of the first num_terms cosine and sine basis functions over [a, b].

    Parameters:
    - num_terms: Number of cosine and sine terms to consider
    - a: Start of the interval
    - b: End of the interval
    - num_samples: integration points

    Returns:
    - A (2*num_terms) x (2*num_terms) matrix of inner products
    """
    # Define sample points for numerical integration
    x_samples = np.linspace(a, b, num_samples)

    # Initialize the matrix to store the inner products
    matrix_size = 2 * num_terms
    inner_product_mat = np.zeros((matrix_size, matrix_size))

    # Compute inner products between all pairs of basis functions
    for i in range(num_terms):
        for j in range(num_terms):
            # Cosine-Cosine inner product
            cos_i = np.cos((i + 1) * x_samples)
            cos_j = np.cos((j + 1) * x_samples)
            inner_product_mat[2 * i, 2 * j] = simpson(cos_i * cos_j, x_samples)

            # Cosine-Sine inner product
            sin_j = np.sin((j + 1) * x_samples)
            inner_product_mat[2 * i, 2 * j + 1] = simpson(cos_i * sin_j, x_samples)

            # Sine-Cosine inner product
            inner_product_mat[2 * i + 1, 2 * j] = simpson(sin_j * cos_i, x_samples)

            # Sine-Sine inner product
            sin_i = np.sin((i + 1) * x_samples)
            inner_product_mat[2 * i + 1, 2 * j + 1] = simpson(sin_i * sin_j, x_samples)

    return inner_product_mat


def coef_approximation_draw(basis_functions: List[Callable],
                            coefficients: np.ndarray,
                            x: np.ndarray,
                            y: np.ndarray,
                            plot: bool = False):
    pred_func = create_func_from_basis_and_coefficients(basis_functions, coefficients)
    pred_y = pred_func(x)
    rmse = calculate_rmse(y, pred_y)

    if plot:
        plt.plot(x, y, color='black', label='The Function', linewidth=2.0)
        plt.scatter(x, pred_y, color='orange', label='Training LS')
        plt.rc('legend', frameon=True, fontsize=13)
        plt.legend()
        plt.show()

    return rmse


def rmse_between_functions(
        func1: Callable[[np.ndarray], np.ndarray],
        func2: Callable[[np.ndarray], np.ndarray],
        start: float,
        end: float,
        num_samples: int
) -> float:
    """
    Calculate the RMSE between two functions over a specified domain.

    Parameters:
    - func1: The first function to compare.
    - func2: The second function to compare.
    - start: The starting point of the domain.
    - end: The ending point of the domain.
    - num_samples: The number of samples in the domain.

    Returns:
    - rmse: The Root Mean Square Error between the two functions over the specified domain.
    """
    # Generate sample points in the domain
    x_values = np.linspace(start, end, num_samples)

    # Evaluate both functions at the sample points
    y_values_func1 = func1(x_values)
    y_values_func2 = func2(x_values)

    return calculate_rmse(y_values_func1, y_values_func2)


def calculate_rmse(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two numpy arrays.

    Parameters:
    array1 (np.ndarray): The first input array.
    array2 (np.ndarray): The second input array.

    Returns:
    float: The RMSE between the two arrays.
    """
    # Ensure both arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError("The input arrays must have the same shape.")

    # Calculate the mean squared error (MSE)
    mse = np.mean((array1 - array2) ** 2)

    # Calculate the root mean square error (RMSE)
    rmse = np.sqrt(mse)

    return rmse


def add_gaussian_noise(y: np.ndarray, SNR_dB: float, seed: int = None) -> np.ndarray:
    """
    Add Gaussian noise to a signal based on a given SNR in decibels.

    Args:
        y (np.ndarray): Original signal.
        SNR_dB (float): Desired signal-to-noise ratio in decibels.
        seed (int, optional): Seed for random number generator. Defaults to None.

    Returns:
        np.ndarray: Noisy signal.
    """
    if seed is not None:
        np.random.seed(seed)

    signal_power = np.mean(y ** 2)
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise = np.random.normal(0, np.sqrt(noise_power), y.shape)
    return y + noise


def find_basis_coefficients_using_inner_product(x: np.ndarray,
                                                y: np.ndarray,
                                                basis_functions: List[Callable],
                                                norm: Union[float, List[float]] = 1) -> List[float]:
    if isinstance(norm, list):
        return [inner_product_y_samples_and_function(x, y, basis_functions[i]) / norm[i] for i in
                range(len(basis_functions))]
    return [inner_product_y_samples_and_function(x, y, f) / norm for f in basis_functions]


def get_norm_of_basis_functions(basis_functions: List[Callable],
                                start_point: float,
                                end_point: float) -> List[float]:
    return [inner_product_between_functions(f, f, start_point, end_point) for f in basis_functions]


def get_domain_transformer(a_1: float, b_1: float, a_2: float, b_2: float) -> Callable:
    """
    [a_1, b_1] -> [a_2, b_2]
    Args:
        a_1:
        b_1:
        a_2:
        b_2:

    Returns:

    """
    c_1 = (a_1 - b_1) / (a_2 - b_2)
    c_2 = a_2 - a_1 * (1 / c_1)
    return lambda x: x / c_1 + c_2


def get_basis_to_basis_mapping(mapper_basis_functions: List[Callable],
                               basis_to_map_functions: List[Callable],
                               a: float, b: float) -> np.ndarray:
    """
    Fits every function from basis_to_map_functions with mapper_basis_functions.
    Returns a matrix M, that each row are the coefficients for mapper_basis_functions to the basis_to_map_functions
    meaning:
    basis_to_map_functions[i] = M[i,0] mapper_basis_functions[0] + ... + M[i,n] mapper_basis_functions[n]
    Args:
        mapper_basis_functions:
        basis_to_map_functions:
        a:
        b:

    Returns:

    """

    x = np.linspace(a, b, 1000)
    matrix_mapping = np.zeros((len(basis_to_map_functions), len(mapper_basis_functions)))
    for i, func in enumerate(basis_to_map_functions):
        y_original = func(x)
        coefficients = find_best_coefs_for_specific_basis_functions(x, y_original, mapper_basis_functions)
        rmse = coef_approximation_draw(mapper_basis_functions, coefficients, x, y_original)
        if rmse > 0.001:
            print(f"Problem with mapping {i + 1}: {rmse}")
        matrix_mapping[i, :] = coefficients

    return matrix_mapping


def get_basis_mapping_orthogonal_domain_to_domain(basis_functions: List[Callable],
                                                  basis_orthogonality_a: float, basis_orthogonality_b: float,
                                                  a_1: float, b_1: float,
                                                  ) -> np.ndarray:
    """
    Answeres the question what basis elements in a_1,b_1 domain are needed to approximate 
    the basis_elements from the original orthogonality domain meaning:
    [basis_orthogonality_a,basis_orthogonality_b] -> [a_1,b_1]
    Args:
        basis_functions:
        basis_orthogonality_a:
        basis_orthogonality_b:
        a_1: 
        b_1:

    Returns:
        Matrix where each row number represent the index of the basis element, and the rows values represent the 
        basis coefficients in the new domain
    """
    num_basis_functions = len(basis_functions)
    x = np.linspace(a_1, b_1, 1000)
    x_t = np.linspace(basis_orthogonality_a, basis_orthogonality_b, 1000)
    matrix_mapping = np.zeros((num_basis_functions, num_basis_functions))
    for i, func in enumerate(basis_functions):
        # print(f"Mapping: {i + 1}")
        y_original = func(x)
        coefficients = find_best_coefs_for_specific_basis_functions(x_t, y_original, basis_functions)
        rmse = coef_approximation_draw(basis_functions, coefficients, x_t, y_original)
        if rmse > 0.001:
            print(
                f"Problem with mapping {i + 1}: {rmse}, ({basis_orthogonality_a},{basis_orthogonality_b})->({a_1},{b_1})")
        matrix_mapping[i, :] = coefficients

    return matrix_mapping


def convert_coefficients_using_mapping(coefficients: Union[np.ndarray, List[float]],
                                       basis_mapping) -> np.ndarray:
    num_coefficients = basis_mapping.shape[0]
    aa = (np.eye(num_coefficients) * coefficients) @ basis_mapping
    return np.ones((num_coefficients,)) @ aa


def convert_basis_function_orthogonality_domain(basis_functions: List[Callable],
                                                basis_orthogonality_a: float, basis_orthogonality_b: float,
                                                a_1: float, b_1: float) -> List[Callable]:
    transformer = get_domain_transformer(a_1, b_1,
                                         basis_orthogonality_a, basis_orthogonality_b,
                                         )
    return [lambda x, f=f: f(transformer(x)) for f in basis_functions]


def calc_extrapolation_condition_number(phi_list: List[Callable],
                                        domain_omega: Tuple[float, float],
                                        domain_xi: Tuple[float, float]
                                        ):
    """
    Calculates the Extrapolation Condition Number κ.

    Parameters:
    - phi_list: list of basis functions {φk}, each a real-valued function.
    - domain_Omega: tuple (a_Omega, b_Omega), domain of integration for Ω.
    - domain_Xi: tuple (a_Xi, b_Xi), domain of integration for Ξ.

    Returns:
    - κ: Extrapolation condition number.
    """
    a_xi, b_xi = domain_xi
    a_omega, b_omega = domain_omega

    d = len(phi_list)  # Number of basis functions

    # Compute squared L² norms in domains Ξ and Ω
    norm_xi_vals = get_norm_of_basis_functions(phi_list, a_xi, b_xi)
    norm_omega_vals = get_norm_of_basis_functions(phi_list, a_omega, b_omega)

    # Compute MΞ and mΩ
    M_Xi = np.max(norm_xi_vals)
    m_Omega = np.min(norm_omega_vals)

    # Compute κ
    kappa = (d * M_Xi) / m_Omega

    return kappa


# ---------- Generic Gram (works for any provided φ-list) ----------
def _gram_matrix_on_interval(
        phi_list: List[Callable[[np.ndarray], np.ndarray]],
        a: float,
        b: float,
        n_quad: int = 4097
) -> np.ndarray:
    """
    Compute Gram matrix G_ij = ∫_a^b φ_i(x) φ_j(x) dx via Simpson's rule on a uniform grid.
    """
    if n_quad % 2 == 0:
        n_quad += 1
    x = np.linspace(a, b, n_quad)
    Phi = np.vstack([phi(x) for phi in phi_list])  # shape (d, n_pts)
    d = Phi.shape[0]
    G = np.empty((d, d), float)
    for i in range(d):
        for j in range(i, d):
            val = simpson(Phi[i] * Phi[j], x)
            G[i, j] = val
            G[j, i] = val
    return G


def calc_extrapolation_condition_number_improved(
        phi_list: List[Callable[[np.ndarray], np.ndarray]],
        domain_xi: Tuple[float, float],
        domain_omega: Tuple[float, float] = None,
        normalize_basis: bool = True,
        n_quad: int = 4097
) -> float:
    """
    Improved extrapolation condition number κ:
      1. Normalize the Ω-orthogonal basis using get_norm_of_basis_functions.
      2. Form Π_ij = <ψ_i, ψ_j>_Ξ with ψ_i = φ_i / ||φ_i||_Ω.
      3. Return κ = λ_max(Π).
    """
    a_xi, b_xi = domain_xi

    # 2. Ξ-Gram in original basis, then transform to Ω-orthonormal basis
    G_xi = _gram_matrix_on_interval(phi_list, a_xi, b_xi, n_quad)
    if normalize_basis:
        a_omega, b_omega = domain_omega
        # 1. Ω-norms squared using your existing function
        norms2_omega = get_norm_of_basis_functions(phi_list, a_omega, b_omega)
        D = np.diag(1.0 / np.sqrt(norms2_omega))  # scaling to Ω-orthonormal
        Pi = D @ G_xi @ D
    else:
        Pi = G_xi

    Pi = 0.5 * (Pi + Pi.T)  # symmetrize numerically

    # 3. κ = largest eigenvalue of Π
    kappa = float(np.linalg.eigvalsh(Pi)[-1])
    return kappa


def calc_inner_extrapolation_condition_number(phi_list: List[Callable],
                                              domain_xi: Tuple[float, float]
                                              ) -> Union[float, None]:
    """
        Calculates the Inner Extrapolation Condition Number κ.

        Parameters:
        - phi_list: list of basis functions {φk}, each a real-valued function, orthonormal on domain_all (Omega U Xi)
        - domain_Xi: tuple (a_Xi, b_Xi), domain of integration for Ξ.
        - domain_Omega: tuple (a_all, b_all), if None will be ignored and 1 is assumed.

        Returns:
        - κ: Extrapolation condition number.
        """
    M_Xi = max(get_norm_of_basis_functions(phi_list, domain_xi[0], domain_xi[1]))
    d = len(phi_list)

    # kappa_all = calc_extrapolation_condition_number_improved(
    #     phi_list=phi_list,
    #     domain_xi=domain_xi,
    #     normalize_basis=False
    # )
    #
    # kappa = kappa_all / (1- kappa_all)
    factor = d * M_Xi

    if factor >= 1:
        return None
    return factor / (1 - factor)


def estimate_noise_variance_from_differences(y_noisy: np.ndarray) -> float:
    differences = np.diff(y_noisy)
    return np.var(differences) / 2


def estimate_noise_psd(y_noisy: np.ndarray, fs=1.0) -> float:
    freqs, psd = welch(y_noisy, fs=fs)
    # Assuming white noise dominates the high frequencies, integrate the PSD over this region
    noise_variance_estimate = np.mean(psd[-10:])  # Estimate from the high-frequency tail
    return float(noise_variance_estimate)


def calc_l2_approximation_error(y_true: np.ndarray, y_pred: np.ndarray, x_samples: np.ndarray) -> float:
    diff = y_true - y_pred
    idx = np.argsort(x_samples)
    return simpson((diff * diff)[idx], x_samples[idx])


def select_elements_from_element_list(
        elements: List[Any],
        must_include_indices: Optional[List[Any]] = None,
        num_elements: Optional[int] = None,
        min_elements: Optional[int] = None,
        max_elements: Optional[int] = None,
        seed: Optional[int] = None
) -> Tuple[List[Any], List[int]]:
    """
    Selects elements from a given list with various constraints and returns their indices.

    Parameters:
    ----------
    elements : List[Any]
        The list of elements to select from.
    must_include_indices : Optional[List[int]], default=None
        Elements that must be included in the final selection.
    num_elements : Optional[int], default=None
        The exact number of elements to select.
    min_elements : Optional[int], default=None
        The minimum number of elements to select (used with max_elements).
    max_elements : Optional[int], default=None
        The maximum number of elements to select (used with min_elements).
    seed : Optional[int], default=None
        Seed for random number generation (for reproducibility).

    Returns:
    -------
    Tuple[List[Any], List[int]]
        A tuple containing the selected elements and their indices in the original list.
    """
    if seed is not None:
        random.seed(seed)

    must_include_indices = must_include_indices or []
    must_include = [elements[i] for i in must_include_indices]
    selectable_elements = [e for e in elements if e not in must_include]

    # Determine the number of elements to select
    if num_elements is not None:
        num_to_select = num_elements
    else:
        num_to_select = random.randint(min_elements or 1, max_elements or len(elements))

    # Adjust number to select considering must_include elements
    remaining_slots = max(0, num_to_select - len(must_include))
    selected_elements = random.sample(selectable_elements, remaining_slots) if remaining_slots > 0 else []
    len(selected_elements)

    # Combine selected elements with must_include elements
    final_selection = must_include + selected_elements

    # Find indices of selected elements in the original list
    selected_indices = [elements.index(element) for element in final_selection]

    return final_selection, selected_indices


def create_feature_matrix(basis_functions, x):
    # Map input data through basis functions to create a feature matrix
    feature_matrix = np.column_stack([func(x) for func in basis_functions])
    return feature_matrix


def fit_basis_functions(
        basis_functions: List[Callable[[np.ndarray], np.ndarray]],
        x_train: np.ndarray,
        y_train: np.ndarray,
        method: Literal["least_squares", "ridge", "lasso"] = "least_squares",
        alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Fits coefficients for a set of basis functions using Least Squares, Ridge, or Lasso regression.

    Parameters:
    ----------
    basis_functions : List[Callable[[np.ndarray], np.ndarray]]
        A list of basis functions that map the input to a feature space.
    x_train : np.ndarray
        Training input data of shape (n_samples,).
    y_train : np.ndarray
        Target output data of shape (n_samples,).
    method : str, default="least_squares"
        The regression method to use. Options: "least_squares", "ridge", "lasso".
    alpha : Optional[float], default=None
        Regularization strength (only applicable for Ridge and Lasso). Ignored for Least Squares.

    Returns:
    -------
    np.ndarray
        The coefficients of the basis functions that best fit the data.

    Raises:
    ------
    ValueError:
        If an invalid method is specified or if regularization strength is missing for Ridge or Lasso.
    """

    # Create the design matrix from the basis functions
    X_design = create_feature_matrix(basis_functions, x_train)
    # Choose the regression method
    if method == "least_squares":
        model = LinearRegression(fit_intercept=False)
    elif method == "ridge":
        if alpha is None:
            raise ValueError("Regularization strength 'alpha' must be specified for Ridge regression.")
        model = Ridge(alpha=alpha, fit_intercept=False)
    elif method == "lasso":
        if alpha is None:
            raise ValueError("Regularization strength 'alpha' must be specified for Lasso regression.")
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    else:
        raise ValueError("Invalid method specified. Choose from 'least_squares', 'ridge', or 'lasso'.")

    # Fit the model to the data
    model.fit(X_design, y_train)

    # Return the coefficients of the basis functions
    return model.coef_


def ell_bounds_known_sigma(n: int, sigma: float, abs_Omega: float, confidence: float = 0.95):
    """
    Returns (lower, upper) CI bounds for ℓ = |Ω| σ² when σ is known.

    Parameters
    ----------
    n : int
        Number of i.i.d. Gaussian samples.
    sigma : float
        Known noise standard deviation.
    abs_Omega : float
        Measure of Ω (for discrete sampling this is just n).
    confidence : float
        Desired confidence level (e.g. 0.95 for a 95% CI).
    """
    alpha = 1 - confidence
    q_low = chi2.ppf(alpha / 2, df=n)
    q_high = chi2.ppf(1 - alpha / 2, df=n)

    ell = abs_Omega * (sigma ** 2)
    return ell * (q_low / n), ell * (q_high / n)


if __name__ == "__main__":
    pass
    ell_bounds_known_sigma(
        n=10,
        sigma=0.1,
        abs_Omega=2,
        confidence=0.95
    )
