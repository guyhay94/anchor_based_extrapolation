import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from find_simple_baselines import create_func_from_basis_and_coefficients
from cheby_utils import fit_regular_ls, create_chebyshev_basis_functions_numpy as create_chebyshev_basis_functions_numpy_normalized
from utils import convert_basis_function_orthogonality_domain, \
    calc_extrapolation_condition_number, add_gaussian_noise, \
    calc_l2_approximation_error, calc_inner_extrapolation_condition_number, select_elements_from_element_list, \
    fit_basis_functions

from typing import List, Callable, Tuple, Dict, Any
from typing_extensions import Literal

AnchorFuncOutput = Dict[str, Any]


def create_anchor_function(x_train: np.ndarray,
                           y_train: np.ndarray,
                           basis_functions: List[Callable],
                           domain_omega: Tuple[float, float] = (-1, 0.5),
                           domain_xi: Tuple[float, float] = (0.5, 1),
                           required_basis_indices: List[int] = None,
                           degree_of_approximation: int = None,
                           max_degree_of_approximation: int = None,
                           min_degree_of_approximation: int = None,
                           approximation_method: Literal["least_squares", "ridge", "lasso"] = "least_squares",
                           approximation_method_alpha: float = None,
                           basis_functions_for_approximation: List[Callable] = None,
                           upper_bound_method: Literal['condition_number', 'inner', 'best'] = 'best'):
    basis_functions_for_approximation = basis_functions if basis_functions_for_approximation is None else basis_functions_for_approximation
    selected_basis_functions, selected_basis_functions_degrees = select_elements_from_element_list(
        elements=basis_functions_for_approximation,
        must_include_indices=required_basis_indices,
        num_elements=degree_of_approximation,
        min_elements=min_degree_of_approximation,
        max_elements=max_degree_of_approximation,
        seed=None)

    approximation_coefs = fit_basis_functions(basis_functions=selected_basis_functions,
                                              x_train=x_train,
                                              y_train=y_train,
                                              method=approximation_method,
                                              alpha=approximation_method_alpha)

    anchor_function = create_func_from_basis_and_coefficients(selected_basis_functions, approximation_coefs)
    omega_error_approx = calc_function_error(approx_function=anchor_function,
                                             x=x_train, y=y_train)

    upper_bound_approx = calc_anchor_function_upper_bound(anchor_function_basis=basis_functions,
                                                          omega_error=omega_error_approx,
                                                          domain_omega=domain_omega,
                                                          domain_xi=domain_xi,
                                                          method=upper_bound_method)

    return anchor_function, upper_bound_approx, omega_error_approx, selected_basis_functions_degrees, approximation_coefs


def create_multiple_anchor_functions(
        x_train: np.ndarray,
        y_train: np.ndarray,
        basis_functions: List[Callable],
        num_to_create: int,
        num_to_select: int,
        degree_of_approximation: int = None,
        max_degree_of_approximation: int = None,
        min_degree_of_approximation: int = None,
        required_basis_indices: List[int] = None,
        max_retries: int = 20,
        domain_omega: Tuple[float, float] = (-1, 0.5),
        domain_xi: Tuple[float, float] = (0.5, 1),
        approximation_method: Literal["least_squares", "ridge", "lasso"] = "least_squares",
        approximation_method_alpha: float = None
) -> List[AnchorFuncOutput]:
    """
    Wrapper function to create multiple anchor functions and select the best ones by omega_error.

    Parameters:
    x_train (List[float]): Training feature data.
    y_train (List[float]): Training target data.
    basis_functions (List): List of basis functions.
    degree_of_anchor_functions_in_omega (int): Number of omega basis elements to select.
    required_basis_indices (List[int]): List of required basis indices.
    num_to_create (int): Total number of anchor functions to create.
    num_to_select (int): Number of best anchor functions to select.

    Returns:
    List[Tuple]: List of tuples containing the best anchor functions, sorted by omega_error or upper_bound.
    Each tuple contains, anchor function, upper_bound, omega_error, which basis where chosen.
    """

    # Set to track unique anchor function configurations
    unique_anchor_functions = set()
    # List to store anchor functions along with their omega_error and other details
    anchor_functions = []
    retries = 0
    while len(anchor_functions) < num_to_create and retries < max_retries:
        # Attempt to create a new anchor function
        anchor_function, upper_bound, omega_error, basis_chosen, basis_coefs = create_anchor_function(
            x_train=x_train,
            y_train=y_train,
            basis_functions=basis_functions,
            domain_omega=domain_omega,
            domain_xi=domain_xi,
            required_basis_indices=required_basis_indices,
            degree_of_approximation=degree_of_approximation,
            max_degree_of_approximation=max_degree_of_approximation,
            min_degree_of_approximation=min_degree_of_approximation,
            approximation_method=approximation_method,
            approximation_method_alpha=approximation_method_alpha,
            basis_functions_for_approximation=None,
            upper_bound_method='best'
        )
        print(f"basis_chosen: {basis_chosen}")
        print(f"omega_error: {omega_error}")
        print(f"upper_bound: {upper_bound}")
        print(f"basis_coefs: {basis_coefs}")
        print("*" * 80)

        # Use basis_chosen as a unique identifier to check for duplicates
        basis_tuple = tuple(sorted(basis_chosen))
        if basis_tuple not in unique_anchor_functions:
            unique_anchor_functions.add(basis_tuple)
            anchor_functions.append({"func": anchor_function,
                                     "upper_bound": upper_bound,
                                     "omega_error": omega_error,
                                     "basis_chosen": basis_chosen,
                                     "basis_coefs": basis_coefs})
            retries = 0  # Reset retries for each new unique anchor function
            print(len(unique_anchor_functions))
        else:
            retries += 1

    if len(anchor_functions) < num_to_create:
        print(
            f"Warning: Only {len(anchor_functions)} unique anchor functions were created after {max_retries} retries.")

    anchor_functions_sorted = sorted(anchor_functions, key=lambda x: x["upper_bound"])
    best_anchor_functions = anchor_functions_sorted[:num_to_select]

    return best_anchor_functions


def prune_out_bad_anchor_functions(anchor_function_output: List[AnchorFuncOutput], x_test: np.ndarray,
                                   upper_bound_key: str = "upper_bound") -> List[
    AnchorFuncOutput]:
    good_anchors = [anchor_function_output[0]]
    for out in anchor_function_output[1:]:
        if check_intersection_exists(good_anchors + [out], x_test, upper_bound_key=upper_bound_key):
            good_anchors.append(out)

    print(f"Anchors left: {len(good_anchors)} / {len(anchor_function_output)}")
    return good_anchors


def check_intersection_exists(
        anchor_functions: List[AnchorFuncOutput],
        x_test: np.ndarray,
        upper_bound_key: str = "upper_bound"
) -> bool:
    """
    Checks if there exists an intersection in the search spaces defined by the anchor functions
    and their upper bound MSE distances, evaluated on x_test.

    Parameters:
    anchor_functions (List[Tuple[Callable[[np.ndarray], np.ndarray], float, float, List[int]]]):
        List of tuples with each anchor function's output, including:
        - anchor_function: The anchor function itself.
        - upper_bound (float): upper bound defining the search space radius.
        - omega_error (float): Error associated with the anchor function.
        - basis_chosen (List[int]): List of basis indices chosen for the anchor function.
    x_test (np.ndarray): Array of test data points used to evaluate the anchor functions.

    Returns:
    bool: True if an intersection exists in all search spaces, False otherwise.
    """

    # Evaluate each anchor function on x_test
    anchor_outputs = [anchor_function["func"](x_test) for anchor_function in anchor_functions]
    upper_bounds = [anchor_function[upper_bound_key] for anchor_function in anchor_functions]

    # Check that the distance between each pair of anchor outputs is within the sum of their upper bounds
    for i, (anchor_i_output, upper_bound_i) in enumerate(zip(anchor_outputs, upper_bounds)):
        for j, (anchor_j_output, upper_bound_j) in enumerate(zip(anchor_outputs, upper_bounds)):
            if i != j:
                # Calculate the distance between anchor i and anchor j on x_test
                l2_distance = calc_l2_approximation_error(anchor_i_output, anchor_j_output, x_test)
                # Check if distance exceeds the combined upper bounds
                if l2_distance > (upper_bound_i + upper_bound_j):
                    return False

    return True


def check_if_function_in_full_search_space(func: Callable,
                                           x_test: np.ndarray,
                                           anchor_functions_output: List[AnchorFuncOutput],
                                           epsilon: float = 0.00001) -> bool:
    candidate_values = func(x_test)
    for anchor_func, upper_bound, _, basis_indices in anchor_functions_output:
        anchor_values = anchor_func(x_test)
        if calc_l2_approximation_error(candidate_values, anchor_values, x_test) > upper_bound + epsilon:
            return False
    return True


def find_function_in_search_space_objective(coefs: np.ndarray, x: np.ndarray,
                                            anchor_functions_output: List[AnchorFuncOutput],
                                            basis_functions: List[Callable]) -> float:
    """Objective function to minimize."""
    # Optional: Sum of squared distances to each anchor function, or use a custom metric
    total_distance = 0
    candidate_function = create_func_from_basis_and_coefficients(basis_functions, coefs)
    candidate_values = candidate_function(x)
    for func, upper_bound, _, basis_indices in anchor_functions_output:
        anchor_values = func(x)
        total_distance += calc_l2_approximation_error(candidate_values, anchor_values, x) / upper_bound
    return total_distance


def modified_objective(coefs: np.ndarray, x: np.ndarray, anchor_functions: List[AnchorFuncOutput],
                       basis_functions: List[Callable], penalty: float = 10.0) -> float:
    """Objective function with a penalty for exceeding distance constraints.
    Barrier method
    """
    total_distance = find_function_in_search_space_objective(coefs, x, anchor_functions, basis_functions)
    penalty_term = 0
    for func, upper_bound, _, basis_indices in anchor_functions:
        candidate_values = create_func_from_basis_and_coefficients(basis_functions, coefs)(x)
        anchor_values = func(x)
        constraint_violation = calc_l2_approximation_error(candidate_values, anchor_values, x) - upper_bound
        if constraint_violation > 0:
            penalty_term += penalty * constraint_violation ** 2
    return total_distance + penalty_term


def find_function_in_full_search_space(
        x: np.ndarray,
        anchor_functions: List[AnchorFuncOutput],
        basis_functions: List[Callable],
        initial_guess: np.ndarray = None
) -> np.ndarray:
    """Find coefficients for a function within the allowed distances of each anchor."""
    # Generate a more informed initial guess
    initial_guess = np.zeros(len(basis_functions)) if initial_guess is None else initial_guess

    # Constraints based on upper bounds for each anchor function
    constraints = []
    for func, upper_bound, _, _ in anchor_functions:
        constraints.append({
            'type': 'ineq',
            'fun': lambda coefs, function=func, ub=upper_bound:
            ub - calc_l2_approximation_error(create_func_from_basis_and_coefficients(basis_functions, coefs)(x),
                                             func(x), x)
        })

    # Solve with SLSQP method and modified objective function
    result = minimize(
        modified_objective,
        initial_guess,
        args=(x, anchor_functions, basis_functions),
        constraints=constraints,
        method='SLSQP'
    )
    if not result.success:
        print(result.message)

    return result.x


def calc_anchor_function_upper_bound(anchor_function_basis: List[Callable],
                                     omega_error: float,
                                     domain_xi: Tuple[float, float],
                                     domain_omega: Tuple[float, float] = None,
                                     method: Literal['condition_number', 'inner', 'best'] = 'condition_number'
                                     ):
    condition_number_upper_bound = np.iinfo(int).max
    inner_upper_bound = np.iinfo(int).max
    if method in ['condition_number', 'best']:
        anchor_function_basis_omega = convert_basis_function_orthogonality_domain(basis_functions=anchor_function_basis,
                                                                                  basis_orthogonality_a=domain_omega[0],
                                                                                  basis_orthogonality_b=domain_xi[1],
                                                                                  a_1=domain_omega[0],
                                                                                  b_1=domain_omega[1])

        condition_number_upper_bound = calc_extrapolation_condition_number(phi_list=anchor_function_basis_omega,
                                                                           domain_omega=domain_omega,
                                                                           domain_xi=domain_xi)
    if method in ['inner', 'best']:
        inner_upper_bound = calc_inner_extrapolation_condition_number(phi_list=anchor_function_basis,
                                                                      domain_xi=domain_xi)
        inner_upper_bound = np.iinfo(int).max if inner_upper_bound is None else inner_upper_bound

    if method == 'condition_number':
        kappa = condition_number_upper_bound
    elif method == "inner":
        kappa = inner_upper_bound
    elif method == "best":
        print(f"condition_number_upper_bound: {condition_number_upper_bound}")
        print(f"inner_upper_bound: {inner_upper_bound}")
        kappa = min(condition_number_upper_bound, inner_upper_bound)
    else:
        raise NotImplementedError(f"Method: {method} is not implemented")

    return omega_error * kappa


def calc_function_error(approx_function: Callable,
                        x: np.ndarray, y: np.ndarray) -> float:
    return calc_l2_approximation_error(y, approx_function(x), x)


def test_one_omega_error_approximation(func_to_extrapolate: Callable,
                                       x_train: np.ndarray, y_train: np.ndarray,
                                       approximation_deg: int = 7,
                                       print_log: bool = False) -> Tuple[float, float]:
    basis_functions = create_chebyshev_basis_functions_numpy_normalized(approximation_deg)
    ls_coefs = fit_regular_ls(x_train, y_train, approximation_deg)
    ls_func = create_func_from_basis_and_coefficients(basis_functions=basis_functions, coefs=ls_coefs)
    omega_approximation = calc_function_error(ls_func, x_train, y_train)
    omega_true = calc_function_error(ls_func, x_train, func_to_extrapolate(x_train))
    if print_log:
        print(f"True Omega error: {omega_true}")
        print(f"App. Omega error: {omega_approximation}")

    return omega_true, omega_approximation


def test_omega_error_approximation(func_to_extrapolate: Callable,
                                   a_omega: float = -1, b_omega: float = 0.9,
                                   num_test_runs: int = 100,
                                   noise_strength: float = None,
                                   approximation_deg: int = 7,
                                   print_log: bool = False):
    omega_true_errors = []
    omega_approximation_errors = []
    for _ in range(num_test_runs):
        x_train = np.linspace(a_omega, b_omega, 100)
        add_noise = True if noise_strength is not None else False
        y_train = (add_gaussian_noise(func_to_extrapolate(x_train), SNR_dB=noise_strength)
                   if add_noise else func_to_extrapolate(x_train))

        omega_true_error, omega_approximation_error = test_one_omega_error_approximation(
            func_to_extrapolate=func_to_extrapolate,
            x_train=x_train, y_train=y_train,
            approximation_deg=approximation_deg,
            print_log=print_log)
        omega_true_errors.append(omega_true_error)
        omega_approximation_errors.append(omega_approximation_error)

    omega_true_errors = np.array(omega_true_errors)
    omega_approximation_errors = np.array(omega_approximation_errors)

    error_of_errors = pd.Series(omega_true_errors - omega_approximation_errors)
    print(error_of_errors.describe(percentiles=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95]))
    error_of_errors.hist(bins=100)
    plt.show()


def test_anchor_function_creation(function_to_extrapolate: Callable,
                                  basis_functions: List[Callable],
                                  domain_omega: Tuple[float, float] = (-1, 0.5),
                                  domain_xi: Tuple[float, float] = (0.5, 1),
                                  noise_strength: float = None,
                                  required_basis_indices: List[int] = None,
                                  degree_of_approximation: int = None,
                                  max_degree_of_approximation: int = None,
                                  min_degree_of_approximation: int = None,
                                  approximation_method: Literal["least_squares", "ridge", "lasso"] = "least_squares",
                                  approximation_method_alpha: float = None,
                                  basis_functions_for_approximation: List[Callable] = None,
                                  upper_bound_method: Literal['condition_number', 'inner', 'best'] = 'best'
                                  ):
    x_train = np.linspace(domain_omega[0], domain_omega[1], 100)
    x_train_true = np.linspace(domain_omega[0], domain_omega[1], 1000)
    x_test = np.linspace(domain_xi[0], domain_xi[1], 1000)

    y_train_true = function_to_extrapolate(x_train_true)
    y_test_true = function_to_extrapolate(x_test)

    add_noise = True if noise_strength is not None else False
    y_train = (add_gaussian_noise(function_to_extrapolate(x_train), SNR_dB=noise_strength)
               if add_noise else function_to_extrapolate(x_train))

    anchor_function, upper_bound_approx, omega_error_approx, selected_basis_functions, coefs = create_anchor_function(
        x_train=x_train,
        y_train=y_train,
        basis_functions=basis_functions,
        domain_omega=domain_omega,
        domain_xi=domain_xi,
        required_basis_indices=required_basis_indices,
        degree_of_approximation=degree_of_approximation,
        max_degree_of_approximation=max_degree_of_approximation,
        min_degree_of_approximation=min_degree_of_approximation,
        approximation_method=approximation_method,
        approximation_method_alpha=approximation_method_alpha,
        basis_functions_for_approximation=basis_functions_for_approximation,
        upper_bound_method=upper_bound_method
    )

    y_train_pred = anchor_function(x_train_true)
    y_test_pred = anchor_function(x_test)

    omega_error = calc_l2_approximation_error(y_train_true, y_train_pred, x_train_true)
    xi_error = calc_l2_approximation_error(y_test_true, y_test_pred, x_test)

    return xi_error, upper_bound_approx, omega_error, omega_error_approx, selected_basis_functions, coefs


def test_anchor_function_creation_wrapper(function_to_extrapolate: Callable,
                                          basis_functions: List[Callable],
                                          domain_omega: Tuple[float, float] = (-1, 0.5),
                                          domain_xi: Tuple[float, float] = (0.5, 1),
                                          noise_strength: float = None,
                                          required_basis_indices: List[int] = None,
                                          degree_of_approximation: int = None,
                                          max_degree_of_approximation: int = None,
                                          min_degree_of_approximation: int = None,
                                          approximation_method: Literal[
                                              "least_squares", "ridge", "lasso"] = "least_squares",
                                          approximation_method_alpha: float = None,
                                          basis_functions_for_approximation: List[Callable] = None,
                                          upper_bound_method: Literal['condition_number', 'inner', 'best'] = 'best',
                                          num_tests: int = 100,
                                          print_each_test: bool = False):
    upper_bound_breached = []
    for _ in range(num_tests):
        xi_error, upper_bound_approx, omega_error, omega_error_approx, selected_basis_functions, coefs = test_anchor_function_creation(
            function_to_extrapolate=function_to_extrapolate,
            basis_functions=basis_functions,
            domain_omega=domain_omega,
            domain_xi=domain_xi,
            noise_strength=noise_strength,
            required_basis_indices=required_basis_indices,
            degree_of_approximation=degree_of_approximation,
            max_degree_of_approximation=max_degree_of_approximation,
            min_degree_of_approximation=min_degree_of_approximation,
            approximation_method=approximation_method,
            approximation_method_alpha=approximation_method_alpha,
            basis_functions_for_approximation=basis_functions_for_approximation,
            upper_bound_method=upper_bound_method,
        )
        upper_bound_breached.append(xi_error > upper_bound_approx)
        if print_each_test:
            print("*" * 80)
            print(f"\t xi_error: {xi_error}")
            print(f"\t upper_bound_approx: {upper_bound_approx}")
            print(f"\t omega_error: {omega_error}")
            print(f"\t omega_error_approx: {omega_error_approx}")
            print(f"\t selected_basis_functions: {selected_basis_functions}")
            print(f"\t coefs: {coefs}")
            if xi_error > upper_bound_approx:
                print("bad")
                print("*" * 80)
                print("*" * 80)
                print("*" * 80)

    print(f"Number of times upper bound failed: {np.sum(upper_bound_breached)}")

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def compare_condition_numbers(
    b_omega_values: List[float],
    basis_sizes: List[int],
    a_omega: float,
    b_xi: float
) -> List[Tuple[int, float, float, float]]:
    """
    Compares condition_number_upper_bound and inner_upper_bound for varying b_omega
    and number of basis functions.

    Args:
        b_omega_values (List[float]): A list of b_omega values to test.
        basis_sizes (List[int]): A list of numbers of basis functions to test.
        a_omega (float): Lower bound of the omega domain.
        b_xi (float): Upper bound of the xi domain.

    Returns:
        List[Tuple[int, float, float, float]]: A list of results, where each tuple contains:
            - Number of basis functions
            - b_omega value
            - condition_number_upper_bound
            - inner_upper_bound
    """
    results = []

    for num_basis in basis_sizes:
        # Create the Chebyshev basis for the current number of basis functions
        anchor_function_basis = create_chebyshev_basis_functions_numpy_normalized(num_basis)

        for b_omega in b_omega_values:
            a_xi = b_omega  # Set a_xi to be equal to b_omega

            # Convert the basis function orthogonality domain
            anchor_function_basis_omega = convert_basis_function_orthogonality_domain(
                basis_functions=anchor_function_basis,
                basis_orthogonality_a=a_omega,
                basis_orthogonality_b=b_xi,
                a_1=a_omega,
                b_1=b_omega
            )

            # Calculate the condition numbers
            condition_number_upper_bound = calc_extrapolation_condition_number(
                phi_list=anchor_function_basis_omega,
                domain_omega=(a_omega, b_omega),
                domain_xi=(a_xi, b_xi)
            )

            inner_upper_bound = calc_inner_extrapolation_condition_number(
                phi_list=anchor_function_basis,
                domain_xi=(a_xi, b_xi)
            )

            if inner_upper_bound is not None:
                # Append the results
                results.append((num_basis, b_omega, condition_number_upper_bound, inner_upper_bound))

    return results


def plot_condition_number_comparison(results: List[Tuple[int, float, float, float]]):
    """
    Plots the comparison between condition_number_upper_bound and inner_upper_bound for each basis size.

    Args:
        results (List[Tuple[int, float, float, float]]): Results from compare_condition_numbers.
    """
    # Separate the results by basis size
    unique_basis_sizes = sorted(set(r[0] for r in results))

    for basis_size in unique_basis_sizes:
        # Create a new figure for each basis size
        plt.figure(figsize=(14, 8))  # Adjusted figure size

        # Filter results for the current basis size
        filtered_results = [r for r in results if r[0] == basis_size]
        b_omega_values = [r[1] for r in filtered_results]
        condition_upper = [r[2] for r in filtered_results]
        inner_upper = [r[3] for r in filtered_results]

        # Plot condition_number_upper_bound with increased line width
        plt.plot(b_omega_values, condition_upper, label="Cond. Upper", linestyle='-', marker='o', linewidth=5)

        # Plot inner_upper_bound with increased line width
        plt.plot(b_omega_values, inner_upper, label="Inner Upper", linestyle='--', marker='x', linewidth=5)

        # Adjust the labels and title font size
        plt.xlabel("b", fontsize=14)  # Increased font size
        plt.ylabel("Condition Number", fontsize=14)  # Added y-axis label with increased font size
        # plt.title(f"Condition Number Comparison (Basis Size: {basis_size})", fontsize=16)  # Title font size
        plt.legend(fontsize=30)  # Adjust legend font size
        plt.grid(True)

        # Save the plot with higher DPI for better quality
        plt.savefig(fr"C:\git\mainfold_approximation_v2\results\compare_bounds_{basis_size}.png", dpi=300)



if __name__ == "__main__":
    number_of_basis_functions = 10
    snr = 35
    np.random.seed(42)  # For reproducibility
    current_function_to_extrapolate = lambda x: 0.2 * basis[0](x) + 0.4 * basis[1](x) + 0.6 * basis[2](x) + 0.3 * basis[
        4](x)
    basis = create_chebyshev_basis_functions_numpy_normalized(number_of_basis_functions)
    # current_function_to_extrapolate = lambda x: basis[0](x) + basis[1](x) + basis[4](x) + basis[6](x)
    # test_omega_error_approximation(func_to_extrapolate=current_function_to_extrapolate,
    #                                a_omega=-1, b_omega=0.5,
    #                                num_test_runs=100,
    #                                noise_strength=35,
    #                                print_log=False)

    # current_b_omega = 0.0
    # test_anchor_function_creation_wrapper(
    #     function_to_extrapolate=current_function_to_extrapolate,
    #     basis_functions=basis,
    #     domain_omega=(-1, current_b_omega),
    #     domain_xi=(current_b_omega, 1),
    #     noise_strength=25,
    #     required_basis_indices=None,
    #     degree_of_approximation=None,
    #     max_degree_of_approximation=None,
    #     min_degree_of_approximation=None,
    #     approximation_method="lasso",
    #     approximation_method_alpha=0.001,
    #     basis_functions_for_approximation=None,
    #     upper_bound_method='best',
    #     num_tests=100,
    #     print_each_test=True)

    # Example Usage
    b_omega_values = np.linspace(0.8, 1, 30).tolist()  # Test b_omega values from -1 to 1
    basis_sizes = [5, 10, 15]  # Test with 5, 10, and 15 basis functions
    a_omega = -1
    b_xi = 1

    comparison_results = compare_condition_numbers(
        b_omega_values=b_omega_values,
        basis_sizes=basis_sizes,
        a_omega=a_omega,
        b_xi=b_xi
    )

    # Plot results
    plot_condition_number_comparison(comparison_results)





