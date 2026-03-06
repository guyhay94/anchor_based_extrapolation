import tensorflow as tf
import numpy as np
from scipy.integrate import simpson
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Callable, Union
import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import Chebyshev



# Define the target function you want to fit
def target_function(x):
    return np.sin(2 * x) + 2 * np.cos(x) + 0.5 * np.sin(3 * x) + 0.1 * np.cos(4 * x)


def create_fourier_basis_functions_tf(deg: int = 7):
    def create_sin_func(freq: float):
        return lambda x: tf.sin(freq * x)
    def create_cos_func(freq: float):
        return lambda x: tf.cos(freq * x)
    functions = []
    for i in range(0, deg):
        if i == 0:
            func = lambda x: tf.ones_like(x)
        elif i % 2 == 1:
            print(f"sin({((i + 1) // 2)}x)")
            func = create_sin_func(((i + 1) // 2))
        else:
            print(f"cos({((i + 1) // 2)}x)")
            func = create_cos_func(((i + 1) // 2))
        functions.append(func)
    return functions


def create_fourier_basis_functions(deg: int = 7, print_functions_symbols: bool = False):
    def create_sin_func(freq: float):
        return lambda x: np.sin(freq * x)
    def create_cos_func(freq: float):
        return lambda x: np.cos(freq * x)
    functions = []
    for i in range(0, deg):
        if i == 0:
            if print_functions_symbols:
                print("ones func")
            func = lambda x: np.ones_like(x)
        elif i % 2 == 1:
            if print_functions_symbols:
                print(f"sin({((i + 1) // 2)}x)")
            func = create_sin_func(((i + 1) // 2))
        else:
            if print_functions_symbols:
                print(f"cos({((i + 1) // 2)}x)")
            func = create_cos_func(((i + 1) // 2))
        functions.append(func)
    return functions


CHEBYSHEV_EXPLICT = [
        lambda x: tf.ones_like(x),
        lambda x: x,
        # lambda x: 2 * x ** 2 - 1,
        # lambda x: 4 * x ** 3 - 3 * x,
        # lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
        # lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        # lambda x: 32 * x ** 6 - 48 * x ** 4 + 18 * x ** 2 - 1,
        # lambda x: 64 * x ** 7 - 112 * x ** 5 + 56 * x ** 3 - 7 * x,
        # lambda x: 128 * x ** 8 - 256 * x ** 6 + 160 * x ** 4 - 32 * x ** 2 + 1,
        # lambda x: 256 * x ** 9 - 576 * x ** 7 + 432 * x ** 5 - 120 * x ** 3 + 9 * x,
        # lambda x: 512 * x ** 10 - 1280 * x ** 8 + 1120 * x ** 6 - 400 * x ** 4 + 50 * x ** 2 - 1,
        # lambda x: 1024 * x ** 11 - 2816 * x ** 9 + 2816 * x ** 7 - 1232 * x ** 5 + 220 * x ** 3 - 11 * x,
        # lambda x: 2048 * x ** 12 - 6144 * x ** 10 + 6912 * x ** 8 - 3584 * x ** 6 + 672 * x ** 4 - 33 * x ** 2 + 1,
        # lambda
        #     x: 4096 * x ** 13 - 13312 * x ** 11 + 16640 * x ** 9 - 9984 * x ** 7 + 2688 * x ** 5 - 252 * x ** 3 + 13 * x,
        # lambda
        #     x: 8192 * x ** 14 - 28672 * x ** 12 + 39424 * x ** 10 - 26880 * x ** 8 + 8448 * x ** 6 - 1092 * x ** 4 + 56 * x ** 2 - 1
    ]

def create_chebyshev_basis_element(deg: int = 7):
    if len(CHEBYSHEV_EXPLICT) > deg:
        return CHEBYSHEV_EXPLICT[deg]
    else:
        return lambda x: 2 * x * create_chebyshev_basis_element(deg - 1)(x) - create_chebyshev_basis_element(deg - 2)(x)

# def create_chebyshev_basis_functions(deg: int = 7):
#     return [create_chebyshev_basis_element(d) for d in range(deg)]



# def create_chebyshev_basis_functions_numpy(n: int) -> List[Callable[[np.ndarray], np.ndarray]]:
#     """Generate a list of Chebyshev polynomial basis functions up to degree n."""
#     basis_functions = []
#     for i in range(n):
#         # Create the Chebyshev polynomial of degree i
#         T_i = Chebyshev.basis(i)
#
#         # Append a function that evaluates this polynomial at given x values
#         basis_functions.append(lambda x, T_i=T_i: T_i(x))
#     return basis_functions


def inner_product_between_functions(func_a: Callable,
                                    func_b: Callable,
                                    start_point: float,
                                    end_point: float,
                                    num_samples: int = 1000) -> float:
    x_samples = np.linspace(start_point, end_point, num_samples)
    y_samples_a = func_a(x_samples)
    y_samples_b = func_b(x_samples)
    return simpson(y_samples_a * y_samples_b, x_samples)




def find_best_coefs_fourier(x: np.ndarray, y: np.ndarray, num_coefs: int, initial_guess: np.ndarray = None):
    initial_guess = np.ones(num_coefs) if initial_guess is None else initial_guess  # Initial guess for the coefficients
    residuals = create_residual_for_basis_functions(create_fourier_basis_functions(num_coefs))
    result = least_squares(residuals, initial_guess, args=(x, y),
                           method='trf', ftol=10e-7, xtol=10e-7, gtol=10e-7,
                           tr_solver='lsmr', jac='3-point')
    print(result)

    # Get the optimized coefficients
    return result.x


# def create_residual_for_basis_functions(basis_functions: List[Callable]) -> Callable:
#     def residuals(coefs, x, y):
#         func = lambda x: np.sum([coefs[j] * basis_functions[j](x) for j in range(len(basis_functions))], axis=0)
#         pred_y = func(x)
#         return np.sum((pred_y - y) ** 2)
#
#     return residuals
#
#
# def find_best_coefs_for_specific_basis_functions(x: np.ndarray, y: np.ndarray,
#                                                  basis_functions: List[Callable]) -> np.ndarray:
#     initial_guess = np.ones(len(basis_functions))
#     residual_function = create_residual_for_basis_functions(basis_functions)
#     result = least_squares(residual_function, initial_guess, args=(x, y),
#                            method='trf', ftol=10e-7, xtol=10e-7, gtol=10e-7,
#                            tr_solver='lsmr', jac='3-point')
#     # Get the optimized coefficients
#     return result.x


def create_residual_for_basis_functions(basis_functions: List[Callable]) -> Callable:
    def residuals(coefs, x, y):
        # Create a matrix where each column is the evaluation of a basis function
        basis_matrix = tf.stack([basis(x) for basis in basis_functions], axis=1)
        pred_y = tf.linalg.matvec(basis_matrix, coefs)  # Matrix-vector multiplication
        return (pred_y - y) ** 2  # Square the residuals array

    return residuals


# def find_best_coefs_for_specific_basis_functions(x: np.ndarray, y: np.ndarray,
#                                                  basis_functions: List[Callable],
#                                                  tol: float = 1e-10) -> np.ndarray:
#     initial_guess = np.ones(len(basis_functions))
#     residual_function = create_residual_for_basis_functions(basis_functions)
#     result = least_squares(residual_function, initial_guess, args=(x, y),
#                            method='trf', ftol=tol, xtol=tol, gtol=tol,
#                            tr_solver='lsmr', jac='3-point')
#     return result.x


def evaluate_basis_functions(x: np.ndarray, basis_functions: List[Callable]) -> np.ndarray:
    """
    Evaluates the basis functions at the given x values and constructs the design matrix.

    Parameters:
        x (np.ndarray): The x-values (independent variable).
        basis_functions (List[Callable]): List of basis functions.

    Returns:
        np.ndarray: The design matrix with shape (len(x), len(basis_functions)).
    """
    # Create a design matrix where each column corresponds to a basis function evaluated at x
    design_matrix = np.column_stack([basis_func(x) for basis_func in basis_functions])
    return design_matrix


def find_best_coefs_for_specific_basis_functions(x: np.ndarray, y: np.ndarray,
                                                 basis_functions: List[Callable]) -> np.ndarray:
    """
    Find the best coefficients for a set of basis functions to fit a given function y(x).

    Parameters:
        x (np.ndarray): The x-values (independent variable).
        y (np.ndarray): The y-values (dependent variable).
        basis_functions (List[Callable]): List of basis functions.

    Returns:
        np.ndarray: Optimal coefficients for the basis functions.
    """
    # Evaluate the basis functions to create the design matrix
    design_matrix = evaluate_basis_functions(x, basis_functions)

    # Solve the linear least squares problem
    coefs, residuals, rank, s = np.linalg.lstsq(design_matrix, y, rcond=None)

    return coefs


def create_func_from_basis_and_coefficients(basis_functions: List[Callable],
                                            coefs: Union[np.ndarray, List[float]]):
    return lambda x: np.sum([coefs[j] * basis_functions[j](x) for j in range(len(basis_functions))], axis=0)


if __name__ == '__main__':
    pass
