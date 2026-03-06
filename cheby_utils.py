import numpy as np
from sklearn.linear_model import Ridge, Lasso

from typing import Dict, Any, List, Callable
from numpy.polynomial.chebyshev import chebvander
from utils import calculate_rmse, get_norm_of_basis_functions, calc_l2_approximation_error


def fit_regular_ls(x: np.ndarray, y: np.ndarray, n: int):
    """
    Fit the Chebyshev polynomial to data (x, y) using regular least squares (no constraints).
    """
    T = chebyshev_basis(x, n)
    # Solve for the coefficients that minimize the LS objective
    coefs, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
    # coefs = fit_ridge(x, y, n, alpha=0)
    return coefs


# Ridge, Lasso, and Elastic Net Regression fitting functions
def fit_ridge(x: np.ndarray, y: np.ndarray, n: int, alpha: float = 1.0):
    """
    Fit the Chebyshev polynomial to data (x, y) using Ridge Regression (L2 regularization).
    """
    if alpha == 0:
        return fit_regular_ls(x, y, n)
    T = chebyshev_basis(x, n)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(T, y)
    return ridge.coef_


def fit_lasso(x: np.ndarray, y: np.ndarray, n: int, alpha: float = 1.0):
    """
    Fit the Chebyshev polynomial to data (x, y) using Lasso Regression (L1 regularization).
    """
    if alpha == 0:
        return fit_regular_ls(x, y, n)
    T = chebyshev_basis(x, n)
    lasso = Lasso(alpha=alpha, fit_intercept=False)
    lasso.fit(T, y)
    return lasso.coef_


def evaluate_chebyshev_polynomial_with_coefs(x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    n = len(coefs)
    return chebyshev_basis(x, n) @ coefs


# Function to generate Chebyshev basis
def chebyshev_basis(x: np.ndarray, n: int):
    """
    Generates the first n Chebyshev polynomials evaluated at points x.
    """
    basis = create_chebyshev_basis_functions_numpy(n)
    T = np.array([basis[deg](x) for deg in range(n)]).T
    return T


def create_chebyshev_basis_functions_numpy(n, n_quad=32749, eps=1e-5):
    """
    Return [phi_0, ..., phi_{n-1}] orthonormal on [-1,1] for plain L2 (no external weight).
    Construction:
      raw b_k(x) = T_k(x) / (1 - x^2)^(1/4)
    Then build Simpson-rule Gram on a uniform grid and apply symmetric orthonormalization G^{-1/2}.
    """
    if n <= 0:
        return []

    # --- Simpson grid (odd n_quad), include endpoints like your checker ---
    if n_quad % 2 == 0:
        n_quad += 1
    a, b = -1.0, 1.0
    xq = np.linspace(a, b, n_quad)
    # avoid evaluating exactly at ±1 inside phi
    xc = np.clip(xq, a + eps, b - eps)

    # Simpson weights
    h = (b - a) / (n_quad - 1)
    w = np.ones(n_quad)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    w *= (h / 3.0)

    # --- Evaluate raw basis on grid ---
    # V[j,k] = T_k(xc[j])  -> chebvander(x, m) returns shape (len(x), m+1)
    V = chebvander(xc, n - 1)                 # shape (n_quad, n)
    denom_q = np.power(1.0 - xc * xc, 0.25)   # (1 - x^2)^(1/4)
    B = (V.T) / denom_q                       # shape (n, n_quad): rows are b_k(xq)

    # --- Gram with Simpson: G = B diag(w) B^T ---
    Bw = B * w                                # broadcast weights over columns
    G = Bw @ B.T                              # shape (n, n)

    # --- Symmetric orthonormalization: U = G^{-1/2} ---
    evals, evecs = np.linalg.eigh(G)
    evals = np.maximum(evals, 1e-15)
    invsqrt = 1.0 / np.sqrt(evals)
    U = evecs @ (invsqrt[:, None] * evecs.T)  # shape (n, n)

    # φ_k(x) = ∑_i U_{k,i} b_i(x)
    def make_phi(k):
        uk = U[k].copy()  # shape (n,)
        def phi(x):
            xx = np.asarray(x)
            scalar = (xx.ndim == 0)
            if scalar:
                xx = xx.reshape(1)
            xx = np.clip(xx, a + eps, b - eps)
            Vx = chebvander(xx, n - 1)                  # shape (len(xx), n)
            Bx = (Vx.T) / np.power(1.0 - xx*xx, 0.25)   # shape (n, len(xx))
            y = np.tensordot(uk, Bx, axes=(0, 0))       # shape (len(xx),)
            if scalar:
                return float(y[0])                      # return true scalar for scalar input
            return y
        return phi

    return [make_phi(k) for k in range(n)]


def evaluate_rmse_for_coefficients(coefs_dict: Dict[str, Dict[str, Any]],
                                   x_train: np.ndarray,
                                   y_train: np.ndarray,
                                   x_test: np.ndarray,
                                   y_test: np.ndarray,
                                   y_train_real: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates and stores RMSE for each coefficient set.

    Args:
        coefs_dict: Dictionary containing coefficient sets for different methods.
        x_train: Training data inputs.
        y_train: True training data outputs.
        x_test: Test data inputs.
        y_test: True test data outputs.

    Returns:
        Dictionary with train and test RMSEs for each coefficient set.
    """
    for name in coefs_dict:
        current_coefs = coefs_dict[name]['coefs']
        if isinstance(current_coefs, bool):
            coefs_dict[name]['train_results'] = {'rmse': False}
            coefs_dict[name]['test_results'] = {'rmse': False}
            coefs_dict[name]['succeeded'] = False
            continue
        coefs_dict[name]['succeeded'] = True
        y_train_preds = evaluate_chebyshev_polynomial_with_coefs(x_train, current_coefs)
        y_test_preds = evaluate_chebyshev_polynomial_with_coefs(x_test, current_coefs)
        coefs_dict[name]['train_results'] = {'rmse': calculate_rmse(y_train, y_train_preds),
                                             'rmse_real': calculate_rmse(y_train_real, y_train_preds),
                                             'preds': y_train_preds}
        coefs_dict[name]['test_results'] = {'rmse': calculate_rmse(y_test, y_test_preds),
                                            'preds': y_test_preds}

    return coefs_dict


def print_rmse_evaluation(coefs_dict: Dict[str, Dict[str, Any]]):
    for name in coefs_dict:
        res = "Failed"
        res_train = "Failed"
        if coefs_dict[name]['succeeded']:
            res = f"{coefs_dict[name]['test_results']['rmse']:.4f}"
            res_train = f"{coefs_dict[name]['train_results']['rmse']:.4f}"
            res_train_real = f"{coefs_dict[name]['train_results']['rmse_real']:.4f}"
        print(f"RMSE - {name}: {res}, Train RMSE: {res_train}, Train RMSE Real: {res_train_real}")

from scipy.integrate import simpson

def _gram_matrix_on_interval(
        phi_list: List[Callable[[np.array], float]],
        a: float,
        b: float,
        n_quad: int = 4097
) -> np.ndarray:
    """Compute Gram matrix G_ij = ∫_a^b φ_i(x) φ_j(x) dx via Simpson's rule."""
    if n_quad % 2 == 0:
        n_quad += 1
    x = np.linspace(a, b, n_quad)
    Phi = np.vstack([phi(x) for phi in phi_list])  # shape (d, n)
    d = Phi.shape[0]
    G = np.empty((d, d), float)
    for i in range(d):
        for j in range(i, d):
            val = simpson(Phi[i] * Phi[j], x)

            G[i, j] = val
            G[j, i] = val
    return G

if __name__ == "__main__":
    phi_list = create_chebyshev_basis_functions_numpy(4)
    from pprint import pprint

    pprint([phi_list[i](0.5) for i in range(4)])  # evaluate at x=0.5

    # check Gram matrix with your routine
    G = _gram_matrix_on_interval(phi_list, -1, 1, n_quad=4097)
    print(np.round(G, 6))

