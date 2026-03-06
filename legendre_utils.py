import numpy as np
from sklearn.linear_model import Ridge, Lasso
from typing import Dict, Any, List, Callable

from numpy.polynomial.legendre import legvander  # <-- Legendre Vandermonde
from utils import calculate_rmse, get_norm_of_basis_functions, calc_l2_approximation_error, _gram_matrix_on_interval



# ---------- Legendre orthonormal basis on [-1,1] ----------
def legendre_basis(x: np.ndarray, n: int) -> np.ndarray:
    """
    Return the first n *orthonormal* Legendre basis functions evaluated at x.
    Orthonormality: ∫_{-1}^1 ϕ_k(x) ϕ_m(x) dx = δ_{km}
    Construction: ϕ_k(x) = sqrt((2k+1)/2) * L_k(x)
    """
    x = np.asarray(x)
    V = legvander(x, n - 1)                       # shape: (len(x), n), columns L_0..L_{n-1}
    scales = np.sqrt((2*np.arange(n) + 1) / 2.0)  # shape: (n,)
    return V * scales                              # broadcast across rows


def evaluate_legendre_polynomial_with_coefs(x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """
    Evaluate f(x) = ∑ c_k ϕ_k(x) where ϕ_k are orthonormal Legendre basis functions.
    """
    n = len(coefs)
    return legendre_basis(x, n) @ coefs


# ---------- Fitting (LS / Ridge / Lasso) on Legendre basis ----------
def fit_regular_ls(x: np.ndarray, y: np.ndarray, n: int):
    """
    Fit orthonormal Legendre series to data (x, y) via ordinary least squares.
    """
    Phi = legendre_basis(x, n)
    coefs, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
    return coefs


def fit_ridge(x: np.ndarray, y: np.ndarray, n: int, alpha: float = 1.0):
    """
    Ridge regression (L2) on orthonormal Legendre basis.
    For alpha=0, reduces to LS.
    """
    if alpha == 0:
        return fit_regular_ls(x, y, n)
    Phi = legendre_basis(x, n)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(Phi, y)
    return ridge.coef_


def fit_lasso(x: np.ndarray, y: np.ndarray, n: int, alpha: float = 1.0):
    """
    Lasso regression (L1) on orthonormal Legendre basis.
    For alpha=0, reduces to LS.
    """
    if alpha == 0:
        return fit_regular_ls(x, y, n)
    Phi = legendre_basis(x, n)
    lasso = Lasso(alpha=alpha, fit_intercept=False)
    lasso.fit(Phi, y)
    return lasso.coef_


# ---------- RMSE evaluation (unchanged interface) ----------
def evaluate_rmse_for_coefficients(coefs_dict: Dict[str, Dict[str, Any]],
                                   x_train: np.ndarray,
                                   y_train: np.ndarray,
                                   x_test: np.ndarray,
                                   y_test: np.ndarray,
                                   y_train_real: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates and stores RMSE for each coefficient set using the Legendre basis evaluator.
    """
    for name in coefs_dict:
        current_coefs = coefs_dict[name]['coefs']
        if isinstance(current_coefs, bool):
            coefs_dict[name]['train_results'] = {'rmse': False}
            coefs_dict[name]['test_results'] = {'rmse': False}
            coefs_dict[name]['succeeded'] = False
            continue

        coefs_dict[name]['succeeded'] = True
        y_train_preds = evaluate_legendre_polynomial_with_coefs(x_train, current_coefs)
        y_test_preds = evaluate_legendre_polynomial_with_coefs(x_test, current_coefs)

        coefs_dict[name]['train_results'] = {
            'rmse': calculate_rmse(y_train, y_train_preds),
            'rmse_real': calculate_rmse(y_train_real, y_train_preds),
            'preds': y_train_preds
        }
        coefs_dict[name]['test_results'] = {
            'rmse': calculate_rmse(y_test, y_test_preds),
            'preds': y_test_preds
        }

    return coefs_dict


def print_rmse_evaluation(coefs_dict: Dict[str, Dict[str, Any]]):
    for name in coefs_dict:
        res = "Failed"
        res_train = "Failed"
        res_train_real = "Failed"
        if coefs_dict[name].get('succeeded', False):
            res = f"{coefs_dict[name]['test_results']['rmse']:.4f}"
            res_train = f"{coefs_dict[name]['train_results']['rmse']:.4f}"
            res_train_real = f"{coefs_dict[name]['train_results']['rmse_real']:.4f}"
        print(f"RMSE - {name}: {res}, Train RMSE: {res_train}, Train RMSE Real: {res_train_real}")



# ---------- Optional: convenience creators for explicit φ_k callables ----------
def create_legendre_basis_functions_numpy(n: int):
    """
    Return [φ_0, ..., φ_{n-1}] where φ_k(x) are the orthonormal Legendre functions.
    """
    scales = np.sqrt((2*np.arange(n) + 1) / 2.0)

    def make_phi(k: int):
        s = float(scales[k])
        def phi(x):
            # Evaluate L_k at x via legvander and pick the column k
            xx = np.asarray(x)
            V = legvander(xx, k)   # shape (len(x), k+1)
            Lk = V[:, k]
            return s * Lk
        return phi

    return [make_phi(k) for k in range(n)]



# ---------- Example / quick self-check ----------
if __name__ == "__main__":
    n = 12
    phi_list = create_legendre_basis_functions_numpy(n)

    # Evaluate a few φ_k at 0.5
    from pprint import pprint
    pprint([phi_list[i](np.array([0.5]))[0] for i in range(n)])

    # Check Gram ≈ I on [-1,1]
    G = _gram_matrix_on_interval(phi_list, -1.0, 1.0, n_quad=4097)
    print(np.round(G, 6))



