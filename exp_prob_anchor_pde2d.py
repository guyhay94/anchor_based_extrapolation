import json
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.stats import beta


"""
Experiment C (simplified): 2D Poisson ground truth + probabilistic anchor projection

- Domain D=[0,1]^2
- Solve -Δu=s with u=0 on boundary (5-pt FD + sparse solve), define f=u
- Training region: Ω = D \\ Ξ (Ξ is a boundary-touching rectangle)
- Fit baseline g in sine basis on Ω samples (ridge)
- Fit anchor a in same basis on Ω samples (LS)
- Build G_Ω and G_Ξ (Simpson integrals) and compute Ω-whitened G_Ξ~ to get eig_max
- kappa_q = eig_max * BetaPPF(q; 1/2, (K-1)/2) with q=1-alpha (dim = K)
- delta = sqrt(kappa_q) * ||a-f||_L2(Ω)   (L2 integral)
- Project g -> h by QCQP in coefficient space: minimize ||h-g||_Ξ s.t. ||h-a||_Ξ <= delta
"""


@dataclass(frozen=True)
class Config:
    N: int = 200
    K: int = 10
    M_train: int = 200
    seed: int = 42
    snr_db: float = 35.0
    predicted_method: str = "lasso"  # {"ridge","lasso"}
    ridge_lambda: float = 1e-2
    lasso_alpha: float = 2e-4
    lasso_max_iters: int = 4000
    lasso_tol: float = 1e-10
    alpha: float = 0.3  # q = 1-alpha
    xi_patch: Tuple[float, float, float, float] = (0.8, 1.0, 0.7, 1.0)
    xi_refine_factor: int = 3  # refine Xi grid for error integrals (>=1)
    # Poisson RHS source: sine-only series (Dirichlet-compatible)
    source_num_terms: int = 25
    # IMPORTANT: to ensure the *true* f is representable in the same basis we predict with,
    # we require source_freq_range[1] <= K (otherwise f contains modes outside the K×K model space).
    source_freq_range: Tuple[int, int] = (2, K)  # inclusive; must satisfy kmax <= K
    # We sample the *solution* coefficients c_f first (so we can make the signal strong),
    # then derive RHS coefficients via c_s = pi^2(kx^2+ky^2) * c_f.
    f_coef_abs_range: Tuple[float, float] = (0.1, 7.0)  # |c_f| range for each chosen mode
    save_dir: str = "./results/exp_prob_anchor_pde2d"
    save_outputs: bool = True
    # Search (requested): find a config where true->anchor is 20% smaller than pred->anchor
    search_mode: bool = False
    search_max_trials: int = 400
    search_print_every: int = 25


def create_basis_functions(K: int) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, int], np.ndarray]]:
    """
    Return:
      - design_matrix(points_xy) -> A (M, K^2)
      - eval_on_grid(coeffs, N) -> values (N,N) with indexing [y,x]
    Basis: sin(pi*kx*x) sin(pi*ky*y), kx,ky=1..K (Dirichlet-compatible).
    """

    def design_matrix(points_xy: np.ndarray) -> np.ndarray:
        x = points_xy[:, 0:1]
        y = points_xy[:, 1:2]
        k = np.arange(1, K + 1, dtype=float)
        Sx = np.sin(np.pi * x @ k[None, :])  # (M,K)
        Sy = np.sin(np.pi * y @ k[None, :])  # (M,K)
        return (Sx[:, :, None] * Sy[:, None, :]).reshape(points_xy.shape[0], K * K)

    def eval_on_grid(coeffs: np.ndarray, N: int) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float).reshape(K, K)
        x = np.linspace(0.0, 1.0, N)
        y = np.linspace(0.0, 1.0, N)
        k = np.arange(1, K + 1, dtype=float)
        Sx = np.sin(np.pi * x[:, None] * k[None, :])  # (N,K)
        Sy = np.sin(np.pi * y[:, None] * k[None, :])  # (N,K)
        U_xy = (Sx @ coeffs) @ Sy.T  # (N,N) indexed [x,y]
        return U_xy.T  # (N,N) indexed [y,x]

    return design_matrix, eval_on_grid


def create_gram_matrix(K: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return G (K^2, K^2) where G_ij = ∫_{rect} φ_i φ_j dx dy (Simpson tensor-product).
    Rectangle is defined by 1D grids x and y.
    """

    def simpson_weights_uniform(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        n = int(t.size)
        if n < 2:
            raise ValueError("Need at least 2 points for Simpson weights.")
        h = float(t[1] - t[0])
        if not np.allclose(np.diff(t), h, rtol=0, atol=1e-12 * (1.0 + abs(h))):
            raise ValueError("Expected uniform grid for Simpson weights.")
        w = np.zeros(n, dtype=float)
        if n % 2 == 1:
            w[0] = 1.0
            w[-1] = 1.0
            w[1:-1:2] = 4.0
            w[2:-1:2] = 2.0
            w *= h / 3.0
            return w
        # even: Simpson on first n-1 + trapezoid on last interval
        n1 = n - 1
        w1 = np.zeros(n1, dtype=float)
        w1[0] = 1.0
        w1[-1] = 1.0
        w1[1:-1:2] = 4.0
        w1[2:-1:2] = 2.0
        w1 *= h / 3.0
        w[:n1] += w1
        w[-2] += h / 2.0
        w[-1] += h / 2.0
        return w

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    wx = simpson_weights_uniform(x)
    wy = simpson_weights_uniform(y)
    k = np.arange(1, K + 1, dtype=float)
    Sx = np.sin(np.pi * x[:, None] * k[None, :])  # (nx,K)
    Sy = np.sin(np.pi * y[:, None] * k[None, :])  # (ny,K)
    Gx = Sx.T @ (wx[:, None] * Sx)
    Gy = Sy.T @ (wy[:, None] * Sy)
    Gx = 0.5 * (Gx + Gx.T)
    Gy = 0.5 * (Gy + Gy.T)
    return np.kron(Gy, Gx)  # vec(C) with ky fastest


def l2_error_on_square(u_yx: np.ndarray, v_yx: np.ndarray, x: np.ndarray, y: np.ndarray, mask_yx: np.ndarray | None = None) -> float:
    """
    Root-L2 error using a tensor-product Simpson integral on the full (x,y) grid.
    If mask_yx is provided, integrates only over that subset (using the same weights).
    """

    def simpson_weights_uniform(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        n = int(t.size)
        if n < 2:
            raise ValueError("Need at least 2 points for Simpson weights.")
        h = float(t[1] - t[0])
        if not np.allclose(np.diff(t), h, rtol=0, atol=1e-12 * (1.0 + abs(h))):
            raise ValueError("Expected uniform grid for Simpson weights.")
        w = np.zeros(n, dtype=float)
        if n % 2 == 1:
            w[0] = 1.0
            w[-1] = 1.0
            w[1:-1:2] = 4.0
            w[2:-1:2] = 2.0
            w *= h / 3.0
            return w
        n1 = n - 1
        w1 = np.zeros(n1, dtype=float)
        w1[0] = 1.0
        w1[-1] = 1.0
        w1[1:-1:2] = 4.0
        w1[2:-1:2] = 2.0
        w1 *= h / 3.0
        w[:n1] += w1
        w[-2] += h / 2.0
        w[-1] += h / 2.0
        return w

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    wx = simpson_weights_uniform(x)
    wy = simpson_weights_uniform(y)
    w2d = wy[:, None] * wx[None, :]
    diff2 = (u_yx - v_yx) ** 2
    if mask_yx is not None:
        val = float(np.sum(diff2[mask_yx] * w2d[mask_yx]))
    else:
        val = float(np.sum(diff2 * w2d))
    return float(np.sqrt(max(val, 0.0)))


def fit_model(
    A: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    ridge_lambda: float = 0.0,
    lasso_alpha: float = 0.0,
    lasso_max_iters: int = 4000,
    lasso_tol: float = 1e-10,
    seed: int = 0,
) -> np.ndarray:
    """
    Fit coefficients in the sine basis.
    - method="ls":    min ||A c - y||^2
    - method="ridge": min ||A c - y||^2 + ridge_lambda ||c||^2
    - method="lasso": min 0.5||A c - y||^2 + lasso_alpha ||c||_1   (ISTA)
    """
    method = method.lower().strip()
    if method == "ls":
        return np.linalg.lstsq(A, y, rcond=None)[0]
    if method == "ridge":
        ATA = A.T @ A
        ATy = A.T @ y
        n = ATA.shape[0]
        return np.linalg.solve(ATA + float(ridge_lambda) * np.eye(n), ATy)
    if method == "lasso":
        # ISTA: c <- S_{t*alpha}(c - t * A^T(Ac-y)), with t <= 1/L (L = ||A^T A||_2)
        rng = np.random.default_rng(int(seed))
        m, n = A.shape
        c = np.zeros(n, dtype=float)

        # Power iteration for L
        v = rng.normal(size=n)
        v /= np.linalg.norm(v) + 1e-12
        for _ in range(50):
            v = A.T @ (A @ v)
            nv = np.linalg.norm(v)
            if nv <= 0:
                break
            v /= nv
        L = float(v @ (A.T @ (A @ v)))
        L = max(L, 1e-12)
        step = 1.0 / L

        lam = float(lasso_alpha)
        tol = float(lasso_tol)
        max_iters = int(lasso_max_iters)

        def soft(z: np.ndarray, t: float) -> np.ndarray:
            return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

        prev = c.copy()
        for _ in range(max_iters):
            grad = A.T @ (A @ c - y)
            c = soft(c - step * grad, step * lam)
            if np.linalg.norm(c - prev) <= tol:
                break
            prev = c.copy()
        return c
    raise ValueError("method must be 'ls' or 'ridge'")


def project_to_feasible(c_g: np.ndarray, c_a: np.ndarray, delta: float, G_xi: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    QCQP projection (SLSQP):
      minimize (c-c_g)^T G_xi (c-c_g)
      s.t.     (c-c_a)^T G_xi (c-c_a) <= delta^2
    """
    G = 0.5 * (G_xi + G_xi.T)
    delta2 = float(delta * delta)

    def obj(c: np.ndarray) -> float:
        d = c - c_g
        return float(d @ G @ d)

    def obj_jac(c: np.ndarray) -> np.ndarray:
        d = c - c_g
        return 2.0 * (G @ d)

    def ineq(c: np.ndarray) -> float:
        d = c - c_a
        return float(delta2 - (d @ G @ d))  # >= 0 feasible

    def ineq_jac(c: np.ndarray) -> np.ndarray:
        d = c - c_a
        return -2.0 * (G @ d)

    x0 = c_g.copy()
    if ineq(x0) < 0.0:
        x0 = c_a.copy()

    res = minimize(
        fun=obj,
        x0=x0,
        jac=obj_jac,
        constraints=[{"type": "ineq", "fun": ineq, "jac": ineq_jac}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 800},
    )
    c_h = res.x if res.success else c_a.copy()
    info = {
        "success": float(bool(res.success)),
        "niter": float(getattr(res, "nit", 0)),
        "final_constraint": float(ineq(c_h)),
        "final_obj": float(getattr(res, "fun", float("nan"))),
    }
    return c_h, info


def main(cfg: Config | None = None) -> None:
    cfg = Config() if cfg is None else cfg
    os.makedirs(cfg.save_dir, exist_ok=True)

    def run_flow(cfg: Config, *, save_outputs: bool, verbose: bool) -> Dict[str, object]:
        """
        Run a single instance of the experiment and return metrics needed for search.
        Keeps the same overall structure as the original main.
        """
        # --- grid + Xi/Omega ---
        x = np.linspace(0.0, 1.0, cfg.N)
        y = np.linspace(0.0, 1.0, cfg.N)
        X, Y = np.meshgrid(x, y, indexing="xy")

        x0, x1, y0, y1 = cfg.xi_patch
        mask_xi = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        mask_omega = ~mask_xi

        # --- source s(x,y): sine-only series ---
        rng = np.random.default_rng(cfg.seed)
        s = np.zeros((cfg.N, cfg.N), dtype=float)
        kmin, kmax = int(cfg.source_freq_range[0]), int(cfg.source_freq_range[1])
        if kmin < 1 or kmax < kmin:
            raise ValueError("source_freq_range must satisfy 1 <= kmin <= kmax")
        if kmax > int(cfg.K):
            raise ValueError(
                f"source_freq_range={cfg.source_freq_range} uses kmax={kmax} but K={cfg.K}. "
                f"Please set source_freq_range so kmax<=K, or increase K."
            )

        possible_modes = [(kx, ky) for kx in range(kmin, kmax + 1) for ky in range(kmin, kmax + 1)]
        if int(cfg.source_num_terms) > len(possible_modes):
            raise ValueError("source_num_terms exceeds number of available (kx,ky) modes in the given range.")
        chosen_modes = rng.choice(len(possible_modes), size=int(cfg.source_num_terms), replace=False)

        f_terms: list[dict] = []
        f_coef_map: dict[tuple[int, int], float] = {}
        source_terms: list[dict] = []
        source_coef_map: dict[tuple[int, int], float] = {}

        cmin, cmax = float(cfg.f_coef_abs_range[0]), float(cfg.f_coef_abs_range[1])
        if cmin <= 0 or cmax < cmin:
            raise ValueError("f_coef_abs_range must satisfy 0 < min <= max")

        for idx in chosen_modes:
            kx, ky = possible_modes[int(idx)]
            cf = float(rng.uniform(cmin, cmax)) * float(rng.choice([-1.0, 1.0]))
            cs = float((np.pi**2) * (kx * kx + ky * ky) * cf)
            f_terms.append({"kx": int(kx), "ky": int(ky), "c_f": float(cf)})
            source_terms.append({"kx": int(kx), "ky": int(ky), "c_s": float(cs)})
            f_coef_map[(int(kx), int(ky))] = float(cf)
            source_coef_map[(int(kx), int(ky))] = float(cs)
            s += cs * np.sin(np.pi * kx * X) * np.sin(np.pi * ky * Y)

        # --- solve Poisson: -Δu = s, u=0 on boundary (5-pt Laplacian) ---
        def poisson_dirichlet_5pt(source: np.ndarray) -> np.ndarray:
            N = int(source.shape[0])
            h = 1.0 / (N - 1)
            n_int = (N - 2) * (N - 2)
            main = 4.0 * np.ones(n_int, dtype=float)
            off1 = -1.0 * np.ones(n_int - 1, dtype=float)
            offN = -1.0 * np.ones(n_int - (N - 2), dtype=float)
            row_len = (N - 2)
            cut = (np.arange(n_int - 1) % row_len) == (row_len - 1)
            off1[cut] = 0.0
            A = sp.diags(
                diagonals=[main, off1, off1, offN, offN],
                offsets=[0, -1, 1, -row_len, row_len],
                shape=(n_int, n_int),
                format="csr",
            )
            rhs = (h**2) * source[1:-1, 1:-1].reshape(-1)
            u_int = spla.spsolve(A, rhs)
            u = np.zeros((N, N), dtype=float)
            u[1:-1, 1:-1] = u_int.reshape((N - 2, N - 2))
            return u

        f = poisson_dirichlet_5pt(s)

        # --- sample Ω training points ---
        omega_idx = np.flatnonzero(mask_omega.reshape(-1))
        if cfg.M_train > omega_idx.size:
            raise ValueError(f"M_train={cfg.M_train} exceeds available Ω grid points {omega_idx.size}.")
        chosen = rng.choice(omega_idx, size=int(cfg.M_train), replace=False)
        iy, ix = np.unravel_index(chosen, (cfg.N, cfg.N))
        pts_train = np.column_stack([x[ix], y[iy]])
        y_train_true = f.reshape(-1)[chosen]

        # Add SNR noise
        p_signal = float(np.mean(y_train_true**2))
        p_noise = p_signal / (10.0 ** (cfg.snr_db / 10.0)) if p_signal > 0 else 0.0
        noise_std = float(np.sqrt(max(p_noise, 0.0)))
        y_train = y_train_true + rng.normal(0.0, noise_std, size=y_train_true.shape)

        # --- basis + fits ---
        design_matrix, eval_on_grid = create_basis_functions(cfg.K)
        A = design_matrix(pts_train)
        c_a = fit_model(A, y_train, method="ls")  # anchor
        c_g = fit_model(
            A,
            y_train,
            method=cfg.predicted_method,
            ridge_lambda=cfg.ridge_lambda,
            lasso_alpha=cfg.lasso_alpha,
            lasso_max_iters=cfg.lasso_max_iters,
            lasso_tol=cfg.lasso_tol,
            seed=cfg.seed,
        )  # baseline
        a = eval_on_grid(c_a, cfg.N)
        g = eval_on_grid(c_g, cfg.N)

        # True coefficients in the prediction basis ordering (kx,ky are 1-indexed; ky fastest).
        c_f = np.zeros(cfg.K * cfg.K, dtype=float)
        for (kx, ky), cf in f_coef_map.items():
            c_f[(kx - 1) * cfg.K + (ky - 1)] = float(cf)

        # --- Gram matrices and whitening ---
        xi_ys = np.where((y >= y0) & (y <= y1))[0]
        xi_xs = np.where((x >= x0) & (x <= x1))[0]
        x_xi = x[xi_xs]
        y_xi = y[xi_ys]
        if x_xi.size < 2 or y_xi.size < 2:
            raise ValueError("Xi patch too small for Simpson-based Gram/integrals (need >=2 grid points each axis).")

        G_total = create_gram_matrix(cfg.K, x=x, y=y)
        G_xi = create_gram_matrix(cfg.K, x=x_xi, y=y_xi)
        G_omega = 0.5 * ((G_total - G_xi) + (G_total - G_xi).T)

        omega_eigs = np.linalg.eigvalsh(G_omega)
        omega_max_ev = float(np.max(np.abs(omega_eigs)))
        omega_eps = float(1e-12 * omega_max_ev) if omega_max_ev > 0 else 1e-12
        eigvals, eigvecs = np.linalg.eigh(G_omega)
        eigvals = np.maximum(eigvals, omega_eps)
        T = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        G_xi_tilde = 0.5 * (T @ G_xi @ T.T + (T @ G_xi @ T.T).T)
        eigs = np.sort(np.linalg.eigvalsh(G_xi_tilde))
        eig_max = float(eigs[-1])
        eig_2 = float(eigs[-2]) if eigs.size >= 2 else float("nan")

        # --- probabilistic kappa (dim = K, q = 1-alpha) ---
        q = float(1.0 - cfg.alpha)
        kappa_q = float(eig_max * beta.ppf(q, 0.5, (cfg.K - 1) / 2.0))
        e_omega_l2 = float(l2_error_on_square(a, f, x=x, y=y, mask_yx=mask_omega))
        delta = float(np.sqrt(max(kappa_q, 0.0)) * e_omega_l2)
        # Also report the radius at alpha=0 (q=1), i.e. the Beta supremum quantile.
        kappa_q1 = float(eig_max * beta.ppf(1.0, 0.5, (cfg.K - 1) / 2.0))
        delta_alpha0 = float(np.sqrt(max(kappa_q1, 0.0)) * e_omega_l2)

        # --- distances to anchor in Xi coefficient metric (THIS is what we search on) ---
        dist_pred_anchor_xi = float(np.sqrt(max((c_g - c_a) @ G_xi @ (c_g - c_a), 0.0)))
        dist_true_anchor_xi = float(np.sqrt(max((c_f - c_a) @ G_xi @ (c_f - c_a), 0.0)))

        # --- projection ---
        pred_in_feasible = dist_pred_anchor_xi <= delta
        c_h, proj_info = (
            project_to_feasible(c_g, c_a, delta, G_xi)
            if not pred_in_feasible
            else (
                c_g.copy(),
                {"success": 1.0, "niter": 0.0, "final_constraint": float(delta * delta - dist_pred_anchor_xi**2), "final_obj": 0.0},
            )
        )
        h = eval_on_grid(c_h, cfg.N)

        # --- Improvement bounds in the Xi metric (same metric as the QCQP) ---
        # This matches the projection-style bounds used across the repo:
        #   upper = ||g-h||_Xi
        #   lower = ||g-h||_Xi * sqrt( ||g-h||_Xi / (||g-h||_Xi + delta) )
        # where delta is the feasible radius and norms are in the Xi metric.
        delta_pred_proj_xi = float(np.sqrt(max((c_g - c_h) @ G_xi @ (c_g - c_h), 0.0)))
        upper_bound = float(delta_pred_proj_xi)
        if upper_bound <= 0.0:
            lower_bound = 0.0
        else:
            lower_bound = float(upper_bound * np.sqrt(upper_bound / (upper_bound + float(delta))))

        # Actual improvement in the same Xi coefficient metric (so we can compare to bounds)
        err_fg_xi_coef = float(np.sqrt(max((c_g - c_f) @ G_xi @ (c_g - c_f), 0.0)))
        err_fh_xi_coef = float(np.sqrt(max((c_h - c_f) @ G_xi @ (c_h - c_f), 0.0)))
        improvement_xi_coef = float(err_fg_xi_coef - err_fh_xi_coef)
        improvement_in_bounds = bool((improvement_xi_coef >= lower_bound - 1e-12) and (improvement_xi_coef <= upper_bound + 1e-12))

        # --- Xi integral error on refined Xi grid ---
        refine = max(int(cfg.xi_refine_factor), 1)
        nx_xi = int(max(21, (x_xi.size - 1) * refine + 1))
        ny_xi = int(max(21, (y_xi.size - 1) * refine + 1))
        if nx_xi % 2 == 0:
            nx_xi += 1
        if ny_xi % 2 == 0:
            ny_xi += 1
        x_xi_f = np.linspace(float(x0), float(x1), nx_xi)
        y_xi_f = np.linspace(float(y0), float(y1), ny_xi)
        Xf, Yf = np.meshgrid(x_xi_f, y_xi_f, indexing="xy")
        pts_f = np.column_stack([Yf.reshape(-1), Xf.reshape(-1)])  # (y,x)

        interp_f = RegularGridInterpolator((y, x), f, bounds_error=False, fill_value=None)
        interp_g = RegularGridInterpolator((y, x), g, bounds_error=False, fill_value=None)
        interp_h = RegularGridInterpolator((y, x), h, bounds_error=False, fill_value=None)

        f_xi_f = interp_f(pts_f).reshape(ny_xi, nx_xi)
        g_xi_f = interp_g(pts_f).reshape(ny_xi, nx_xi)
        h_xi_f = interp_h(pts_f).reshape(ny_xi, nx_xi)

        err_fg_xi = l2_error_on_square(f_xi_f, g_xi_f, x=x_xi_f, y=y_xi_f, mask_yx=None)
        err_fh_xi = l2_error_on_square(f_xi_f, h_xi_f, x=x_xi_f, y=y_xi_f, mask_yx=None)
        improvement = float(err_fg_xi - err_fh_xi)

        if verbose:
            print("=" * 90)
            print(f"N={cfg.N} | K={cfg.K} | M_train={cfg.M_train} | method={cfg.predicted_method}")
            if cfg.predicted_method.lower() == "ridge":
                print(f"ridge_lambda={cfg.ridge_lambda:g}")
            else:
                print(f"lasso_alpha={cfg.lasso_alpha:g} | lasso_max_iters={cfg.lasso_max_iters} | lasso_tol={cfg.lasso_tol:g}")
            print(
                f"Xi patch={cfg.xi_patch} | xi_refine_factor={cfg.xi_refine_factor} "
                f"(xi fine grid {ny_xi}x{nx_xi}) | alpha={cfg.alpha} (q={q:.2f}) | SNR={cfg.snr_db:.1f} dB | noise_std={noise_std:.3e}"
            )
            print(
                f"eig_max={eig_max:.6e} | eig_2={eig_2:.6e} | "
                f"kappa_q={kappa_q:.6e} | delta={delta:.6e} | "
                f"delta(alpha=0)={delta_alpha0:.6e}"
            )
            print(f"pred->anchor (Xi coef metric): {dist_pred_anchor_xi:.6e}")
            print(f"true->anchor (Xi coef metric): {dist_true_anchor_xi:.6e}")
            print(f"predicted Xi error:   ||f-g||_Xi = {err_fg_xi:.6e}")
            print(f"projected Xi error:   ||f-h||_Xi = {err_fh_xi:.6e}")
            print(f"improvement:          {improvement:.6e}")
            print(
                f"bounds (Xi coef metric): lower={lower_bound:.6e} | upper={upper_bound:.6e} | "
                f"impr_coef={improvement_xi_coef:.6e} | in_region={int(improvement_in_bounds)}"
            )
            print(f"projection(SLSQP): success={int(proj_info.get('success', 0))} | niter={int(proj_info.get('niter', 0))} | constraint={proj_info.get('final_constraint', float('nan')):.3e}")
            print("=" * 90)

        results: Dict[str, object] = {
            "dist_pred_anchor_xi_coef": dist_pred_anchor_xi,
            "dist_true_anchor_xi_coef": dist_true_anchor_xi,
            "ratio_true_over_pred": (dist_true_anchor_xi / dist_pred_anchor_xi) if dist_pred_anchor_xi > 0 else float("inf"),
            "noise_std": noise_std,
            "eig_max": eig_max,
            "eig_2": eig_2,
            "kappa_q": kappa_q,
            "delta": delta,
            "kappa_q1": kappa_q1,
            "delta_alpha0": delta_alpha0,
            "e_omega_l2": e_omega_l2,
            "err_fg_xi": err_fg_xi,
            "err_fh_xi": err_fh_xi,
            "improvement": improvement,
            "err_fg_xi_coef": err_fg_xi_coef,
            "err_fh_xi_coef": err_fh_xi_coef,
            "improvement_xi_coef": improvement_xi_coef,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "improvement_in_bounds": bool(improvement_in_bounds),
            "proj_info": proj_info,
            "f_terms": f_terms,
            "source_terms": source_terms,
            "source_coef_map": {f"{kx},{ky}": v for (kx, ky), v in source_coef_map.items()},
            "f_coef_map": {f"{kx},{ky}": v for (kx, ky), v in f_coef_map.items()},
        }

        if save_outputs:
            # Save raw fields + figures with Xi marked (red rectangle) and NO titles.
            os.makedirs(cfg.save_dir, exist_ok=True)

            # Save arrays (ground truth f, predictor g, projection h) for inspection.
            np.savez(
                os.path.join(cfg.save_dir, "fields.npz"),
                x=x,
                y=y,
                f=f,
                g=g,
                h=h,
                xi_patch=np.array(cfg.xi_patch, dtype=float),
                seed=np.array([cfg.seed], dtype=int),
            )

            def save_heatmap(arr_yx: np.ndarray, path: str, *, cmap: str = "viridis", vmin=None, vmax=None) -> None:
                plt.figure(figsize=(6, 5))
                plt.imshow(arr_yx, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=(0, 1, 0, 1))
                plt.colorbar()
                # Xi rectangle overlay (red lines)
                plt.plot([x0, x1], [y0, y0], color="red", linewidth=2)
                plt.plot([x0, x1], [y1, y1], color="red", linewidth=2)
                plt.plot([x0, x0], [y0, y1], color="red", linewidth=2)
                plt.plot([x1, x1], [y0, y1], color="red", linewidth=2)
                plt.tight_layout()
                plt.savefig(path, dpi=150)
                plt.close()

            # Save value fields
            save_heatmap(f, os.path.join(cfg.save_dir, "f_values.png"), cmap="viridis")
            save_heatmap(g, os.path.join(cfg.save_dir, "g_values.png"), cmap="viridis")
            save_heatmap(h, os.path.join(cfg.save_dir, "h_values.png"), cmap="viridis")

            # Save error fields (mask outside Xi for clarity)
            err_fg = np.where(mask_xi, np.abs(f - g), np.nan)
            err_fh = np.where(mask_xi, np.abs(f - h), np.nan)
            # Use a shared vmax for fair comparison (if h is much closer, this will show it).
            vmax_err = float(np.nanmax([err_fg, err_fh]))
            save_heatmap(err_fg, os.path.join(cfg.save_dir, "err_f_minus_g_xi.png"), cmap="magma", vmin=0.0, vmax=vmax_err)
            save_heatmap(err_fh, os.path.join(cfg.save_dir, "err_f_minus_h_xi.png"), cmap="magma", vmin=0.0, vmax=vmax_err)

            # Save JSON summary
            payload = {"config": asdict(cfg), "results": results}
            with open(os.path.join(cfg.save_dir, "results.json"), "w") as f_json:
                json.dump(payload, f_json, indent=2)
        return results

    if not cfg.search_mode:
        _ = run_flow(cfg, save_outputs=cfg.save_outputs, verbose=True)
        return

    # --- Search loop (requested) ---
    # Target: dist_true_anchor_xi <= 0.8 * dist_pred_anchor_xi  (20% smaller), both in Xi coef metric.
    rng = np.random.default_rng(cfg.seed)
    best_ratio = float("inf")
    best_cfg: Config | None = None
    best_res: Dict[str, object] | None = None

    for t in range(int(cfg.search_max_trials)):
        # Allowed parameters to change:
        # - xi_patch: choose x0,y0 in [0.5,0.9] (endpoints allow 1.0, but 1.0 gives degenerate Xi; avoid).
        x0 = float(rng.uniform(0.5, 0.9))
        y0 = float(rng.uniform(0.5, 0.9))
        xi_patch = (x0, 1.0, y0, 1.0)

        # - source_num_terms: 5..60
        source_num_terms = int(rng.integers(5, 61))

        # - f_coef_abs_range: 0.1..10 (use min=0.1, sample max)
        fmax = float(rng.uniform(0.1, 10.0))
        f_coef_abs_range = (0.1, max(0.1, fmax))

        # - lasso_alpha: 1e-4..1e-1 (log-uniform)
        loga = float(rng.uniform(np.log10(1e-4), np.log10(1e-1)))
        lasso_alpha = float(10.0 ** loga)

        # - snr_db: 15..35
        snr_db = float(rng.uniform(15.0, 35.0))

        cfg_t = Config(
            N=cfg.N,
            K=cfg.K,
            M_train=cfg.M_train,
            seed=int(cfg.seed) + t,  # vary the seed for variety, still deterministic overall
            snr_db=snr_db,
            predicted_method="lasso",
            ridge_lambda=cfg.ridge_lambda,
            lasso_alpha=lasso_alpha,
            lasso_max_iters=cfg.lasso_max_iters,
            lasso_tol=cfg.lasso_tol,
            alpha=cfg.alpha,
            xi_patch=xi_patch,
            xi_refine_factor=cfg.xi_refine_factor,
            source_num_terms=source_num_terms,
            source_freq_range=(2, cfg.K),
            f_coef_abs_range=f_coef_abs_range,
            save_dir=cfg.save_dir,
            save_outputs=False,
            search_mode=False,
            search_max_trials=cfg.search_max_trials,
            search_print_every=cfg.search_print_every,
        )

        try:
            res = run_flow(cfg_t, save_outputs=False, verbose=False)
        except Exception:
            continue

        ratio = float(res["ratio_true_over_pred"])
        if ratio < best_ratio:
            best_ratio = ratio
            best_cfg = cfg_t
            best_res = res

        if (t + 1) % int(cfg.search_print_every) == 0:
            print(f"[trial {t+1:4d}/{cfg.search_max_trials}] best ratio so far: {best_ratio:.3f}")

        if ratio <= 0.8:
            best_cfg = cfg_t
            best_res = res
            break

    if best_cfg is None or best_res is None:
        raise RuntimeError("Search failed: no valid trials succeeded.")

    print("=" * 90)
    print("FOUND INSTANCE (true->anchor <= 0.8 * pred->anchor) in Xi coefficient metric")
    print(f"ratio={best_ratio:.3f}")
    print(f"xi_patch={best_cfg.xi_patch}")
    print(f"alpha={best_cfg.alpha} | snr_db={best_cfg.snr_db:.1f} | lasso_alpha={best_cfg.lasso_alpha:.3e}")
    print(f"source_num_terms={best_cfg.source_num_terms} | f_coef_abs_range={best_cfg.f_coef_abs_range} | seed={best_cfg.seed}")
    print(
        f"pred->anchor={float(best_res['dist_pred_anchor_xi_coef']):.6e} | "
        f"true->anchor={float(best_res['dist_true_anchor_xi_coef']):.6e}"
    )
    print("=" * 90)

    # Save winner
    winner_cfg = Config(**{**asdict(best_cfg), "save_outputs": True, "search_mode": False})
    winner_res = run_flow(winner_cfg, save_outputs=True, verbose=True)
    with open(os.path.join(cfg.save_dir, "search_winner.json"), "w") as f_json:
        json.dump({"config": asdict(best_cfg), "results": winner_res}, f_json, indent=2)
    print(f"Saved winner to: {os.path.join(cfg.save_dir, 'search_winner.json')}")
    return


if __name__ == "__main__":
    main()


