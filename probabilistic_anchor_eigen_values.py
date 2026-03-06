import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from spherical_harmonic_experiment import (
    calc_gram_matrix_on_sphere,
    create_spherical_harmonics_functions,
)


def build_g_xi_tilde(
    deg: int,
    omega_size: float = np.pi / 4,
    domain_xi_phi=(0.0, np.pi / 2),
    n_quad: int = 500,
) -> np.ndarray:
    """
    Construct G_xi_tilde as in calc_kappa_spherical_harmonics: orthonormalize the
    basis on Omega then express the Xi Gram matrix in that basis.
    """
    basis_functions = create_spherical_harmonics_functions(deg)

    domain_omega_phi = (np.pi - omega_size, np.pi)

    G_omega = calc_gram_matrix_on_sphere(
        basis_functions=basis_functions,
        domain_phi=domain_omega_phi,
        n_quad=n_quad,
    )
    G_xi = calc_gram_matrix_on_sphere(
        basis_functions=basis_functions,
        domain_phi=domain_xi_phi,
        n_quad=n_quad,
    )

    # Symmetrize to reduce tiny numerical asymmetries
    G_omega = 0.5 * (G_omega + G_omega.T)
    G_xi = 0.5 * (G_xi + G_xi.T)

    # Orthonormalize on Omega: find T so that T G_omega T^T = I
    eigvals, eigvecs = np.linalg.eigh(G_omega)
    max_ev = np.max(np.abs(eigvals))
    eps = 1e-12 * max_ev if max_ev > 0 else 1e-12
    eigvals_clipped = np.maximum(eigvals, eps)
    T = np.diag(1.0 / np.sqrt(eigvals_clipped)) @ eigvecs.T  # shape (d,d)

    # Express Xi Gram in the Omega-orthonormal basis
    G_xi_tilde = T @ G_xi @ T.T
    G_xi_tilde = 0.5 * (G_xi_tilde + G_xi_tilde.T)
    return G_xi_tilde


def sample_uniform_unit_ball(dim: int) -> np.ndarray:
    """
    Sample a vector uniformly from the unit L2 ball in R^dim.
    """
    v = np.random.normal(size=dim)
    norm = np.linalg.norm(v)
    if norm == 0:
        return sample_uniform_unit_ball(dim)
    u = v / norm
    r = np.random.random() ** (1.0 / dim)
    return r * u


def probabilistic_kappa(eig_max: float, quantile: float, dim: int) -> float:
    """
    Return the theoretical kappa quantile under the Beta model:
        Z ~ Beta(1/2, (d-1)/2), kappa = eig_max * Z.
    """
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must be in (0, 1].")
    if dim < 2:
        raise ValueError("dim must be >= 2 for the Beta model.")
    a = 0.5
    b = (dim - 1) / 2.0
    return eig_max * beta.ppf(quantile, a, b)


def run_probabilistic_eigs(
    deg: int = 9,
    omega_size: float = np.pi / 4,
    domain_xi_phi=(0.0, np.pi / 2),
    n_quad: int = 500,
    num_samples: int = 10000,
    save_dir: str = "./results",
    overlay_beta: bool = True,
    beta_num_points: int = 600,
):
    G_xi_tilde = build_g_xi_tilde(
        deg=deg,
        omega_size=omega_size,
        domain_xi_phi=domain_xi_phi,
        n_quad=n_quad,
    )
    d = G_xi_tilde.shape[0]

    eigvals = np.linalg.eigvalsh(G_xi_tilde)
    print("Eigenvalues:", eigvals)
    eig_max = float(np.max(eigvals))
    eig_mean = float(np.mean(eigvals))

    values = []
    for _ in range(num_samples):
        c = sample_uniform_unit_ball(d)
        c_norm_sq = float(np.linalg.norm(c) ** 2)
        val = float(c @ G_xi_tilde @ c) / c_norm_sq
        values.append(val)

    values = np.array(values)

    # Theoretical (sphere model): s ~ Unif(S^{d-1}), Z = s_1^2 ~ Beta(1/2, (d-1)/2)
    # kappa = lambda_max * Z when normalizing by ||c||^2.
    beta_scale = eig_max
    beta_x = np.linspace(0.0, beta_scale, beta_num_points)
    beta_pdf = beta.pdf(beta_x / beta_scale, 0.5, (d - 1) / 2.0) / beta_scale if overlay_beta else None
    q50 = probabilistic_kappa(eig_max=eig_max, quantile=0.5, dim=d)
    q90 = probabilistic_kappa(eig_max=eig_max, quantile=0.9, dim=d)
    q100 = probabilistic_kappa(eig_max=eig_max, quantile=1.0, dim=d)

    print(
        f"Samples of c^T G_xi_tilde c over unit ball (deg={deg}, n={num_samples}): "
        f"\nmin={values.min():.4e}, max={values.max():.4e}, "
        f"\nmean={values.mean():.4e}, std={values.std():.4e}, "
        f"\neig_max={eig_max:.4e}, eig_mean={eig_mean:.4e}"
        f"\nBelow max: {(values <= eig_max).mean()}"
        f"\nBelow mean: {(values <= eig_mean).mean()}"
        f"\nBeta median (normalized): {q50:.4e}"
        f"\nBeta 90% quantile (normalized): {q90:.4e}"
        f"\nBeta supremum (normalized): {q100:.4e}"
        f"\nBelow beta median: {(values <= q50).mean():.4f}"
        f"\nBelow beta 90%: {(values <= q90).mean():.4f}"
        f"\nBelow beta supremum: {(values <= q100).mean():.4f}"
    )

    # Plot histogram/PDF
    fig_pdf, ax_pdf = plt.subplots(1, 1, figsize=(6, 4))
    ax_pdf.hist(values, bins=30, color="tab:blue", alpha=0.65, edgecolor="black", density=True)
    ax_pdf.axvline(eig_max, color="tab:red", linestyle="--", label=f"max eig={eig_max:.3e}")
    # ax_pdf.axvline(eig_mean, color="tab:green", linestyle=":", label=f"mean eig={eig_mean:.3e}")
    ax_pdf.axvline(q50, color="tab:purple", linestyle="-.", label=f"beta 50%={q50:.3e}")
    ax_pdf.axvline(q90, color="tab:brown", linestyle="--", label=f"beta 90%={q90:.3e}")

    if overlay_beta and beta_pdf is not None:
        ax_pdf.plot(beta_x, beta_pdf, color="tab:purple", linewidth=2, label="Beta theory (||c||=1)")
    ax_pdf.set_title(r"PDF of $c^T G_{\Xi}^\sim c / \|c\|_2^2$")
    ax_pdf.set_xlabel("value")
    ax_pdf.set_ylabel("density")
    ax_pdf.legend()
    fig_pdf.tight_layout()

    # Plot sorted values
    fig_sorted, ax_sorted = plt.subplots(1, 1, figsize=(6, 4))
    sorted_vals = np.sort(values)
    ax_sorted.plot(sorted_vals, marker=".", linestyle="none", color="tab:orange")
    ax_sorted.axhline(eig_max, color="tab:red", linestyle="--", label="max eig")
    # ax_sorted.axhline(eig_mean, color="tab:green", linestyle=":", label="mean eig")
    ax_sorted.axhline(q50, color="tab:purple", linestyle="-.", label="beta 50%")
    ax_sorted.axhline(q90, color="tab:brown", linestyle="--", label="beta 90%")

    ax_sorted.set_title("Sorted samples")
    ax_sorted.set_xlabel("sample index (sorted)")
    ax_sorted.set_ylabel("value")
    ax_sorted.legend()
    fig_sorted.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, "probabilistic_eigs.png")
        sorted_path = os.path.join(save_dir, "probabilistic_eigs_sorted.png")
        fig_pdf.savefig(pdf_path, dpi=150)
        fig_sorted.savefig(sorted_path, dpi=150)
        print(f"Saved plot to {pdf_path}")
        print(f"Saved sorted plot to {sorted_path}")
    else:
        plt.show()


if __name__ == "__main__":
    run_probabilistic_eigs()

