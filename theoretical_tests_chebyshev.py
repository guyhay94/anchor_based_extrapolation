import numpy as np
import matplotlib.pyplot as plt

from cheby_utils import create_chebyshev_basis_functions_numpy
from utils import calc_extrapolation_condition_number, \
    convert_basis_function_orthogonality_domain, \
    calc_inner_extrapolation_condition_number


def plot_kappa_comparison_and_inner(num_basis: int, b_values: np.ndarray) -> None:
    kappa_1_list = []
    kappa_2_list = []
    valid_b_values_kappa_2 = []

    basis_functions = create_chebyshev_basis_functions_numpy(num_basis)
    for b in b_values:
        basis_functions_omega = convert_basis_function_orthogonality_domain(
            basis_functions,
            basis_orthogonality_a=-1,
            basis_orthogonality_b=1,
            a_1=-1,
            b_1=b
        )

        kappa_1 = calc_extrapolation_condition_number(
            phi_list=basis_functions_omega,
            domain_omega=(-1, b),
            domain_xi=(b, 1)
        )
        kappa_2 = calc_inner_extrapolation_condition_number(
            phi_list=basis_functions,
            domain_xi=(b, 1)
        )

        kappa_1_list.append(kappa_1)
        if kappa_2 is not None:
            kappa_2_list.append(kappa_2)
            valid_b_values_kappa_2.append(b)

    plt.figure(figsize=(8, 5))
    plt.plot(b_values, kappa_1_list, label=r'$\kappa$', marker='+', color="green")
    plt.plot(valid_b_values_kappa_2, kappa_2_list, label=r'$\kappa_{r}$', marker='s', linestyle='--', color="blue")

    plt.yscale("log")
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fr"./results/kappa_comparison_{num_basis}_vs_inner_cheby")




if __name__ == "__main__":
    # b values to test (you can increase the resolution if needed)
    b_vals = np.linspace(0.95, 1.0, 50)

    print("starting inner vs reg")
    plot_kappa_comparison_and_inner(num_basis=15, b_values=b_vals)
    plot_kappa_comparison_and_inner(num_basis=10, b_values=b_vals)
    plot_kappa_comparison_and_inner(num_basis=5, b_values=b_vals)

