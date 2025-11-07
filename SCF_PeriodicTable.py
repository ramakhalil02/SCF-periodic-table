import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, eigh
from scipy.interpolate import BSpline

iterations = 40
eta = 0.7
gauss_n = 32
z, w = np.polynomial.legendre.leggauss(gauss_n)
r_min = 1e-6
r_max = 70
delta_r = 0.05
N_knots = 280
N = 1000
threshold = 1e-3

def gauss_integral(x, y):
    if len(x) < 2:
        return 0
    a, b = x[0], x[-1]
    avg_sum = (a + b) / 2
    avg_dif = (b - a) / 2
    integral = 0
    for i in range(len(w)):
        xi = z[i] * avg_dif + avg_sum
        yi = np.interp(xi, x, y)
        integral += avg_dif * w[i] * yi
    return integral


def radial_grid(r_min, r_max, num):
    return np.linspace(r_min, r_max, num)

def spline_knots(r, k=3):
    indices = np.linspace(0, len(r) - 1, N_knots, dtype=int)
    core = r[indices]
    ghost_left = np.repeat(r[0], k)
    ghost_right = np.repeat(r[-1], k)
    return np.concatenate([ghost_left, core, ghost_right])

def spline_basis(tknot, r, k=3):
    num_splines = len(tknot) - k - 1
    basis = []
    basis2 = []
    for i in range(1, num_splines - 1):
        c = np.zeros(num_splines)
        c[i] = 1
        s = BSpline(tknot, c, k)
        basis.append(s(r))
        basis2.append(s.derivative(2)(r))
    return basis, basis2


def print_energy_table(E_nl, P_nl, Vee, Nocc, n_values, l_values, r, label):
    print(f"\n{label: <5}  Orbital energy (Hartree)    <−V_ee/2> (Hartree)   occupation number     Total (Hartree)")
    total_energy = 0.0
    for i in range(len(Nocc)):
        l = l_values[i]
        n = n_values[i]
        energy = E_nl[-1, l, n]
        vee_term = -0.5 * np.trapz(P_nl[l, :, n] * Vee[:, -1] * P_nl[l, :, n], r)
        total = Nocc[i] * (energy + vee_term)
        print(f" {n+1}{'spdf'[l]}      {energy:>8.2f}                {vee_term:>8.2f}                 {Nocc[i]:>2}              {total:>8.2f}")
        total_energy += total
    print(f"{'':<57}Sum        {total_energy:>8.2f}\n")
    return total_energy


def run_scf(Z, Nocc, n_values, l_values, label="Atom"):
    k=3
    r = radial_grid(r_min, r_max, N)
    tknot = spline_knots(r)
    basis, basis2 = spline_basis(tknot, r)

    lamda = sorted(set(l_values))
    num_splines = len(tknot) - k - 1
    m = num_splines - 2

    Vdir = np.zeros((len(r), iterations + 1))
    Vex = np.zeros((len(r), iterations + 1))
    Vee = Vdir + Vex
    E_nl = np.zeros((iterations, len(lamda), m))
    rho = np.zeros((iterations, len(r)))
    P_nl = np.zeros((len(lamda), len(r), m))

    for it in range(iterations):
        P_nl.fill(0)
        for l in lamda:
            H = np.zeros((m, m))
            B = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    max_idx = max(i, j)
                    min_idx = min(i, j)
                    interval = (r >= tknot[max_idx]) & (r <= tknot[min_idx + k + 1])
                    H_ij = basis[j][interval] * (
                        -0.5 * basis2[i][interval]
                        + 0.5 * l * (l + 1) / r[interval]**2 * basis[i][interval]
                        - Z  / r[interval] * basis[i][interval]
                        + Vee[interval, it] * basis[i][interval]
                    )
                    B_ij = basis[j][interval] * basis[i][interval]
                    H[i, j] = gauss_integral(r[interval], H_ij)
                    B[i, j] = gauss_integral(r[interval], B_ij)

            E, c = eigh(H, B)
            E_nl[it, l] = E
            for i in range(m):
                for j in range(m):
                    P_nl[l, :, i] += c[j, i] * basis[j]

        rho[it] = 0
        for i in range(len(Nocc)):
            l, n = l_values[i], n_values[i]
            rho[it] += (1 / (4 * np.pi)) * Nocc[i] * (P_nl[l, :, n] / r) ** 2

        A = np.zeros((num_splines - 1, num_splines - 1))
        for i in range(1, num_splines):
            c = np.zeros(num_splines)
            c[i] = 1
            s = BSpline(tknot, c, k)
            d2 = s.derivative(2)
            A[i - 1, i - 1] = d2(tknot[i + 2])
            if i < num_splines - 1:
                A[i, i - 1] = d2(tknot[i + 3])
        for i in range(2, num_splines):
            c = np.zeros(num_splines)
            c[i] = 1
            s = BSpline(tknot, c, k)
            d2 = s.derivative(2)
            A[i - 2, i - 1] = d2(tknot[i + 1])
        A[-1, -1] = 30
        A[-1, -2] = -30

        rhs = -tknot[3:-2] * 4 * np.pi * np.interp(tknot[3:-2], r, rho[it])
        Xm = solve(A, rhs)
        Xm = np.insert(Xm, 0, 0)
        phi = BSpline(tknot, Xm, k)
        Vdir[:, it + 1] = phi(r) / r
        Vex[:, it + 1] = -3 * (3 * rho[it] / (8 * np.pi)) ** (1 / 3)
        Vee[:, it + 1] = (1 - eta) * (Vdir[:, it + 1] + Vex[:, it + 1]) + eta * Vee[:, it]

        total_energy = 0.0
        for i in range(len(Nocc)):
            l, n = l_values[i], n_values[i]
            energy = E_nl[it, l, n]
            vee_term = -0.5 * np.trapz(P_nl[l, :, n] * Vee[:, it + 1] * P_nl[l, :, n], r)
            total_energy += Nocc[i] * (energy + vee_term)

    total_energy = print_energy_table(E_nl, P_nl, Vee, Nocc, n_values, l_values, r, label)
    return total_energy, rho, r

def plot_density(r, rho_atom, rho_ion, label):
    plt.figure() 
    plt.plot(r, rho_atom[-1, :] * 4 * np.pi * r**2, label=f"{label}")
    plt.plot(r, rho_ion[-1, :] * 4 * np.pi * r**2, label=f"{label}+")
    plt.xlabel('r')
    plt.ylabel('$\\rho(r) \\cdot (4 \\pi r^2)$')
    plt.title(f'Electron Probability Density – {label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"electron_density_{label}.png", dpi=800)
    plt.show()
    
    
def plot_ionization_energy(elements, ionization_results):
    Z_vals, labels, computed_IE, reference_IE = zip(*[
        (e["Z"], e["label"], comp, ref)
        for e, (label, comp, ref) in zip(elements, ionization_results)
    ])
    
    computed_IE_eV = [ie * 27.2114 for ie in computed_IE]
    reference_IE_eV = [ie * 27.2114 for ie in reference_IE]

    plt.figure()
    plt.plot(Z_vals, computed_IE_eV, marker='o', markersize=3.5, ls='--', lw=0.8, color='b' ,label ='Numerical')
    plt.plot(Z_vals, reference_IE_eV, marker='o',markersize=3.5, ls='--', lw=0.8, color='r' ,label ='Theoretical')

    for Z, label, y in zip(Z_vals, labels, computed_IE_eV):
        plt.text(Z, y + 0.1, label, ha='center', fontsize=9)

    plt.xlabel("Atomic Number (Z)")
    plt.ylabel("Ionization Energy (eV)")
    plt.title("Ionization Energies vs. Atomic Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ionization_energy_comparison.png", dpi=300)
    plt.show()


def main():
    elements = [
        {"label": "He", "Z": 2,
          "Nocc_atom": [2], "nvals_atom": [0], "lvals_atom": [0],
          "Nocc_ion": [1], "nvals_ion": [0], "lvals_ion": [0],
          "theoretical_IE": 0.9036},

        {"label": "Ne", "Z": 10,
          "Nocc_atom": [2, 2, 6], "nvals_atom": [0, 1, 0], "lvals_atom": [0, 0, 1],
          "Nocc_ion": [2, 2, 5], "nvals_ion": [0, 1, 0], "lvals_ion": [0, 0, 1],
          "theoretical_IE": 0.7925},

        {"label": "Na", "Z": 11,
          "Nocc_atom": [2, 2, 6, 1], "nvals_atom": [0, 1, 0, 2], "lvals_atom": [0, 0, 1, 0],
          "Nocc_ion": [2, 2, 6], "nvals_ion": [0, 1, 0], "lvals_ion": [0, 0, 1],
          "theoretical_IE": 0.1888},

        {"label": "K", "Z": 19,
          "Nocc_atom": [2, 2, 6, 2, 6, 1], "nvals_atom": [0, 1, 0, 2, 1, 3], "lvals_atom": [0, 0, 1, 0, 1, 0],
          "Nocc_ion": [2, 2, 6, 2, 6], "nvals_ion": [0, 1, 0, 2, 1], "lvals_ion": [0, 0, 1, 0, 1],
          "theoretical_IE": 0.1595},

        {"label": "Zn", "Z": 30,
          "Nocc_atom": [2, 2, 6, 2, 6, 2, 10], "nvals_atom": [0, 1, 0, 2, 1, 3, 0], "lvals_atom": [0, 0, 1, 0, 1, 0, 2],
          "Nocc_ion": [2, 2, 6, 2, 6, 2, 9], "nvals_ion": [0, 1, 0, 2, 1, 3, 0], "lvals_ion": [0, 0, 1, 0, 1, 0, 2],
          "theoretical_IE": 0.345},

        {"label": "Ga", "Z": 31,
          "Nocc_atom": [2, 2, 6, 2, 6, 2, 10, 1], "nvals_atom": [0, 1, 0, 2, 1, 3, 0, 2], "lvals_atom": [0, 0, 1, 0, 1, 0, 2, 1],
          "Nocc_ion": [2, 2, 6, 2, 6, 2, 10], "nvals_ion": [0, 1, 0, 2, 1, 3, 0], "lvals_ion": [0, 0, 1, 0, 1, 0, 2],
          "theoretical_IE": 0.220},

        {"label": "Kr", "Z": 36,
          "Nocc_atom": [2, 2, 6, 2, 6, 2, 10, 6], "nvals_atom": [0, 1, 0, 2, 1, 3, 0, 2], "lvals_atom": [0, 0, 1, 0, 1, 0, 2, 1],
          "Nocc_ion": [2, 2, 6, 2, 6, 2, 10, 5], "nvals_ion": [0, 1, 0, 2, 1, 3, 0, 2], "lvals_ion": [0, 0, 1, 0, 1, 0, 2, 1],
          "theoretical_IE": 0.515},

        {"label": "Rb", "Z": 37,
          "Nocc_atom": [2, 2, 6, 2, 6, 2, 10, 6, 1], "nvals_atom": [0, 1, 0, 2, 1, 3, 0, 2, 4], "lvals_atom": [0, 0, 1, 0, 1, 0, 2, 1, 0],
          "Nocc_ion": [2, 2, 6, 2, 6, 2, 10, 6], "nvals_ion": [0, 1, 0, 2, 1, 3, 0, 2], "lvals_ion": [0, 0, 1, 0, 1, 0, 2, 1],
          "theoretical_IE": 0.154},

        {"label": "Xe", "Z": 54,
          "Nocc_atom": [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6], "nvals_atom": [0, 1, 0, 2, 1, 3, 0, 2, 4, 1, 3], "lvals_atom": [0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1],
          "Nocc_ion": [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5], "nvals_ion": [0, 1, 0, 2, 1, 3, 0, 2, 4, 1, 3], "lvals_ion": [0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1],
          "theoretical_IE": 0.446},

        {"label": "Cs", "Z": 55,
          "Nocc_atom": [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1], "nvals_atom": [0, 1, 0, 2, 1, 3, 0, 2, 4, 1, 3, 5], "lvals_atom": [0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0],
          "Nocc_ion": [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6], "nvals_ion": [0, 1, 0, 2, 1, 3, 0, 2, 4, 1, 3], "lvals_ion": [0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1],
          "theoretical_IE": 0.1431}
    ]

    # # Potassium in two configurations:
    # elements = [
    #     # Ground state configuration: K atom (4s¹) and K+ ion 
    #     {"label": "K_ground", "Z": 19,
    #      "Nocc_atom": [2, 2, 6, 2, 6, 1],
    #      "nvals_atom": [0, 1, 0, 2, 1, 3],
    #      "lvals_atom": [0, 0, 1, 0, 1, 0],
    #      "Nocc_ion": [2, 2, 6, 2, 6],
    #      "nvals_ion": [0, 1, 0, 2, 1],
    #      "lvals_ion": [0, 0, 1, 0, 1],
    #      "theoretical_IE": 0.1595},

    #     # K atom with last e⁻ in 3d and K+ ion 
    #     {"label": "K_ion_3d", "Z": 19,
    #      "Nocc_atom": [2, 2, 6, 2, 6, 1],
    #      "nvals_atom": [0, 1, 0, 2, 1, 0],
    #      "lvals_atom": [0, 0, 1, 0, 1, 2],
    #      "Nocc_ion": [2, 2, 6, 2, 6],
    #      "nvals_ion": [0, 1, 0, 2, 1],
    #      "lvals_ion": [0, 0, 1, 0, 1],
    #      "theoretical_IE": None}  # no theoretical reference
    #]

    ionization_results = []

    for elem in elements:
        print(f"Running SCF for {elem['label']}...")
        E_atom, rho_atom, r = run_scf(elem['Z'], elem['Nocc_atom'], elem['nvals_atom'], elem['lvals_atom'], label=elem['label'])
        E_ion, rho_ion, _ = run_scf(elem['Z'], elem['Nocc_ion'], elem['nvals_ion'], elem['lvals_ion'], label=elem['label'] + '+')
        IE_numerical = E_ion - E_atom
        print(f"Estimated Ionization Energy for {elem['label']}: {IE_numerical:.4f} Hartree\n")
        ionization_results.append((elem['label'], IE_numerical, elem['theoretical_IE']))
        #plot_density(r, rho_atom, rho_ion, elem['label'])

    plot_ionization_energy(elements, ionization_results)

if __name__ == "__main__":
    main()


