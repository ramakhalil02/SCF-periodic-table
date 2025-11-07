# Self-Consistent Mean-field Model for the Periodic Table

This project is from independent research and development conducted during my **Master of Science program in Computational Physics**. The entire methodology, analysis, and implementation of all numerical solvers were executed solely by me, and the full work is documented in the accompanying report and code.

---

## Table of Contents

* [About The Project](#about-the-project)
* [Core Objectives](#core-objectives)
* [Languages and Libraries](#languages-and-libraries)
* [Methods Implemented](#methods-implemented)
* [Key Findings](#key-findings)
* [Getting Started](#getting-started)
* [Full Project Report](#full-project-report)
* [Contact](#contact)

---

## About The Project

This work implements a **Self-Consistent Field (SCF) Mean-field Model** to solve for the electronic structure of multi-electron atomic systems. By first iteratively solving the coupled **radial Schrödinger and Poisson equations** using B-spline basis functions, the goal is to compute the electron density, total energy, and finally, the **ionization energy** for various elements.

The model demonstrates how a mean-field approach can effectively predict and explain the fundamental periodic trends of atomic structure across the table.

## Core Objectives

1.  **Implement SCF Iteration:** Develop a robust self-consistent loop to iteratively refine the mean-field potential until convergence is reached.

2.  **Calculate Atomic Properties:** Compute the total electron density and total energy for neutral and ionized atoms (e.g., Neon and Potassium).

3.  **Verify Ground State Configuration:** Use the calculated total energies to determine and verify the actual ground state configuration (e.g., confirming $4s^1$ over $3d^1$ for Potassium).

4.  **Map Periodic Trends:** Calculate the ionization energy ($\Delta E$) as a function of atomic number ($Z$) to compare numerical results against known periodic trends.

---

## Languages and Libraries

| Category | Tools & Libraries | Competency Demonstrated |
| :--- | :--- | :--- |
| **Language** | Python | Efficient development and handling of iterative numerical schemes. |
| **Numerical** | NumPy, SciPy | Advanced matrix assembly, linear algebra for solving generalized eigenvalue problems, and integration. |
| **Visualization** | Matplotlib | Generating high-quality plots comparing theoretical and numerical ionization energies across the periodic table. |

---

## Methods Implemented

The computational core of the project implements the following techniques:

| Method | Role in Project | Key Implementation Detail |
| :--- | :--- | :--- |
| **Self-Consistent Field (SCF)** | Core simulation loop. | Iteratively updates the electron density, which defines the mean-field potential, and recalculates the wave functions until the potential stabilizes. |
| **B-spline Collocation** | Discretization method. | Utilized to transform the coupled radial Schrödinger and Poisson equations into systems of linear equations. |
| **Mean-field Potential ($V_{ee}$)** | Modeling electron interaction. | Includes both the **Direct (Coulomb) potential** and the **Exchange potential** (local approximation) to simulate electron-electron repulsion. |
| **Ionization Energy Calculation** | Analysis. | Computed as the difference in the total energy ($\Delta E$) between the neutral atom and its corresponding single-ionized state. |

## Key Findings

* **Configuration Stability:** The SCF method successfully verified the ground state electronic configuration for Alkali metals like Potassium, confirming that the $4s^1$ state is more stable (lower energy) than the excited $3d^1$ state.
* **Periodic Structure Capture:** The numerical calculation of ionization energies as a function of $Z$ correctly captured the overall periodic structure (e.g., high ionization energy for Noble Gases like Neon, low for Alkali Metals).
* **Model Limitations:** Deviations between numerical and theoretical ionization energies were observed for atoms with complex, partially filled outer shells. This discrepancy is primarily attributed to the limitations of the **local exchange approximation** used in the model.

---

## Getting Started

### Execution

To run the simulation and generate the results and visualizations, execute the core solver script:

```bash
python SCF_PeriodicTable.py
```
---

## Full Project Report

For a complete breakdown of the theoretical derivations and full numerical results (including the plots), please see the final project report:

[**Full SCF Project Report (PDF)**](SCF_PeriodicTable.pdf)

---

## Contact

I'm happy to hear your feedback or answer any questions about this project!

**Author** Rama Khalil

**Email**  rama.khalil.990@gmail.com
