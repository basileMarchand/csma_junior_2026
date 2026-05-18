"""Parametric Arlequin run for a single RVE size.

Called by Snakemake for each RVE size in the convergence study.
All physical parameters are fixed here except rve_length, which is
passed on the command line.

Usage:
    python -m workflows.run_parametric --rve-length 0.01 --output results/parametric/error_0.01.csv
"""

import argparse
from pathlib import Path

import numpy as np

from pybeam.mesh import create_mesh_linear
from pybeam.fe_operator import (
    compute_stiffness_matrix,
    volumetric_load_vector,
    compute_l2_operator,
)
from pybeam.dirichlet_bc import apply_dirichlet_boundary_conditions
from pybeam.arlequin import compute_coupling, build_full_system

# ---------------------------------------------------------------------------
# Fixed physical parameters
# ---------------------------------------------------------------------------
BEAM_LENGTH       = 1.0
RVE_FV            = 0.5
YOUNG_1           = 200_000.0
YOUNG_2           = 10_000.0     # Arlequin canonical value
N_ELEM_BY_RVE     = 10
X_START_COUPLING  = 0.3
X_END_COUPLING    = 0.5
H1_COEFF          = 1.0
REFINEMENT_FACTOR = 2.0
LOAD              = -1000.0


def _young_homo(young_1, young_2, fv):
    return 1.0 / (fv / young_1 + (1.0 - fv) / young_2)


def run(rve_length: float, output: Path) -> None:
    """Solve the Arlequin problem for the given RVE size and write the error.

    Args:
        rve_length: RVE period in meters.
        output: path for the output CSV file.
    """
    E_homo = _young_homo(YOUNG_1, YOUNG_2, RVE_FV)
    nb_elem = int(BEAM_LENGTH / rve_length * N_ELEM_BY_RVE)
    elem_size = BEAM_LENGTH / nb_elem

    x_start = X_START_COUPLING
    x_end   = X_END_COUPLING

    mesh1 = create_mesh_linear(0, x_end, int(x_end / elem_size))
    mesh2 = create_mesh_linear(
        x_start, BEAM_LENGTH,
        int((BEAM_LENGTH - x_start) / (elem_size * REFINEMENT_FACTOR)),
    )

    def alpha1(x):
        if x < x_start:            return 0.999
        elif x_start <= x <= x_end: return 0.5
        return 0.001

    def alpha2(x):
        return 1.0 - alpha1(x)

    def load_f(x):
        return LOAD

    # --- Model 1: heterogeneous ---
    young1 = []
    for elem in mesh1["elements"]:
        xs = mesh1["coords"][elem]
        e_center = (xs[0] + xs[1]) / 2.0
        nb_rve  = int(e_center / rve_length)
        loc_pos = e_center - nb_rve * rve_length
        young1.append(YOUNG_1 if loc_pos < RVE_FV * rve_length else YOUNG_2)

    K1 = compute_stiffness_matrix(mesh1, young1, alpha_f=alpha1)
    f1 = volumetric_load_vector(mesh1, load_f, alpha_f=alpha1)
    Kc1, fc1 = apply_dirichlet_boundary_conditions(K1, f1, fixed_dofs=[0])

    # --- Model 2: homogeneous ---
    K2 = compute_stiffness_matrix(mesh2, [E_homo] * mesh2["nb_elements"], alpha_f=alpha2)
    f2 = volumetric_load_vector(mesh2, load_f, alpha_f=alpha2)
    Kc2, fc2 = apply_dirichlet_boundary_conditions(K2, f2, fixed_dofs=[-1])

    # --- Coupling ---
    lag_h    = np.max([mesh1["dx"].min(), mesh2["dx"].min()])
    lag_mesh = create_mesh_linear(x_start, x_end, int((x_end - x_start) / lag_h))
    C1 = compute_coupling(lag_mesh, mesh1, param_h1=H1_COEFF)
    C2 = compute_coupling(lag_mesh, mesh2, param_h1=H1_COEFF)

    lhs, rhs = build_full_system(Kc1, Kc2, fc1, fc2, C1, C2)
    sol = np.linalg.solve(lhs, rhs)
    u1 = sol[: mesh1["nb_nodes"]].flatten()

    # --- Reference (fine heterogeneous monolithic) ---
    nb_ref   = int(BEAM_LENGTH / rve_length * N_ELEM_BY_RVE)
    mesh_ref = create_mesh_linear(0, BEAM_LENGTH, nb_ref)
    young_ref = []
    for elem in mesh_ref["elements"]:
        xs = mesh_ref["coords"][elem]
        e_center = (xs[0] + xs[1]) / 2.0
        nb_rve  = int(e_center / rve_length)
        loc_pos = e_center - nb_rve * rve_length
        young_ref.append(YOUNG_1 if loc_pos < RVE_FV * rve_length else YOUNG_2)

    K_ref = compute_stiffness_matrix(mesh_ref, young_ref)
    f_ref = volumetric_load_vector(mesh_ref, load_f)
    Kc_r, fc_r = apply_dirichlet_boundary_conditions(K_ref, f_ref, fixed_dofs=[0, -1])
    u_ref = np.linalg.solve(Kc_r, fc_r)[:-2].flatten()

    # --- L2 error on ZOI [0, 0.2] ---
    zoi  = 0.2
    mask = mesh1["coords"] <= zoi
    delta = u_ref[: u1.shape[0]][mask] - u1[mask]
    l2_op     = compute_l2_operator(mesh1)
    l2_op_zoi = l2_op[np.ix_(mask, mask)]
    error     = float(
        np.sqrt((delta.reshape(1, -1) @ l2_op_zoi @ delta.reshape(-1, 1)).item()) / x_end
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        np.array([[rve_length, error]]),
        delimiter=",",
        header="rve_length,arlequin_l2_error",
        comments="",
    )
    print(f"[parametric] rve_length={rve_length:.4f}  error={error:.6e}  -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rve-length", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.rve_length, args.output)
