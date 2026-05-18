"""Arlequin homo-homo with incompatible meshes vs analytical solution.

Standalone validation script — no pytest, no argparse.
Run directly:   python analytical_arlequin_homo_default.py
Output:         analytical_arlequin_homo_default.csv  (next to this script)

Physical setup
--------------
Strong form:  -E·u'' = q,  u(0) = u(L) = 0
Exact:        u(x) = q·x·(L-x) / (2·E)

Both Arlequin models share the same Young's modulus E.
The method should recover the analytical solution up to discretisation error.

Mesh incompatibility (intentional)
-----------------------------------
- Model 1: [0, 0.5], 10 elements   → h1 = 0.05  m
- Model 2: [0.3, 1], 21 elements   → h2 ≈ 0.033 m  (INCOMPATIBLE in overlap)
- Overlap: [0.3, 0.5], H1 coupling (h1_coeff = 1).
- Lagrange multiplier mesh: coarser scale, 4 elements.

Output scalars
--------------
- l2_error_zoi  : consistent-mass L2 norm of (u_model1 - u_exact) on ZOI [0, x_start]
- l2_relative   : l2_error_zoi / L2-norm of u_exact on the same ZOI
"""
from pathlib import Path

import numpy as np

from pybeam.arlequin import build_full_system, compute_coupling
from pybeam.dirichlet_bc import apply_dirichlet_boundary_conditions
from pybeam.fe_operator import (
    compute_l2_operator,
    compute_stiffness_matrix,
    volumetric_load_vector,
)
from pybeam.mesh import create_mesh_linear

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
YOUNG       = 200_000.0
LOAD        = -1_000.0
BEAM_LENGTH = 1.0
X_START     = 0.3     # left edge of overlap (= right edge of ZOI)
X_END       = 0.5     # right edge of model 1
NB_ELEM_1   = 10      # model 1: [0, 0.5]   → h1 = 0.05
NB_ELEM_2   = 21      # model 2: [0.3, 1.]  → h2 ≈ 0.033
H1_COEFF    = 1.0


def load_f(x):
    return LOAD


def alpha1(x):
    if x < X_START:              return 0.999
    elif X_START <= x <= X_END:  return 0.5
    return 0.001


def alpha2(x):
    return 1.0 - alpha1(x)


# ---------------------------------------------------------------------------
# Model 1: homogeneous [0, X_END]
# ---------------------------------------------------------------------------
mesh1 = create_mesh_linear(0.0, X_END, NB_ELEM_1)
K1 = compute_stiffness_matrix(mesh1, [YOUNG] * mesh1["nb_elements"], alpha_f=alpha1)
f1 = volumetric_load_vector(mesh1, load_f, alpha_f=alpha1)
Kc1, fc1 = apply_dirichlet_boundary_conditions(K1, f1, fixed_dofs=[0])

# ---------------------------------------------------------------------------
# Model 2: homogeneous [X_START, BEAM_LENGTH]
# ---------------------------------------------------------------------------
mesh2 = create_mesh_linear(X_START, BEAM_LENGTH, NB_ELEM_2)
K2 = compute_stiffness_matrix(mesh2, [YOUNG] * mesh2["nb_elements"], alpha_f=alpha2)
f2 = volumetric_load_vector(mesh2, load_f, alpha_f=alpha2)
Kc2, fc2 = apply_dirichlet_boundary_conditions(K2, f2, fixed_dofs=[-1])

# ---------------------------------------------------------------------------
# Lagrange multiplier coupling (coarser scale)
# ---------------------------------------------------------------------------
lag_h    = np.max([mesh1["dx"].min(), mesh2["dx"].min()])
lag_mesh = create_mesh_linear(X_START, X_END, int((X_END - X_START) / lag_h))
C1 = compute_coupling(lag_mesh, mesh1, param_h1=H1_COEFF)
C2 = compute_coupling(lag_mesh, mesh2, param_h1=H1_COEFF)

lhs, rhs = build_full_system(Kc1, Kc2, fc1, fc2, C1, C2)
sol      = np.linalg.solve(lhs, rhs)
u_model1 = sol[: mesh1["nb_nodes"]].flatten()

# ---------------------------------------------------------------------------
# Analytical solution on model 1 nodes
# ---------------------------------------------------------------------------
coords1 = mesh1["coords"]
u_exact = LOAD * coords1 * (BEAM_LENGTH - coords1) / (2.0 * YOUNG)

# ---------------------------------------------------------------------------
# Consistent-mass L2 error on ZOI [0, X_START]
# ---------------------------------------------------------------------------
mask     = coords1 <= X_START
delta    = u_model1[mask] - u_exact[mask]
l2_op    = compute_l2_operator(mesh1)
l2_zoi   = l2_op[np.ix_(mask, mask)]

l2_error = float(np.sqrt((delta.reshape(1, -1) @ l2_zoi @ delta.reshape(-1, 1)).item()))
l2_norm  = float(np.sqrt((u_exact[mask].reshape(1, -1) @ l2_zoi @ u_exact[mask].reshape(-1, 1)).item()))
l2_rel   = l2_error / l2_norm

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
out = Path(__file__).with_suffix(".csv")
np.savetxt(
    out,
    np.array([[l2_error, l2_rel]]),
    delimiter=",",
    header="l2_error_zoi,l2_relative",
    comments="",
)
print(f"[arlequin_homo] l2_error={l2_error:.6e}  l2_rel={l2_rel:.3e}  wrote {out}")
