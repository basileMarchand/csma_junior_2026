"""Analytical validation: homogeneous bar under uniform traction.

Standalone validation script — no pytest, no argparse.
Run directly:   python analytical_traction_default.py
Output:         analytical_traction_default.csv  (next to this script)

Physical setup
--------------
Strong form:  -E·u'' = q,  u(0) = u(L) = 0
Exact:         u(x) = q·x·(L-x) / (2·E)

Superconvergence property
-------------------------
For P1 elements on a uniform mesh with 2-point Gauss quadrature,
the FEM nodal values are *exact* for this quadratic solution.
The FEM error is orthogonal to the P1 space; the load integral is
integrated exactly by Gauss-2. Nodal residual ≈ LAPACK round-off (~1e-13).

Output columns (one row per node, 101 rows)
-------------------------------------------
- x       : node coordinate
- u_fem   : FEM displacement
- u_exact : analytical displacement  q·x·(L-x) / (2·E)
"""
from pathlib import Path

import numpy as np

from pybeam.dirichlet_bc import apply_dirichlet_boundary_conditions
from pybeam.fe_operator import compute_stiffness_matrix, volumetric_load_vector
from pybeam.mesh import create_mesh_linear

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
YOUNG       = 200_000.0
LOAD        = -1_000.0
BEAM_LENGTH = 1.0
NB_ELEM     = 100

# ---------------------------------------------------------------------------
# FEM solve
# ---------------------------------------------------------------------------
mesh = create_mesh_linear(0.0, BEAM_LENGTH, NB_ELEM)
K    = compute_stiffness_matrix(mesh, [YOUNG] * mesh["nb_elements"])
f    = volumetric_load_vector(mesh, lambda x: LOAD)
Kc, fc = apply_dirichlet_boundary_conditions(K, f, fixed_dofs=[0, -1])
u_fem = np.linalg.solve(Kc, fc)[:-2].flatten()

# ---------------------------------------------------------------------------
# Analytical solution (vectorised)
# ---------------------------------------------------------------------------
x       = mesh["coords"]
u_exact = LOAD * x * (BEAM_LENGTH - x) / (2.0 * YOUNG)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
out = Path(__file__).with_suffix(".csv")
np.savetxt(
    out,
    np.column_stack([x, u_fem, u_exact]),
    delimiter=",",
    header="x,u_fem,u_exact",
    comments="",
)
print(f"[analytical_traction] u_max_fem={np.max(np.abs(u_fem)):.6e}  wrote {out}")
