"""Monolithic beam: heterogeneous vs homogeneous solution.

Standalone validation script — no pytest, no argparse.
Run directly:   python monolithic_default.py
Output:         monolithic_default.csv  (next to this script)

Physical setup
--------------
- Bar of length L = 1 m, clamped at both ends.
- Periodic microstructure: RVE of period 0.01 m, 50 % volume fraction.
- Phase 1: E1 = 200 000 MPa  /  Phase 2: E2 = 20 000 MPa
- Uniform load q = -1000 N/m

Output scalars
--------------
- u_max_heter : max |u| for the heterogeneous FE solution
- u_max_homo  : max |u| for the homogenized FE solution
"""
from pathlib import Path

import numpy as np

from pybeam.dirichlet_bc import apply_dirichlet_boundary_conditions
from pybeam.fe_operator import compute_stiffness_matrix, volumetric_load_vector
from pybeam.mesh import create_mesh_linear

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BEAM_LENGTH   = 1.0
RVE_LENGTH    = 0.01
RVE_FV        = 0.5
YOUNG_1       = 200_000.0
YOUNG_2       = 20_000.0
N_ELEM_BY_RVE = 10
LOAD          = -1000.0

E_HOMO = 1.0 / (RVE_FV / YOUNG_1 + (1.0 - RVE_FV) / YOUNG_2)

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
nb_elem = int(BEAM_LENGTH / RVE_LENGTH * N_ELEM_BY_RVE)
mesh = create_mesh_linear(0, BEAM_LENGTH, nb_elem)


def load_f(x):
    return LOAD


fixed_dof = [0, -1]

# ---------------------------------------------------------------------------
# Heterogeneous solution
# ---------------------------------------------------------------------------
young_heter = []
for i in range(mesh["nb_elements"]):
    elem_rank_in_rve = i % N_ELEM_BY_RVE
    young_heter.append(YOUNG_1 if elem_rank_in_rve < int(RVE_FV * N_ELEM_BY_RVE) else YOUNG_2)

K = compute_stiffness_matrix(mesh, young_heter)
f = volumetric_load_vector(mesh, load_f)
Kc, fc = apply_dirichlet_boundary_conditions(K, f, fixed_dofs=fixed_dof)
u_heter = np.linalg.solve(Kc, fc)[: -len(fixed_dof)].flatten()

# ---------------------------------------------------------------------------
# Homogeneous solution
# ---------------------------------------------------------------------------
K = compute_stiffness_matrix(mesh, [E_HOMO] * mesh["nb_elements"])
f = volumetric_load_vector(mesh, load_f)
Kc, fc = apply_dirichlet_boundary_conditions(K, f, fixed_dofs=fixed_dof)
u_homo = np.linalg.solve(Kc, fc)[: -len(fixed_dof)].flatten()

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
out = Path(__file__).with_suffix(".csv")
np.savetxt(
    out,
    np.array([[float(np.max(np.abs(u_heter))), float(np.max(np.abs(u_homo)))]]),
    delimiter=",",
    header="u_max_heter,u_max_homo",
    comments="",
)
print(f"[monolithic] wrote {out}")
