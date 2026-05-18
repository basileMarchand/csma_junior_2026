"""VER homogenization: periodic BVP → effective Young's modulus.

Standalone validation script — no pytest, no argparse.
Run directly:   python homogenization_default.py
Output:         homogenization_default.csv  (next to this script)

Physical setup
--------------
- RVE of period 0.01 m with symmetric phase distribution (phase 1 split at
  both ends of the RVE, phase 2 in the center).
- Phase 1: E1 = 200 000 MPa  /  Phase 2: E2 = 20 000 MPa
- Volume fraction: 50 % (continuous); ~40 % effective on a 10-element mesh
  due to discrete phase boundaries.

Output scalars
--------------
- young_effective : homogenized Young's modulus computed from the VER
                    periodic BVP via the integral of localization tensor B.
"""
from pathlib import Path

import numpy as np

from pybeam.dirichlet_bc import apply_periodic_conditions
from pybeam.fe_operator import (
    compute_strain_and_stress,
    compute_stiffness_matrix,
    integrate,
    periodic_load_vector,
)
from pybeam.mesh import create_mesh_linear

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RVE_LENGTH    = 0.01
RVE_FV        = 0.5
YOUNG_1       = 200_000.0
YOUNG_2       = 20_000.0
N_ELEM_BY_RVE = 10

# ---------------------------------------------------------------------------
# Phase assignment on the VER mesh (symmetric split, phase 1 at boundaries)
# ---------------------------------------------------------------------------
mesh_rve = create_mesh_linear(0, RVE_LENGTH, N_ELEM_BY_RVE)

young_rve = []
for elem in mesh_rve["elements"]:
    xs = mesh_rve["coords"][elem]
    local_pos = (xs[0] + xs[1]) / 2.0  # element center (already in [0, RVE_LENGTH])
    phase1 = (
        local_pos < RVE_FV * RVE_LENGTH / 2.0
        or local_pos > RVE_LENGTH - RVE_FV * RVE_LENGTH / 2.0
    )
    young_rve.append(YOUNG_1 if phase1 else YOUNG_2)

# ---------------------------------------------------------------------------
# Periodic BVP on the VER
# ---------------------------------------------------------------------------
K = compute_stiffness_matrix(mesh_rve, young_rve)
f = periodic_load_vector(mesh_rve, young_rve)

fixed_dof_periodic = [(0, -1)]
Kc, fc = apply_periodic_conditions(K, f, fixed_dofs=fixed_dof_periodic)
sol = np.linalg.solve(Kc, fc)
u_rve = sol[: -len(fixed_dof_periodic)]   # shape (n, 1) — required by compute_strain_and_stress

# ---------------------------------------------------------------------------
# Localization tensors  A = 1 + ε,  B = E · A
# Effective modulus   E* = <B> = (1 / |Y|) ∫_Y B dY
# ---------------------------------------------------------------------------
strain_rve, _ = compute_strain_and_stress(mesh_rve, young_rve, u_rve)
tensor_A = 1.0 + strain_rve
tensor_B = np.array(young_rve).reshape((-1, 1)) * tensor_A
young_effective = float(integrate(mesh_rve, tensor_B) / RVE_LENGTH)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
out = Path(__file__).with_suffix(".csv")
np.savetxt(
    out,
    np.array([[young_effective]]),
    delimiter=",",
    header="young_effective",
    comments="",
)
print(f"[homogenization] young_effective={young_effective:.6e}  wrote {out}")
