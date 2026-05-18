"""Approach 1 — Classical pytest: one Python function per test, no CSV.

This file illustrates the *classical* pytest style:
  - each test is a named Python function
  - all inputs and expected values are written directly in the code
  - adding a new case means adding a new function

Compare with ``test_validation.py`` (Approach 2), where every case is
defined by a pair of files in ``tests/cases/`` and zero Python code changes.

When to prefer this style
--------------------------
- Validating a single, specific mathematical property (superconvergence,
  symmetry, energy conservation, …).
- When the expected value can be expressed as a closed-form formula
  *in the code itself* — making the test self-documenting.
- When you want full IDE support and trivial breakpoint debugging.

Case: homogeneous bar under uniform traction
--------------------------------------------
Strong form:  -E·u'' = q,  u(0) = u(L) = 0
Exact:        u(x) = q·x·(L-x) / (2·E)

Superconvergence justification (rtol = 1e-10)
----------------------------------------------
For P1 elements on a *uniform* mesh with 2-point Gauss quadrature:
  - The exact solution is quadratic.
  - The FEM error is orthogonal to the P1 space.
  - The load integral ∫N_i·q dx is exact (polynomial degree ≤ 3).
  ⟹  Nodal FEM values == nodal analytical values up to LAPACK round-off.
"""

import numpy as np

from pybeam.dirichlet_bc import apply_dirichlet_boundary_conditions
from pybeam.fe_operator import compute_stiffness_matrix, volumetric_load_vector
from pybeam.mesh import create_mesh_linear

# ---------------------------------------------------------------------------
# Shared setup (module-level constants, not fixtures)
# ---------------------------------------------------------------------------
YOUNG       = 200_000.0
LOAD        = -1_000.0
BEAM_LENGTH = 1.0
NB_ELEM     = 100

mesh  = create_mesh_linear(0.0, BEAM_LENGTH, NB_ELEM)
K     = compute_stiffness_matrix(mesh, [YOUNG] * mesh["nb_elements"])
f     = volumetric_load_vector(mesh, lambda x: LOAD)
Kc, fc = apply_dirichlet_boundary_conditions(K, f, fixed_dofs=[0, -1])
U_FEM  = np.linalg.solve(Kc, fc)[:-2].flatten()      # FEM nodal displacements
X      = mesh["coords"]
U_EXACT = LOAD * X * (BEAM_LENGTH - X) / (2.0 * YOUNG)  # closed form


# ---------------------------------------------------------------------------
# Test 1 — peak displacement
# ---------------------------------------------------------------------------

def test_u_max_equals_analytical():
    """Peak FEM displacement equals the formula q·L² / (8·E).

    For this case:  |q|·L² / (8·E) = 1000·1² / (8·200000) = 6.25e-4 m.
    Tolerance: rtol=1e-10 (superconvergence — see module docstring).
    """
    expected = abs(LOAD) * BEAM_LENGTH**2 / (8.0 * YOUNG)  # = 6.25e-4 m

    np.testing.assert_allclose(
        np.max(np.abs(U_FEM)),
        expected,
        rtol=1e-10,
        err_msg=(
            f"u_max = {np.max(np.abs(U_FEM)):.6e}, expected {expected:.6e} — "
            "check stiffness assembly or Dirichlet enforcement"
        ),
    )


# ---------------------------------------------------------------------------
# Test 2 — full nodal field
# ---------------------------------------------------------------------------

def test_full_field_equals_analytical():
    """Every FEM nodal value matches the closed-form solution u(x)=q·x·(L-x)/(2E).

    Validates simultaneously:
      - the stiffness matrix assembly (K)
      - the load vector assembly    (f)
      - the Dirichlet BC enforcement
    A bug in any of these breaks the nodal match even if the peak value
    still looks roughly correct.
    """
    np.testing.assert_allclose(
        U_FEM,
        U_EXACT,
        rtol=1e-10,
        atol=1e-15,
        err_msg="FEM nodal field deviates from u(x)=q·x·(L-x)/(2E) at one or more nodes",
    )
