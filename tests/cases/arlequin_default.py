"""Arlequin coupling: local heterogeneous + global homogeneous model.

Standalone validation script — no pytest, no argparse.
Run directly:   python arlequin_default.py
Output:         arlequin_default.csv  (next to this script)

Physical setup
--------------
- Bar L = 1 m clamped at both ends, uniform load q = -1000 N/m.
- Microstructure: RVE = 0.01 m, fv = 0.5, E1 = 200 000 MPa, E2 = 10 000 MPa.
- Model 1 (heterogeneous) : [0, 0.5],  h1 = 0.001 m  (1 000 elements)
- Model 2 (homogeneous)   : [0.3, 1.],  h2 ≈ 0.002 m  (refinement × 2)
- Overlap zone            : [0.3, 0.5], H1 coupling (h1_coeff = 1)
- Reference               : fine heterogeneous monolithic solve on [0, 1].

Output scalars
--------------
- arlequin_l2_error : weighted L2 norm of (u_arlequin - u_ref) on ZOI [0, 0.2]
                      normalised by x_end = 0.5.
- u_max_model1      : max |u| on model 1
- u_max_model2      : max |u| on model 2

Optional plot
-------------
If the environment variable ``CSMA_PLOT=1`` is set, a comparison figure of the
displacement fields (Arlequin model 1, Arlequin model 2, monolithic reference)
is written to ``results/figures/arlequin_default.png``. The validation CSV
itself is *not* affected, so reference oracles remain valid.
"""
import os
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
BEAM_LENGTH       = 1.0
RVE_LENGTH        = 0.01
RVE_FV            = 0.5
YOUNG_1           = 200_000.0
YOUNG_2           = 10_000.0
N_ELEM_BY_RVE     = 10
X_START_COUPLING  = 0.3
X_END_COUPLING    = 0.5
H1_COEFF          = 1.0
REFINEMENT_FACTOR = 2.0
LOAD              = -1000.0

E_HOMO = 1.0 / (RVE_FV / YOUNG_1 + (1.0 - RVE_FV) / YOUNG_2)
nb_elem  = int(BEAM_LENGTH / RVE_LENGTH * N_ELEM_BY_RVE)
elem_size = BEAM_LENGTH / nb_elem

x_start = X_START_COUPLING
x_end   = X_END_COUPLING


def load_f(x):
    return LOAD


def alpha1(x):
    if x < x_start:             return 0.999
    elif x_start <= x <= x_end: return 0.5
    return 0.001


def alpha2(x):
    return 1.0 - alpha1(x)


# ---------------------------------------------------------------------------
# Model 1 — heterogeneous [0, x_end]
# ---------------------------------------------------------------------------
mesh1 = create_mesh_linear(0, x_end, int(x_end / elem_size))

young1 = []
for elem in mesh1["elements"]:
    xs = mesh1["coords"][elem]
    e_center = (xs[0] + xs[1]) / 2.0
    nb_rve   = int(e_center / RVE_LENGTH)
    loc_pos  = e_center - nb_rve * RVE_LENGTH
    young1.append(YOUNG_1 if loc_pos < RVE_FV * RVE_LENGTH else YOUNG_2)

K1 = compute_stiffness_matrix(mesh1, young1, alpha_f=alpha1)
f1 = volumetric_load_vector(mesh1, load_f, alpha_f=alpha1)
fixed_dof1 = [0]
Kc1, fc1 = apply_dirichlet_boundary_conditions(K1, f1, fixed_dofs=fixed_dof1)

# ---------------------------------------------------------------------------
# Model 2 — homogeneous [x_start, BEAM_LENGTH]
# ---------------------------------------------------------------------------
mesh2 = create_mesh_linear(
    x_start, BEAM_LENGTH,
    int((BEAM_LENGTH - x_start) / (elem_size * REFINEMENT_FACTOR)),
)

K2 = compute_stiffness_matrix(mesh2, [E_HOMO] * mesh2["nb_elements"], alpha_f=alpha2)
f2 = volumetric_load_vector(mesh2, load_f, alpha_f=alpha2)
fixed_dof2 = [-1]
Kc2, fc2 = apply_dirichlet_boundary_conditions(K2, f2, fixed_dofs=fixed_dof2)

# ---------------------------------------------------------------------------
# Lagrange multiplier coupling
# ---------------------------------------------------------------------------
lag_h    = np.max([mesh1["dx"].min(), mesh2["dx"].min()])
lag_mesh = create_mesh_linear(x_start, x_end, int((x_end - x_start) / lag_h))
C1 = compute_coupling(lag_mesh, mesh1, param_h1=H1_COEFF)
C2 = compute_coupling(lag_mesh, mesh2, param_h1=H1_COEFF)

lhs, rhs = build_full_system(Kc1, Kc2, fc1, fc2, C1, C2)
sol = np.linalg.solve(lhs, rhs)

u_model1 = sol[: mesh1["nb_nodes"]].flatten()
start_dof = mesh1["nb_nodes"] + len(fixed_dof1)
u_model2  = sol[start_dof : start_dof + mesh2["nb_nodes"]].flatten()

# ---------------------------------------------------------------------------
# Reference: fine heterogeneous monolithic solve
# ---------------------------------------------------------------------------
mesh_ref = create_mesh_linear(0, BEAM_LENGTH, nb_elem)
young_ref = []
for elem in mesh_ref["elements"]:
    xs = mesh_ref["coords"][elem]
    e_center = (xs[0] + xs[1]) / 2.0
    nb_rve   = int(e_center / RVE_LENGTH)
    loc_pos  = e_center - nb_rve * RVE_LENGTH
    young_ref.append(YOUNG_1 if loc_pos < RVE_FV * RVE_LENGTH else YOUNG_2)

K_ref = compute_stiffness_matrix(mesh_ref, young_ref)
f_ref = volumetric_load_vector(mesh_ref, load_f)
Kc_r, fc_r = apply_dirichlet_boundary_conditions(K_ref, f_ref, fixed_dofs=[0, -1])
u_ref = np.linalg.solve(Kc_r, fc_r)[:-2].flatten()

# ---------------------------------------------------------------------------
# L2 error on ZOI [0, 0.2]
# ---------------------------------------------------------------------------
zoi  = 0.2
mask = mesh1["coords"] <= zoi
delta     = u_ref[: u_model1.shape[0]][mask] - u_model1[mask]
l2_op     = compute_l2_operator(mesh1)
l2_op_zoi = l2_op[np.ix_(mask, mask)]
error     = float(
    np.sqrt((delta.reshape(1, -1) @ l2_op_zoi @ delta.reshape(-1, 1)).item()) / x_end
)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
out = Path(__file__).with_suffix(".csv")
np.savetxt(
    out,
    np.array([[error, float(np.max(np.abs(u_model1))), float(np.max(np.abs(u_model2)))]]),
    delimiter=",",
    header="arlequin_l2_error,u_max_model1,u_max_model2",
    comments="",
)
print(f"[arlequin] error={error:.6e}  wrote {out}")

# ---------------------------------------------------------------------------
# Optional comparison plot (CSMA_PLOT=1)
# ---------------------------------------------------------------------------
if os.environ.get("CSMA_PLOT") == "1":
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend (CI-safe)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        mesh_ref["coords"], u_ref,
        color="black", linewidth=1.2,
        label="Référence monolithique (hétérogène fin)",
    )
    ax.plot(
        mesh1["coords"], u_model1,
        color="tab:blue", linewidth=1.5, marker="o", markersize=3, markevery=20,
        label="Arlequin — modèle 1 (hétérogène)",
    )
    ax.plot(
        mesh2["coords"], u_model2,
        color="tab:red", linewidth=1.5, marker="s", markersize=3, markevery=10,
        label="Arlequin — modèle 2 (homogénéisé)",
    )
    ax.axvspan(
        X_START_COUPLING, X_END_COUPLING,
        color="gray", alpha=0.12, label="Zone de recouvrement",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("u [m]")
    ax.set_title("arlequin_default — champ de déplacement Arlequin vs référence")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    fig_dir = Path(__file__).resolve().parents[2] / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "arlequin_default.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[arlequin] plot wrote {fig_path}")
