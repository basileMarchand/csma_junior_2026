"""Approach 2 — Data-driven validation engine.

Discovery rule
--------------
For every ``<name>.ref.csv`` found in ``tests/cases/``, one test is created:

    test_case[<name>]

No Python code needs to change when a new case is added — just drop two files:
  - ``tests/cases/<name>.py``      : the standalone simulation script
  - ``tests/cases/<name>.ref.csv`` : the reference output (committed oracle)

Optionally add ``tests/cases/<name>.toml`` to override the default tolerances.

How it works
------------
1. Discover all ``*.ref.csv`` in ``tests/cases/``.
2. For each one, assert the companion ``<name>.py`` exists.
3. Run ``<name>.py`` as a subprocess (inherits the installed environment).
   The script writes ``<name>.csv`` next to itself (using ``Path(__file__)``).
4. Load both CSVs with ``np.loadtxt`` and compare with ``assert_allclose``.
   Tolerances come from ``<name>.toml`` (default: rtol=1e-8, atol=0.0).

Default tolerances
------------------
If ``<name>.toml`` is absent, rtol=1e-8 and atol=0.0 are used.
Override per-case::

    # analytical_traction_default.toml
    rtol = 1e-10
    atol = 1e-15
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

CASES_DIR = Path(__file__).parent / "cases"
DEFAULT_TOL = {"rtol": 1e-8, "atol": 0.0}


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def list_references() -> list[Path]:
    return sorted(CASES_DIR.glob("*.ref.csv"))


# ---------------------------------------------------------------------------
# Parametrized test — one entry per .ref.csv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ref_csv",
    list_references(),
    ids=lambda p: p.name.removesuffix(".ref.csv"),
)
def test_case(ref_csv: Path) -> None:
    """Run the companion script and compare its CSV output to the reference.

    The test name shown by pytest uses the file stem, e.g.::

        test_case[monolithic_default]
        test_case[analytical_arlequin_homo_default]
    """
    stem   = ref_csv.name.removesuffix(".ref.csv")
    script = CASES_DIR / f"{stem}.py"
    toml   = CASES_DIR / f"{stem}.toml"

    # 1. Companion script must exist
    assert script.is_file(), (
        f"No companion script '{script.name}' found for '{ref_csv.name}'.\n"
        f"Every .ref.csv needs a matching .py in {CASES_DIR}."
    )

    # 2. Run the script (uses the same Python / installed packages as pytest)
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"Script '{script.name}' exited with code {proc.returncode}.\n"
        f"--- stdout ---\n{proc.stdout}"
        f"--- stderr ---\n{proc.stderr}"
    )

    # 3. The script must have produced <stem>.csv next to itself
    produced = CASES_DIR / f"{stem}.csv"
    assert produced.is_file(), (
        f"Script '{script.name}' ran successfully but did not produce "
        f"'{produced.name}'.\nCheck that the script writes to "
        f"Path(__file__).with_suffix('.csv')."
    )

    # 4. Load & compare
    actual   = np.loadtxt(produced,  delimiter=",", skiprows=1)
    expected = np.loadtxt(ref_csv,   delimiter=",", skiprows=1)

    tol = DEFAULT_TOL.copy()
    if toml.is_file():
        tol.update(tomllib.loads(toml.read_text()))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tol["rtol"],
        atol=tol["atol"],
        err_msg=f"[{stem}] output CSV does not match reference",
    )
