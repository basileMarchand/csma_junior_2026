"""Plot Arlequin L2 error vs RVE size (convergence study).

Used as a Snakemake script rule:
    - ``snakemake.input`` : summary.csv
    - ``snakemake.output``: convergence_plot.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(summary_file, output_file):
    """Read summary CSV and produce a log-log convergence plot.

    Args:
        summary_file: Path to the summary CSV with columns
            ``rve_length`` and ``arlequin_l2_error``.
        output_file: Path for the output PNG figure.
    """
    with open(summary_file) as fid:
        reader = csv.DictReader(fid)
        rows = list(reader)

    rve_lengths = np.array([float(r["rve_length"]) for r in rows])
    errors = np.array([float(r["arlequin_l2_error"]) for r in rows])

    # Sort by rve_length
    order = np.argsort(rve_lengths)
    rve_lengths = rve_lengths[order]
    errors = errors[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(rve_lengths, errors, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("RVE length (m)", fontsize=12)
    ax.set_ylabel("Arlequin L2 error", fontsize=12)
    ax.set_title("Convergence: Arlequin error vs RVE size", fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_file}")


if __name__ == "__main__":
    # When called as a Snakemake script
    plot_convergence(snakemake.input[0], snakemake.output[0])  # noqa: F821
