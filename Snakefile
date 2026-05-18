"""Snakemake workflow — Arlequin convergence study.

Runs the Arlequin coupling for several RVE sizes, aggregates the L2 error
results into a summary CSV, and produces a log-log convergence plot.

Usage:
    snakemake --cores all           # run the full pipeline
    snakemake --cores 1 -n          # dry-run (show what would be done)
    snakemake --dag | dot -Tpng > dag.png  # visualize the DAG
"""

RVE_SIZES = ["0.10", "0.08", "0.06", "0.04", "0.02", "0.01"]


rule all:
    input:
        "results/parametric/convergence_plot.png",


rule run_rve:
    """Run a single Arlequin study for a given RVE size."""
    output:
        "results/parametric/error_{rve_size}.csv",
    shell:
        "python -m workflows.run_parametric "
        "--rve-length {wildcards.rve_size} "
        "--output {output}"


rule aggregate:
    """Collect all per-RVE-size error CSVs into a single summary table."""
    input:
        expand("results/parametric/error_{rve_size}.csv", rve_size=RVE_SIZES),
    output:
        "results/parametric/summary.csv",
    script:
        "scripts/aggregate.py"


rule plot:
    """Produce a log-log convergence plot from the summary table."""
    input:
        "results/parametric/summary.csv",
    output:
        "results/parametric/convergence_plot.png",
    script:
        "scripts/plot_convergence.py"
