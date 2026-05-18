"""Aggregate individual parametric error CSVs into a single summary file.

Used as a Snakemake script rule:
    - ``snakemake.input`` : list of per-RVE-size CSV files
    - ``snakemake.output``: single summary CSV
"""

import csv
from pathlib import Path


def aggregate(input_files, output_file):
    """Read all single-row CSVs and concatenate into one summary CSV.

    Args:
        input_files: iterable of Path-like objects pointing to per-case CSVs.
        output_file: Path-like for the output summary CSV.
    """
    rows = []
    for f in sorted(input_files, key=lambda p: float(Path(p).stem.split("_")[-1])):
        with open(f) as fid:
            reader = csv.DictReader(fid)
            for row in reader:
                rows.append(row)

    if not rows:
        raise RuntimeError("No input files to aggregate")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[aggregate] wrote {output_file} ({len(rows)} rows)")


if __name__ == "__main__":
    # When called as a Snakemake script
    aggregate(snakemake.input, snakemake.output[0])  # noqa: F821
