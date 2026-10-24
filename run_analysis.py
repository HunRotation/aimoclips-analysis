"""Run the manuscript analyses from the included de-identified inputs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = [
    "descriptive_statistics",
    "core_statistics",
    "system_deviation_tests",
    "distance_wald_tests",
    "metric_associations",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run with Python 3.10.12 after installing requirements.txt. Outputs are written to results/.",
    )
    parser.parse_args()
    environment = os.environ.copy()
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        environment[variable] = "1"
    for stage in STAGES:
        print(f"Running {stage}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "analysis" / f"{stage}.py")],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    print("Complete. Numerical outputs are in results/.")


if __name__ == "__main__":
    main()
