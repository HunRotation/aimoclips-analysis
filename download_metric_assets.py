"""Download the CLAP checkpoint or FAD reference audio with SHA-256 verification.

Run with --help for options. Downloads require --download; omitting it prints
the plan. Uses the downloader preserved with the reference-recovery package.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    helper = (
        Path(__file__).resolve().parent
        / "reports/analysis/aimoclips_metric_recovery/scripts/download_metric_assets.py"
    )
    runpy.run_path(str(helper), run_name="__main__")
