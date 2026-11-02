#!/usr/bin/env python3
"""
FMD Risk Surveillance Pipeline — sequential executor.

Usage:
    python run_pipeline.py            # run all steps
    python run_pipeline.py --from 4  # resume from step 4 onwards

Steps:
    1  Load data (cadastro + GTAS movements)
    2a Javali spatial distribution
    2b Temperature / LST raster extraction
    3a Road animal pressure (heavy — cached on disk)
    3b Spatial network features
    4  Distance to international border
    5  Combine and validate all features
    6  FMD risk analysis → results/fmd_risk_scored.csv

Before the first run, build the full road network once (not part of this
automatic sequence — see that script's header):
    python pipeline/3_0_download_full_roads.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import RESULTS, INTERMEDIATE  # also triggers ensure_dirs()

PIPELINE = PROJECT_ROOT / "pipeline"

STEPS = [
    (1,  "Load data (cadastro + GTAS movements)",
         PIPELINE / "1_load_data.py"),
    (2,  "Javali spatial distribution",
         PIPELINE / "2_1_javalis_spatial_distribution.py"),
    (3,  "Temperature / LST raster extraction",
         PIPELINE / "2_get_temperature_rs_with_raster.py"),
    (4,  "Road animal pressure (routing — heavy, cached)",
         PIPELINE / "3_1_road_animal_preassure.py"),
    (5,  "Spatial network features",
         PIPELINE / "3_get_spatial_net_features.py"),
    (6,  "Distance to international border",
         PIPELINE / "5_distance_to_border.py"),
    (7,  "Combine and validate all features",
         PIPELINE / "7_combine_and_check_data.py"),
    (8,  "FMD risk analysis → results/",
         PIPELINE / "8_fmd_risk_analysis.py"),
]


def run_step(num: int, label: str, script: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {num:>2}: {label}")
    print(f"  Script : {script.name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print(f"\n[FAILED] Step {num} exited with code {result.returncode}.")
        print("Pipeline halted. Fix the error above and re-run.")
        sys.exit(result.returncode)

    print(f"[OK] Step {num} complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="FMD Risk Pipeline runner")
    parser.add_argument(
        "--from",
        dest="from_step",
        type=int,
        default=1,
        metavar="N",
        help="Start execution from step N (default: 1 = full pipeline)",
    )
    args = parser.parse_args()

    active = [(n, lbl, scr) for n, lbl, scr in STEPS if n >= args.from_step]

    if not active:
        print(f"No steps >= {args.from_step}. Nothing to do.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("  FMD Risk Surveillance Pipeline")
    print(f"  Running steps {active[0][0]} – {active[-1][0]}")
    print(f"  Intermediate cache : {INTERMEDIATE}")
    print(f"  Final results      : {RESULTS}")
    print("=" * 60)

    for num, label, script in active:
        run_step(num, label, script)

    csv_out = RESULTS / "fmd_risk_scored.csv"
    print("\n" + "=" * 60)
    print("  FMD Risk Surveillance Pipeline — COMPLETE")
    if csv_out.exists():
        print(f"  Final output : {csv_out}")
    else:
        print(f"  WARNING: {csv_out} not found — check step 8 logs.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
