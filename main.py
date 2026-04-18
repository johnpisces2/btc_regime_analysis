"""
BTC Market Regime Launcher
==========================

Default:
    python main.py
    Launch the Qt GUI.

CLI pipeline:
    python main.py --cli
"""

import argparse
import subprocess
import sys


def run_step(name, args):
    print(f"\n{'=' * 60}")
    print(f"Step: {name}")
    print("=" * 60)
    completed = subprocess.run([sys.executable, *args], check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_cli_pipeline():
    print("=" * 60)
    print("BTC Market Regime Pipeline")
    print("=" * 60)

    run_step("1. Fetch Historical Data", ["collect_data.py"])
    run_step("2. Train Regime Model (default: HMM)", ["train.py", "--model", "hmm"])
    run_step("3. Generate Prediction Charts", ["predict.py", "--update"])
    return 0


def main():
    parser = argparse.ArgumentParser(description="BTC market regime launcher")
    parser.add_argument("--cli", action="store_true", help="Run the original CLI pipeline")
    args = parser.parse_args()

    if args.cli:
        raise SystemExit(run_cli_pipeline())

    from gui import launch

    raise SystemExit(launch())


if __name__ == "__main__":
    main()
