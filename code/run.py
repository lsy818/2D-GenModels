"""Entry point — train models, run experiments, generate all figures.

Usage:
    python -m code.run              # train + generate all figures
    python -m code.run --figs-only  # only generate figures (models must exist)
    python -m code.run --ext-only   # only run extension experiments
"""

from __future__ import annotations

import sys

from code.config import set_seed, get_device

set_seed(42)


def main():
    args = set(sys.argv[1:])
    figs_only = "--figs-only" in args
    ext_only = "--ext-only" in args

    print(f"Device: {get_device()}")
    print("=" * 60)

    if not figs_only and not ext_only:
        print("STEP 1/3: Training main models ...")
        from code.experiments import train_main
        train_main()

        print("\nSTEP 2/3: Running extension experiments ...")
        run_extensions()

    elif ext_only:
        print("Running extension experiments only ...")
        run_extensions()

    if not ext_only:
        print("\nSTEP 3/3: Generating all figures ...")
        from code.visualization import make_all_figures
        make_all_figures()

    print("\nDone.")


def run_extensions():
    from code.experiments import ablation, conditional, robustness
    e1 = ablation()
    e2 = conditional()
    e3 = robustness()
    print("\nExtension results summary saved.")
    return e1, e2, e3


if __name__ == "__main__":
    main()
