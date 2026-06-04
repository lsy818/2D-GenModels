"""Compute quality metrics and produce evaluation figures.

Usage (from project root)::

    python -m code.main_evaluate
"""

from __future__ import annotations

from code.config import ExperimentConfig, SystemConfig
from code.evaluate import run_evaluation


def main() -> None:
    cfg = ExperimentConfig()
    SystemConfig.set_seed_all(cfg.seed)

    print("=" * 60)
    print("MODEL EVALUATION  (Tasks 3 & 4)")
    print("=" * 60)
    run_evaluation(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
