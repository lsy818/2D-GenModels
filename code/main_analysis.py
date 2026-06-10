"""Run data analysis and produce all analysis figures.

Usage (from project root)::

    python -m code.main_analysis
"""

from __future__ import annotations

from code.config import ExperimentConfig, SystemConfig
from code.visualization import make_all_analysis_figures


def main() -> None:
    cfg = ExperimentConfig()
    SystemConfig.set_seed_all(cfg.seed)

    print("=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)
    make_all_analysis_figures()
    print("\nDone.")


if __name__ == "__main__":
    main()
