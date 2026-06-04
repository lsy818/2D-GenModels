"""Train all four generative models and produce comparison figures.

Usage (from project root)::

    python -m code.main_train
"""

from __future__ import annotations

from code.config import ExperimentConfig, SystemConfig
from code.dataset import load_data
from code.trainer import train_all_models
from code.visualize import make_all_generation_figures


def main() -> None:
    cfg = ExperimentConfig()
    SystemConfig.set_seed_all(cfg.seed)

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    all_models, train, train_label = train_all_models(cfg.model)

    print("\n" + "=" * 60)
    print("MODEL VISUALISATION")
    print("=" * 60)
    make_all_generation_figures(
        all_models, train, train_label,
        n_samples=cfg.n_samples_generate,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
