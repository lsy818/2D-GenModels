"""Evaluation orchestration — compute metrics for all trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from code.config import CLASS_NAMES, N_CLASSES, MODEL_DIR, ExperimentConfig
from code.dataset import load_data, get_class_data
from code.metrics import compute_all_metrics
from code.models import KDEModel, GMMModel, VAEModel, DiffusionModel
from code.visualize import (
    plot_metrics_heatmap, plot_metrics_bars, plot_metrics_radar,
    plot_mode_coverage_detail, print_metrics_table,
)


def load_all_models() -> Dict[str, Dict[int, object]]:
    """Load all 16 trained models from disk."""
    all_models: Dict[str, Dict[int, object]] = {}

    for mtype, cls in [("KDE", KDEModel), ("GMM", GMMModel)]:
        all_models[mtype] = {}
        for c in range(N_CLASSES):
            all_models[mtype][c] = cls.load(
                MODEL_DIR / f"{mtype.lower()}_class{c}.pt")

    for mtype, cls in [("VAE", VAEModel), ("Diffusion", DiffusionModel)]:
        all_models[mtype] = {}
        for c in range(N_CLASSES):
            all_models[mtype][c] = cls.load(
                MODEL_DIR / f"{mtype.lower()}_class{c}.pt", map_location="cpu")

    return all_models


def evaluate_all(all_models: Dict[str, Dict[int, object]],
                 n_gen: int = 2000) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute all metrics for every model × class combination.

    Returns
    -------
    metrics_table : dict
        ``{class_name: {model_type: {metric: value}}}``
    """
    test, test_label = load_data("test")
    metrics_table: Dict[str, Dict[str, Dict[str, float]]] = {}

    for c in range(N_CLASSES):
        cn = CLASS_NAMES[c]
        real = get_class_data(test, test_label, c)
        print(f"\n{'='*50}")
        print(f"Evaluating: {cn}")
        print(f"{'='*50}")

        metrics_table[cn] = {}
        for mtype in ["KDE", "GMM", "VAE", "Diffusion"]:
            model = all_models[mtype][c]
            generated = model.sample(n_gen)
            mets = compute_all_metrics(real, generated, model, mtype, c, cn)
            metrics_table[cn][mtype] = mets

            # Print one-line summary
            print(f"  {mtype:<12}  "
                  f"MMD={mets['MMD']:.4f}  "
                  f"Wass={mets['Wasserstein']:.3f}  "
                  f"Prec={mets['Precision']:.3f}  "
                  f"Cov={mets['Coverage']:.3f}  "
                  f"NLL={mets['NLL']:.4f}  "
                  f"ModeCov={mets['ModeCoverage']:.3f}")

    return metrics_table


def run_evaluation(cfg: ExperimentConfig = None) -> None:
    """Load models, compute metrics, produce all evaluation figures."""
    if cfg is None:
        cfg = ExperimentConfig()

    print("Loading trained models ...")
    all_models = load_all_models()

    print("\nComputing quality metrics ...")
    metrics_table = evaluate_all(all_models, n_gen=2000)

    print("\nGenerating evaluation figures ...")
    plot_metrics_heatmap(metrics_table)
    plot_metrics_bars(metrics_table)
    plot_metrics_radar(metrics_table)
    plot_mode_coverage_detail(metrics_table)
    print_metrics_table(metrics_table)

    print("\nEvaluation complete.")
