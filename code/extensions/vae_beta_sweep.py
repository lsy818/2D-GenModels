"""Further beta sweep for VAE diagnostics.

This script focuses on the two most informative failure cases by default:
Two Moons and Spiral. It explores whether weakening the KL term beyond
beta=0.1 keeps improving generation or creates a new prior-sampling failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.config import CLASS_COLORS, CLASS_NAMES, DATA_DIR, FIGURES_DIR, MODEL_DIR, set_seed
from code.models.vae_model import VAEModel
from code.extensions.vae_improvement_experiments import (
    VariantResult,
    _evaluate,
    _load_custom_vae,
    _prior_sample,
    train_or_load_custom_vae,
)


OUT_DIR = FIGURES_DIR / "extensions"
MODEL_OUT = MODEL_DIR / "extensions"
SHORT_NAMES = ["GMM", "Ring", "Moons", "Spiral"]
BETAS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.0]
DEFAULT_CLASSES = [2, 3]


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)


def _beta_tag(beta: float) -> str:
    if beta == 1.0:
        return "1"
    if beta == 0.0:
        return "0"
    return str(beta).replace(".", "p")


def _variant_name(beta: float) -> str:
    if beta == 1.0:
        return "Beta1.0"
    if beta == 0.0:
        return "Beta0"
    return f"Beta{beta:g}"


def _load_beta_model(class_id: int, beta: float, train_class: np.ndarray):
    if beta == 1.0:
        wrapper = VAEModel.load(MODEL_DIR / f"vae_class{class_id}.pt", map_location="cpu")
        assert wrapper.vae is not None
        wrapper.vae.eval()
        return wrapper.vae

    if beta == 0.1:
        existing = MODEL_OUT / f"vae_beta01_class{class_id}.pt"
        if existing.exists():
            print(f"  loading beta=0.1: {existing.name}")
            return _load_custom_vae(existing)

    path = MODEL_OUT / f"vae_beta_sweep_b{_beta_tag(beta)}_class{class_id}.pt"
    return train_or_load_custom_vae(
        path,
        train_class,
        beta=beta,
        free_bits=0.0,
        seed=7000 + class_id * 100 + int(round(beta * 1000)),
        label=fr"beta={beta:g}",
    )


def run_beta_sweep(classes: list[int] | None = None) -> tuple[list[VariantResult], dict[str, dict[int, np.ndarray]]]:
    _ensure_dirs()
    set_seed(42)
    if classes is None:
        classes = DEFAULT_CLASSES

    train = np.load(DATA_DIR / "train.npy").astype(np.float32)
    train_label = np.load(DATA_DIR / "train_label.npy").astype(np.int64)
    test = np.load(DATA_DIR / "test.npy").astype(np.float32)
    test_label = np.load(DATA_DIR / "test_label.npy").astype(np.int64)

    results: list[VariantResult] = []
    samples: dict[str, dict[int, np.ndarray]] = {"Data": {}}
    for beta in BETAS:
        samples[_variant_name(beta)] = {}

    for class_id in classes:
        train_class = train[train_label == class_id]
        test_class = test[test_label == class_id]
        samples["Data"][class_id] = test_class

        print(f"\nClass {class_id}: {CLASS_NAMES[class_id]}")
        for beta in BETAS:
            model = _load_beta_model(class_id, beta, train_class)
            generated = _prior_sample(model, len(test_class), seed=8100 + class_id * 100 + int(round(beta * 1000)))
            variant = _variant_name(beta)
            samples[variant][class_id] = generated
            row = _evaluate(class_id, variant, model, train_class, test_class, generated)
            results.append(row)
            print(
                f"    beta={beta:<4g} MMD={row.mmd:.4f} Cov={row.coverage:.3f} "
                f"Prec={row.precision:.3f} Off={row.off_support:.3f} "
                f"Recon={row.recon_mse:.4f} KL={row.kl_total:.3f}"
            )

    return results, samples


def _format_xy(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])
    ax.grid(color="#e7eaf0", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#ccd1dc")
        spine.set_linewidth(0.7)


def _subsample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[rng.choice(len(points), n, replace=False)]


def figure_sample_grid(samples: dict[str, dict[int, np.ndarray]], classes: list[int]) -> None:
    rng = np.random.default_rng(42)
    columns = ["Data"] + [_variant_name(beta) for beta in BETAS]
    titles = ["Data"] + [fr"$\beta={beta:g}$" for beta in BETAS]
    fig, axes = plt.subplots(len(classes), len(columns), figsize=(12.4, 2.45 * len(classes)), constrained_layout=True)
    if len(classes) == 1:
        axes = np.expand_dims(axes, axis=0)

    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=10, weight="bold")

    for i, class_id in enumerate(classes):
        for j, col in enumerate(columns):
            ax = axes[i, j]
            pts = _subsample(samples[col][class_id], 1000, rng)
            color = "#30343b" if col == "Data" else CLASS_COLORS[class_id]
            ax.scatter(pts[:, 0], pts[:, 1], s=4.0, c=color, alpha=0.52, linewidths=0, rasterized=True)
            _format_xy(ax)
            if j == 0:
                ax.set_ylabel(SHORT_NAMES[class_id], fontsize=9, weight="bold")

    fig.suptitle("VAE beta sweep: visual prior samples", fontsize=12, weight="bold")
    fig.savefig(OUT_DIR / "vae_beta_sweep_samples.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_metric_curves(results: list[VariantResult], classes: list[int]) -> None:
    labels = [fr"{beta:g}" for beta in BETAS]
    x = np.arange(len(BETAS))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), constrained_layout=True)
    panels = [
        ("coverage", "Coverage"),
        ("precision", "Precision"),
        ("off_support", "Off-support"),
        ("mmd", "MMD"),
        ("recon_mse", "Recon MSE"),
        ("kl_total", "KL"),
    ]

    for ax, (attr, title) in zip(axes.ravel(), panels):
        for class_id in classes:
            vals = []
            for beta in BETAS:
                row = next(r for r in results if r.class_id == class_id and r.variant == _variant_name(beta))
                vals.append(getattr(row, attr))
            ax.plot(x, vals, marker="o", lw=1.8, color=CLASS_COLORS[class_id], label=SHORT_NAMES[class_id])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r"$\beta$")
        ax.set_title(title, weight="bold")
        ax.grid(color="#e7eaf0", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("#ccd1dc")
            spine.set_linewidth(0.7)

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Metric trends under weaker KL regularisation", fontsize=12, weight="bold")
    fig.savefig(OUT_DIR / "vae_beta_sweep_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    classes = DEFAULT_CLASSES
    results, samples = run_beta_sweep(classes)
    figure_sample_grid(samples, classes)
    figure_metric_curves(results, classes)
    print(f"\nSaved figures to {OUT_DIR}")
    print(f"Computed {len(results)} beta-sweep rows.")


if __name__ == "__main__":
    main()
