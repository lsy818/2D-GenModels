"""Redesigned visualisations for report/main.tex.

This script writes the exact filenames used by the LaTeX report.  The design
goal is to keep each figure readable in an A4 paper: fewer panels, consistent
styling, and more density/contour summaries instead of large scatter grids.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy import spatial
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.config import CLASS_COLORS, CLASS_NAMES, FIGURES_DIR, MODEL_DIR, N_CLASSES, set_seed
from code.data import (
    effective_dims,
    get_class,
    load_data,
    local_curvature,
    local_noise,
    manifold_thickness,
)
from code.models.diffusion_model import DiffusionModel
from code.models.vae_model import VAEModel

warnings.filterwarnings("ignore", category=UserWarning)

set_seed(42)
RNG = np.random.default_rng(42)

ANALYSIS_DIR = FIGURES_DIR / "analysis"
GEN_DIR = FIGURES_DIR / "generation"
EVAL_DIR = FIGURES_DIR / "evaluation"

MODEL_COLORS = {
    "Real": "#30343b",
    "VAE": "#2f6fb0",
    "DDPM": "#d05a2b",
}

SHORT_NAMES = ["GMM", "Ring", "Moons", "Spiral"]

RESULTS = {
    "Gaussian Mixture": {
        "VAE": {"MMD": 0.0033, "Wasserstein": 0.132, "Coverage": 0.552, "Precision": 0.475, "Score": 0.413},
        "DDPM": {"MMD": 0.0040, "Wasserstein": 0.151, "Coverage": 0.963, "Precision": 0.966, "Score": 0.607},
    },
    "Ring": {
        "VAE": {"MMD": 0.0047, "Wasserstein": 0.116, "Coverage": 0.664, "Precision": 0.806, "Score": 0.250},
        "DDPM": {"MMD": 0.0002, "Wasserstein": 0.095, "Coverage": 0.972, "Precision": 0.992, "Score": 0.851},
    },
    "Two Moons": {
        "VAE": {"MMD": 0.0141, "Wasserstein": 0.111, "Coverage": 0.176, "Precision": 0.536, "Score": 0.131},
        "DDPM": {"MMD": 0.0010, "Wasserstein": 0.087, "Coverage": 0.968, "Precision": 0.966, "Score": 0.799},
    },
    "Spiral": {
        "VAE": {"MMD": 0.0098, "Wasserstein": 0.115, "Coverage": 0.777, "Precision": 0.976, "Score": 0.486},
        "DDPM": {"MMD": 0.0043, "Wasserstein": 0.104, "Coverage": 0.847, "Precision": 0.976, "Score": 0.670},
    },
}

ABLATION_RESULTS = {
    "VAE": {
        "param": [0.1, 1.0, 5.0],
        "label": [r"$\beta=0.1$", r"$\beta=1.0$", r"$\beta=5.0$"],
        "MMD": [0.0462, 0.0942, 0.3243],
        "Precision": [0.962, 0.996, 1.000],
        "Coverage": [0.856, 0.834, 0.040],
    },
    "DDPM": {
        "param": [100, 1000],
        "label": [r"$T=100$", r"$T=1000$"],
        "MMD": [0.0625, 0.0571],
        "Precision": [1.000, 1.000],
        "Coverage": [0.906, 0.904],
    },
}

CONDITIONAL_RESULTS = {
    "Gaussian Mixture": {
        "CVAE": {"MMD": 0.0794, "Precision": 0.522, "Coverage": 0.630},
        "CondDDPM": {"MMD": 0.0369, "Precision": 0.966, "Coverage": 0.956},
    },
    "Ring": {
        "CVAE": {"MMD": 0.0616, "Precision": 0.844, "Coverage": 0.726},
        "CondDDPM": {"MMD": 0.0352, "Precision": 0.992, "Coverage": 0.980},
    },
    "Two Moons": {
        "CVAE": {"MMD": 0.1182, "Precision": 0.648, "Coverage": 0.196},
        "CondDDPM": {"MMD": 0.0580, "Precision": 0.982, "Coverage": 0.930},
    },
    "Spiral": {
        "CVAE": {"MMD": 0.0848, "Precision": 1.000, "Coverage": 0.786},
        "CondDDPM": {"MMD": 0.0632, "Precision": 1.000, "Coverage": 0.864},
    },
}

ROBUSTNESS_RESULTS = {
    0: {
        "VAE": {"MMD": 0.0913, "Precision": 0.998, "Coverage": 0.862},
        "DDPM": {"MMD": 0.0556, "Precision": 0.996, "Coverage": 0.884},
    },
    5: {
        "VAE": {"MMD": 0.0711, "Precision": 0.986, "Coverage": 0.838},
        "DDPM": {"MMD": 0.0799, "Precision": 0.936, "Coverage": 0.842},
    },
    10: {
        "VAE": {"MMD": 0.0388, "Precision": 0.974, "Coverage": 0.856},
        "DDPM": {"MMD": 0.0931, "Precision": 0.912, "Coverage": 0.810},
    },
}


def _set_style() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        rc={
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def _save(fig: plt.Figure, base: Path, dpi: int = 260) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _format_xy(ax: plt.Axes, lim: tuple[float, float] = (-4, 4), ticks: bool = True) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    if ticks:
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d7dbe2")
        spine.set_linewidth(0.7)


def _subsample(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[RNG.choice(len(points), n, replace=False)]


def _class_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("class_cmap", ["#ffffff", color])


def _kde_grid(
    points: np.ndarray,
    bandwidth: float = 0.18,
    grid_n: int = 140,
    lim: tuple[float, float] = (-4, 4),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.linspace(lim[0], lim[1], grid_n)
    xg, yg = np.meshgrid(g, g)
    grid = np.column_stack([xg.ravel(), yg.ravel()])
    kde = KernelDensity(bandwidth=bandwidth).fit(points)
    z = np.exp(kde.score_samples(grid)).reshape(xg.shape)
    z = z / (z.max() + 1e-12)
    return xg, yg, z


def _contour_levels(z: np.ndarray) -> np.ndarray:
    vals = z[z > np.percentile(z, 60)]
    if vals.size == 0:
        return np.array([0.35, 0.55, 0.75])
    qs = np.quantile(vals, [0.30, 0.55, 0.78, 0.92])
    return np.unique(np.clip(qs, 0.05, 0.98))


def _cov_ellipse(points: np.ndarray, ax: plt.Axes, color: str) -> None:
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(vals)
    ell = Ellipse(
        xy=points.mean(axis=0),
        width=width,
        height=height,
        angle=theta,
        facecolor="none",
        edgecolor=color,
        lw=1.2,
        alpha=0.9,
    )
    ax.add_patch(ell)


def _smooth(y: list[float] | np.ndarray, window: int = 25) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if len(arr) < window:
        return arr
    return gaussian_filter1d(arr, sigma=max(window / 6, 1))


def load_models() -> dict[str, dict[int, object]]:
    models: dict[str, dict[int, object]] = {"VAE": {}, "DDPM": {}}
    for c in range(N_CLASSES):
        models["VAE"][c] = VAEModel.load(MODEL_DIR / f"vae_class{c}.pt", map_location="cpu")
        models["DDPM"][c] = DiffusionModel.load(MODEL_DIR / f"diffusion_class{c}.pt", map_location="cpu")
    return models


def figure_comparison_overview(train: np.ndarray, train_label: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    notes = ["8 separated modes", "closed thin support", "two non-convex arcs", "high-curvature path"]

    for c, ax in enumerate(axes.ravel()):
        pts = get_class(train, train_label, c)
        xg, yg, z = _kde_grid(pts, bandwidth=0.18)
        ax.contourf(xg, yg, z, levels=np.linspace(0.08, 1.0, 12), cmap=_class_cmap(CLASS_COLORS[c]), alpha=0.95)
        ax.contour(xg, yg, z, levels=_contour_levels(z), colors="#2f343b", linewidths=0.45, alpha=0.45)
        sub = _subsample(pts, 650)
        ax.scatter(sub[:, 0], sub[:, 1], s=3.5, c=CLASS_COLORS[c], alpha=0.45, linewidths=0, rasterized=True)
        _format_xy(ax)
        ax.set_title(CLASS_NAMES[c], weight="bold")
        ax.text(
            0.03,
            0.04,
            notes[c],
            transform=ax.transAxes,
            color="#40444c",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d7dbe2", alpha=0.90),
        )

    fig.suptitle("Data landscape: scatter + KDE density", weight="bold", y=1.02)
    _save(fig, ANALYSIS_DIR / "comparison_overview")


def figure_pca_analysis(train: np.ndarray, train_label: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4), constrained_layout=True)
    for c, ax in enumerate(axes):
        pts = get_class(train, train_label, c)
        pca = PCA().fit(pts)
        sub = _subsample(pts, 700)
        ax.scatter(sub[:, 0], sub[:, 1], s=3.2, c=CLASS_COLORS[c], alpha=0.38, linewidths=0, rasterized=True)
        _cov_ellipse(pts, ax, "#222222")
        center = pts.mean(axis=0)
        for i, color in enumerate(["#d62728", "#1f77b4"]):
            length = np.sqrt(pca.explained_variance_[i]) * 1.65
            vec = pca.components_[i] * length
            ax.arrow(
                center[0],
                center[1],
                vec[0],
                vec[1],
                width=0.015,
                head_width=0.16,
                length_includes_head=True,
                color=color,
                alpha=0.9,
            )
        _format_xy(ax, ticks=False)
        ratio = pca.explained_variance_ratio_
        ax.set_title(SHORT_NAMES[c], weight="bold")
        ax.text(
            0.04,
            0.05,
            f"PC1 {ratio[0]:.1%}\nPC2 {ratio[1]:.1%}",
            transform=ax.transAxes,
            color="#222",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d7dbe2", alpha=0.92),
        )

    legend = [
        Line2D([0], [0], color="#d62728", lw=1.4, label="PC1"),
        Line2D([0], [0], color="#1f77b4", lw=1.4, label="PC2"),
        Line2D([0], [0], color="#222222", lw=1.2, label="1σ covariance ellipse"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    _save(fig, ANALYSIS_DIR / "pca_analysis")


def figure_noise_structure(train: np.ndarray, train_label: np.ndarray) -> None:
    raw_rows = []
    knn_values = []
    for c in range(N_CLASSES):
        pts = get_class(train, train_label, c)
        noise = local_noise(pts, 30)
        thick = manifold_thickness(pts, 20)
        curv = local_curvature(pts, 15)
        tree = spatial.cKDTree(pts)
        dists, _ = tree.query(pts, k=6)
        radii = dists[:, -1]
        raw_rows.append([noise.mean(), thick.mean(), curv.mean(), np.percentile(radii, 95)])
        knn_values.append(radii)

    raw = np.asarray(raw_rows)
    norm = (raw - raw.min(axis=0)) / (raw.max(axis=0) - raw.min(axis=0) + 1e-12)
    labels = np.array([[f"{v:.3f}" for v in row] for row in raw])

    fig = plt.figure(figsize=(7.2, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        norm,
        ax=ax0,
        annot=labels,
        fmt="",
        cmap="mako_r",
        vmin=0,
        vmax=1,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "relative level"},
    )
    ax0.set_yticklabels(SHORT_NAMES, rotation=0)
    ax0.set_xticklabels(["local\nnoise", "manifold\nthickness", "local\ncurvature", "kNN p95"], rotation=0)
    ax0.set_title("Local structure summary", weight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    for c, vals in enumerate(knn_values):
        sns.kdeplot(vals, ax=ax1, color=CLASS_COLORS[c], lw=1.7, label=SHORT_NAMES[c], clip=(0, vals.max() * 1.1))
        ax1.axvline(np.percentile(vals, 95), color=CLASS_COLORS[c], lw=0.8, ls="--", alpha=0.65)
    ax1.set_title("kNN radius distribution", weight="bold")
    ax1.set_xlabel("distance to 5th neighbour")
    ax1.set_ylabel("density")
    ax1.grid(axis="y", color="#e7eaf0", lw=0.6)
    sns.despine(ax=ax1)
    ax1.legend(frameon=False, ncol=1)

    _save(fig, ANALYSIS_DIR / "noise_structure")


def figure_effective_dimension(train: np.ndarray, train_label: np.ndarray) -> None:
    k_vals = [5, 10, 20, 40, 80, 160]
    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    ax.axhspan(0.9, 1.1, color="#e8f1ff", alpha=0.7, label="near 1D")
    ax.axhspan(1.9, 2.1, color="#f6ebff", alpha=0.45, label="near 2D")
    for c in range(N_CLASSES):
        est = effective_dims(get_class(train, train_label, c), k_vals)
        ax.plot(k_vals, est, marker="o", ms=4.2, lw=1.8, color=CLASS_COLORS[c], label=SHORT_NAMES[c])
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(4.5, 180)
    ax.set_ylim(0.35, 2.35)
    ax.set_xlabel("neighbourhood size k")
    ax.set_ylabel("estimated intrinsic dimension")
    ax.set_title("Cross-scale effective dimensionality", weight="bold")
    ax.grid(color="#e7eaf0", lw=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    sns.despine(ax=ax)
    _save(fig, ANALYSIS_DIR / "effective_dimension")


def figure_polar_analysis(train: np.ndarray, train_label: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    for j, c in enumerate([1, 3]):
        pts = get_class(train, train_label, c)
        r = np.sqrt(np.sum(pts**2, axis=1))
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        ax = axes[0, j]
        ax.scatter(theta, r, s=3.2, c=CLASS_COLORS[c], alpha=0.45, linewidths=0, rasterized=True)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xlabel(r"angle $\theta$")
        ax.set_ylabel("radius r")
        ax.set_title(f"{CLASS_NAMES[c]} in polar coordinates", weight="bold")
        ax.grid(color="#e7eaf0", lw=0.6)
        sns.despine(ax=ax)

        ax = axes[1, j]
        ax.hist(r, bins=46, density=True, color=CLASS_COLORS[c], alpha=0.72, edgecolor="white", linewidth=0.35)
        ax.axvline(r.mean(), color="#222222", ls="--", lw=1.1, label=fr"$\bar r={r.mean():.2f}$")
        ax.set_xlabel("radius r")
        ax.set_ylabel("density")
        ax.set_title("radial density", weight="bold")
        ax.grid(axis="y", color="#e7eaf0", lw=0.6)
        ax.legend(frameon=False)
        sns.despine(ax=ax)
    _save(fig, ANALYSIS_DIR / "polar_analysis")


def figure_summary_radar(train: np.ndarray, train_label: np.ndarray) -> None:
    metrics = {
        "Noise": [],
        "Thickness": [],
        "Curvature": [],
        "Nonlinear": [],
        "Spread": [],
    }
    for c in range(N_CLASSES):
        pts = get_class(train, train_label, c)
        metrics["Noise"].append(local_noise(pts, 30).mean())
        metrics["Thickness"].append(manifold_thickness(pts, 20).mean())
        metrics["Curvature"].append(local_curvature(pts, 15).mean())
        pca = PCA().fit(pts)
        metrics["Nonlinear"].append(1.0 - abs(float(pca.explained_variance_ratio_[0]) - 0.5) * 2)
        metrics["Spread"].append(np.mean(spatial.distance.pdist(_subsample(pts, 700))))

    keys = list(metrics)
    values = np.asarray([metrics[k] for k in keys], dtype=float)
    norm = (values - values.min(axis=1, keepdims=True)) / (values.max(axis=1, keepdims=True) - values.min(axis=1, keepdims=True) + 1e-12)

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw={"polar": True}, constrained_layout=True)
    for c in range(N_CLASSES):
        vals = np.concatenate([norm[:, c], norm[:1, c]])
        ax.plot(angles, vals, lw=1.8, marker="o", ms=3.5, color=CLASS_COLORS[c], label=SHORT_NAMES[c])
        ax.fill(angles, vals, color=CLASS_COLORS[c], alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels([])
    ax.grid(color="#d9dde6", lw=0.7)
    ax.spines["polar"].set_color("#cfd5df")
    ax.set_title("Geometry difficulty portrait", weight="bold", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
    _save(fig, ANALYSIS_DIR / "summary_radar")


def figure_generation_comparison(
    train: np.ndarray,
    train_label: np.ndarray,
    samples: dict[str, dict[int, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.0), constrained_layout=True)
    for c, ax in enumerate(axes.ravel()):
        real = get_class(train, train_label, c)
        xg, yg, zr = _kde_grid(real, bandwidth=0.18, grid_n=150)
        ax.contourf(xg, yg, zr, levels=np.linspace(0.10, 1.0, 10), cmap="Greys", alpha=0.34)
        ax.contour(xg, yg, zr, levels=_contour_levels(zr), colors="#4a4f57", linewidths=0.8, alpha=0.70)
        for model_name, ls, lw in [("VAE", "--", 1.25), ("DDPM", "-", 1.45)]:
            _, _, z = _kde_grid(samples[model_name][c], bandwidth=0.18, grid_n=150)
            ax.contour(
                xg,
                yg,
                z,
                levels=_contour_levels(z),
                colors=MODEL_COLORS[model_name],
                linewidths=lw,
                linestyles=ls,
                alpha=0.95,
            )
        _format_xy(ax)
        ax.set_title(CLASS_NAMES[c], weight="bold")

    handles = [
        Line2D([0], [0], color="#4a4f57", lw=1.0, label="real KDE"),
        Line2D([0], [0], color=MODEL_COLORS["VAE"], lw=1.4, ls="--", label="VAE generated KDE"),
        Line2D([0], [0], color=MODEL_COLORS["DDPM"], lw=1.6, label="DDPM generated KDE"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    _save(fig, GEN_DIR / "generation_comparison")


def figure_diffusion_process(train: np.ndarray, train_label: np.ndarray, model: DiffusionModel) -> None:
    pts = _subsample(get_class(train, train_label, 1), 700)
    x0 = torch.tensor(pts, dtype=torch.float32)
    noise = torch.randn_like(x0)
    forward_ts = [0, 200, 600, 999]
    reverse_ts = [999, 600, 200, 0]

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.7), constrained_layout=True)

    for i, t in enumerate(forward_ts):
        if t == 0:
            arr = x0.numpy()
        else:
            arr = (model._sqrt_alphas_cumprod[t] * x0 + model._sqrt_one_minus[t] * noise).numpy()
        ax = axes[0, i]
        ax.scatter(arr[:, 0], arr[:, 1], s=3.2, c="#777b84", alpha=0.42, linewidths=0, rasterized=True)
        _format_xy(ax, ticks=False)
        ax.set_title(fr"$t={t}$")

    model.denoiser.eval()
    n = 700
    x = torch.randn(n, 2)
    states = {model.n_steps - 1: x.detach().cpu().numpy()}
    wanted = set(reverse_ts)
    with torch.no_grad():
        for ti in range(model.n_steps - 1, -1, -1):
            t_tensor = torch.full((n,), ti, dtype=torch.long)
            pred = model.denoiser(x, t_tensor)
            alpha_t = model._alphas[ti]
            beta_t = model._betas[ti]
            mean = (x - beta_t / model._sqrt_one_minus[ti] * pred) / np.sqrt(alpha_t)
            if ti > 0:
                x = mean + np.sqrt(beta_t) * torch.randn_like(x)
                state_t = ti - 1
            else:
                x = mean
                state_t = 0
            if state_t in wanted:
                states[state_t] = x.detach().cpu().numpy()

    for i, t in enumerate(reverse_ts):
        arr = states[t]
        ax = axes[1, i]
        ax.scatter(arr[:, 0], arr[:, 1], s=3.2, c=MODEL_COLORS["DDPM"], alpha=0.46, linewidths=0, rasterized=True)
        _format_xy(ax, ticks=False)
        ax.set_title(fr"$t={t}$")

    axes[0, 0].set_ylabel("forward", weight="bold")
    axes[1, 0].set_ylabel("reverse", weight="bold")
    fig.suptitle("DDPM dynamics on Ring", weight="bold", y=1.03)
    _save(fig, GEN_DIR / "diffusion_process")


def figure_training_curves(models: dict[str, dict[int, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    panels = [("VAE", "ELBO loss", axes[0]), ("DDPM", "noise prediction MSE", axes[1])]
    for model_name, ylabel, ax in panels:
        for c in range(N_CLASSES):
            losses = np.asarray(models[model_name][c].train_losses, dtype=float)
            x = np.arange(1, len(losses) + 1)
            ax.plot(x, _smooth(losses), color=CLASS_COLORS[c], lw=1.5, label=f"{SHORT_NAMES[c]}  final={losses[-1]:.3f}")
        ax.set_title(f"{model_name} convergence", weight="bold")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(color="#e7eaf0", lw=0.6)
        ax.legend(frameon=False)
        sns.despine(ax=ax)
    _save(fig, GEN_DIR / "training_curves")


def _metric_arrays() -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    metrics = ["MMD", "Wasserstein", "Coverage", "Precision", "Score"]
    rows = []
    row_labels = []
    raw = []
    for cls in CLASS_NAMES:
        for model in ["VAE", "DDPM"]:
            row_labels.append(f"{SHORT_NAMES[CLASS_NAMES.index(cls)]} / {model}")
            raw.append([RESULTS[cls][model][m] for m in metrics])
    raw_arr = np.asarray(raw, dtype=float)
    good = np.zeros_like(raw_arr)
    for j, m in enumerate(metrics):
        col = raw_arr[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            good[:, j] = 0.5
        elif m in {"MMD", "Wasserstein"}:
            good[:, j] = (hi - col) / (hi - lo)
        else:
            good[:, j] = (col - lo) / (hi - lo)
    return good, row_labels, metrics, raw_arr


def figure_metrics_heatmap() -> None:
    good, row_labels, metrics, raw = _metric_arrays()
    annot = np.empty(raw.shape, dtype=object)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            annot[i, j] = f"{raw[i, j]:.4f}" if metrics[j] == "MMD" else f"{raw[i, j]:.3f}"

    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    sns.heatmap(
        good,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.55,
        linecolor="white",
        cbar_kws={"label": "normalised quality"},
    )
    ax.set_xticklabels(["MMD↓", "Wass.↓", "Coverage↑", "Precision↑", "Score↑"], rotation=0)
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_title("Metric matrix: raw value, colour = normalised quality", weight="bold")
    for y in [2, 4, 6]:
        ax.hlines(y, *ax.get_xlim(), colors="#2f343b", lw=1.0)
    _save(fig, EVAL_DIR / "metrics_heatmap")


def figure_metrics_bars() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), constrained_layout=True)

    ax = axes[0]
    ax.set_title("Precision-Coverage plane", weight="bold")
    ax.axhspan(0.9, 1.02, color="#edf7ee", alpha=0.65)
    ax.axvspan(0.9, 1.02, color="#edf7ee", alpha=0.45)
    label_offsets = {
        "Gaussian Mixture": (0.010, -0.030),
        "Ring": (0.010, 0.012),
        "Two Moons": (0.010, -0.010),
        "Spiral": (0.015, 0.006),
    }
    for c, cls in enumerate(CLASS_NAMES):
        v = RESULTS[cls]["VAE"]
        d = RESULTS[cls]["DDPM"]
        ax.scatter(v["Coverage"], v["Precision"], s=44, facecolors="white", edgecolors=MODEL_COLORS["VAE"], lw=1.5, zorder=3)
        ax.scatter(d["Coverage"], d["Precision"], s=48, color=MODEL_COLORS["DDPM"], edgecolors="white", lw=0.5, zorder=4)
        ax.annotate(
            "",
            xy=(d["Coverage"], d["Precision"]),
            xytext=(v["Coverage"], v["Precision"]),
            arrowprops=dict(arrowstyle="->", color=CLASS_COLORS[c], lw=1.0, alpha=0.75),
        )
        dx, dy = label_offsets[cls]
        ax.text(d["Coverage"] + dx, d["Precision"] + dy, SHORT_NAMES[c], color=CLASS_COLORS[c], fontsize=7)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0.42, 1.06)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Precision")
    ax.grid(color="#e7eaf0", lw=0.6)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=MODEL_COLORS["VAE"], label="VAE", markersize=6),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["DDPM"], markeredgecolor="white", label="DDPM", markersize=6),
        ],
        frameon=False,
        loc="lower right",
    )
    sns.despine(ax=ax)

    ax = axes[1]
    ax.set_title("Composite score by distribution", weight="bold")
    x = np.arange(N_CLASSES)
    width = 0.34
    vae_scores = [RESULTS[cls]["VAE"]["Score"] for cls in CLASS_NAMES]
    ddpm_scores = [RESULTS[cls]["DDPM"]["Score"] for cls in CLASS_NAMES]
    ax.bar(x - width / 2, vae_scores, width, color=MODEL_COLORS["VAE"], alpha=0.86, label="VAE")
    ax.bar(x + width / 2, ddpm_scores, width, color=MODEL_COLORS["DDPM"], alpha=0.90, label="DDPM")
    for i, (v, d) in enumerate(zip(vae_scores, ddpm_scores)):
        ax.plot([i - width / 2, i + width / 2], [v, d], color="#5d6470", lw=0.8, alpha=0.7)
        ax.text(i, max(v, d) + 0.035, f"+{d - v:.2f}", ha="center", fontsize=7, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("score")
    ax.grid(axis="y", color="#e7eaf0", lw=0.6)
    ax.legend(frameon=False, loc="upper left")
    sns.despine(ax=ax)

    _save(fig, EVAL_DIR / "metrics_bars")


def figure_ablation_sensitivity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), constrained_layout=True)

    for ax, model_name in zip(axes, ["VAE", "DDPM"]):
        data = ABLATION_RESULTS[model_name]
        x = np.arange(len(data["label"]))
        ax.bar(x, data["MMD"], width=0.48, color="#c8d7eb", edgecolor="white", label="MMD")
        ax.set_ylabel("MMD", color="#476582")
        ax.tick_params(axis="y", labelcolor="#476582")
        ax.set_xticks(x)
        ax.set_xticklabels(data["label"])
        ax.grid(axis="y", color="#e7eaf0", lw=0.6)
        sns.despine(ax=ax, right=False)

        ax2 = ax.twinx()
        ax2.plot(x, data["Coverage"], marker="o", color="#4daf4a", lw=1.8, label="Coverage")
        ax2.plot(x, data["Precision"], marker="s", color="#d05a2b", lw=1.8, label="Precision")
        ax2.set_ylim(0, 1.08)
        ax2.set_ylabel("ratio", color="#40444c")
        ax2.tick_params(axis="y", labelcolor="#40444c")

        title = r"VAE: KL weight $\beta$" if model_name == "VAE" else r"DDPM: diffusion steps $T$"
        ax.set_title(title, weight="bold")
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="center right")

    fig.suptitle("Hyperparameter sensitivity on Spiral", weight="bold", y=1.03)
    _save(fig, EVAL_DIR / "ablation_sensitivity")


def figure_conditional_metrics() -> None:
    metrics = ["MMD", "Precision", "Coverage"]
    rows = []
    row_labels = []
    raw = []
    for cls in CLASS_NAMES:
        for model in ["CVAE", "CondDDPM"]:
            rows.append((cls, model))
            row_labels.append(f"{SHORT_NAMES[CLASS_NAMES.index(cls)]} / {model}")
            raw.append([CONDITIONAL_RESULTS[cls][model][m] for m in metrics])

    raw_arr = np.asarray(raw, dtype=float)
    quality = np.zeros_like(raw_arr)
    for j, metric in enumerate(metrics):
        col = raw_arr[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            quality[:, j] = 0.5
        elif metric == "MMD":
            quality[:, j] = (hi - col) / (hi - lo)
        else:
            quality[:, j] = (col - lo) / (hi - lo)

    annot = np.empty(raw_arr.shape, dtype=object)
    for i in range(raw_arr.shape[0]):
        for j, metric in enumerate(metrics):
            annot[i, j] = f"{raw_arr[i, j]:.4f}" if metric == "MMD" else f"{raw_arr[i, j]:.3f}"

    fig = plt.figure(figsize=(7.4, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        quality,
        ax=ax0,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.55,
        linecolor="white",
        cbar=False,
    )
    ax0.set_xticklabels(["MMD↓", "Precision↑", "Coverage↑"], rotation=0)
    ax0.set_yticklabels(row_labels, rotation=0)
    ax0.set_title("conditional generation quality", weight="bold")
    for y in [2, 4, 6]:
        ax0.hlines(y, *ax0.get_xlim(), colors="#2f343b", lw=0.9)

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(N_CLASSES)
    width = 0.34
    cvae_cov = [CONDITIONAL_RESULTS[cls]["CVAE"]["Coverage"] for cls in CLASS_NAMES]
    ddpm_cov = [CONDITIONAL_RESULTS[cls]["CondDDPM"]["Coverage"] for cls in CLASS_NAMES]
    ax1.bar(x - width / 2, cvae_cov, width, color=MODEL_COLORS["VAE"], alpha=0.84, label="CVAE")
    ax1.bar(x + width / 2, ddpm_cov, width, color=MODEL_COLORS["DDPM"], alpha=0.90, label="Conditional DDPM")
    for i, (v, d) in enumerate(zip(cvae_cov, ddpm_cov)):
        ax1.text(i, max(v, d) + 0.025, f"+{d - v:.2f}", ha="center", fontsize=7, color="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SHORT_NAMES)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Coverage")
    ax1.set_title("coverage gain by conditioning", weight="bold")
    ax1.grid(axis="y", color="#e7eaf0", lw=0.6)
    ax1.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax1)

    _save(fig, EVAL_DIR / "conditional_metrics")


def figure_robustness_curves() -> None:
    rhos = np.array(sorted(ROBUSTNESS_RESULTS))
    metrics = ["MMD", "Precision", "Coverage"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9), constrained_layout=True)

    for ax, metric in zip(axes, metrics):
        for model_name, color in [("VAE", MODEL_COLORS["VAE"]), ("DDPM", MODEL_COLORS["DDPM"])]:
            vals = [ROBUSTNESS_RESULTS[int(rho)][model_name][metric] for rho in rhos]
            ax.plot(rhos, vals, marker="o", lw=1.8, color=color, label=model_name)
        ax.set_title(metric, weight="bold")
        ax.set_xlabel("contamination ratio (%)")
        if metric == "MMD":
            ax.set_ylabel("value")
        else:
            ax.set_ylim(0.78, 1.02)
            ax.set_ylabel("ratio")
        ax.set_xticks(rhos)
        ax.grid(color="#e7eaf0", lw=0.6)
        sns.despine(ax=ax)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Robustness under uniform outlier contamination on Spiral", weight="bold", y=1.05)
    _save(fig, EVAL_DIR / "robustness_curves")


def make_all() -> None:
    _set_style()
    train, train_label = load_data("train")

    print("Generating analysis figures...")
    figure_comparison_overview(train, train_label)
    figure_pca_analysis(train, train_label)
    figure_noise_structure(train, train_label)
    figure_effective_dimension(train, train_label)
    figure_polar_analysis(train, train_label)
    figure_summary_radar(train, train_label)

    print("Loading trained models...")
    models = load_models()

    print("Sampling models for generation overlays...")
    samples: dict[str, dict[int, np.ndarray]] = {"VAE": {}, "DDPM": {}}
    for c in range(N_CLASSES):
        torch.manual_seed(100 + c)
        samples["VAE"][c] = models["VAE"][c].sample(2000)
        torch.manual_seed(200 + c)
        samples["DDPM"][c] = models["DDPM"][c].sample(2000)

    print("Generating generation figures...")
    figure_generation_comparison(train, train_label, samples)
    figure_diffusion_process(train, train_label, models["DDPM"][1])
    figure_training_curves(models)

    print("Generating evaluation figures...")
    figure_metrics_heatmap()
    figure_metrics_bars()
    figure_ablation_sensitivity()
    figure_conditional_metrics()
    figure_robustness_curves()

    print(f"Done. Figures saved under {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    make_all()
