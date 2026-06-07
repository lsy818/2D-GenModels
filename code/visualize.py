"""Visualisation functions — data analysis and model comparison plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import spatial
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from code.config import (CLASS_NAMES, CLASS_COLORS, N_CLASSES,
                          AnalysisConfig, ModelConfig, FIGURES_DIR)
from code.analysis import (estimate_local_noise, compute_curvature,
                            compute_thickness, analyse_distribution)
from code.dataset import load_data, get_class_data
from code.utils import ensure_dir

# ── Matplotlib defaults ───────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.titlesize": 14, "text.usetex": False,
})

ANALYSIS_DIR = FIGURES_DIR / "analysis"
GEN_DIR = FIGURES_DIR / "generation"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA-ANALYSIS FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comprehensive_per_class(
    train: np.ndarray, train_label: np.ndarray,
    test: np.ndarray, test_label: np.ndarray,
    cfg: AnalysisConfig = None,
) -> None:
    """9-panel comprehensive figure for each distribution class."""
    cfg = cfg or AnalysisConfig()
    out_dir = ensure_dir(ANALYSIS_DIR)
    file_stems = ["gaussian_mixture", "ring", "two_moons", "spiral"]

    for c in range(N_CLASSES):
        tr = get_class_data(train, train_label, c)
        te = get_class_data(test, test_label, c)
        results = analyse_distribution(tr, cfg)

        print(f"\n  Class {c} — {CLASS_NAMES[c]}")
        print(f"    mean = ({results['mean'][0]:.3f}, {results['mean'][1]:.3f})")
        print(f"    std  = ({results['std'][0]:.3f}, {results['std'][1]:.3f})")
        print(f"    anisotropy = {results['global_anisotropy']:.3f}")
        print(f"    PCA ratio  = {results['pca_ratio']}")
        print(f"    noise mean = {results['local_noise_mean']:.4f}")
        print(f"    thickness  = {results['thickness_mean']:.4f}")

        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)
        color = CLASS_COLORS[c]

        # (0,0) scatter
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(tr[:, 0], tr[:, 1], s=3, alpha=0.5, c=color,
                   linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"{CLASS_NAMES[c]}\nScatter (Train, n={len(tr)})",
               xlabel="x", ylabel="y")
        ax.grid(alpha=0.2)

        # (0,1) KDE heatmap
        ax = fig.add_subplot(gs[0, 1])
        gd = results["density_grid"]
        im = ax.pcolormesh(gd["X"], gd["Y"], gd["Z"], cmap="hot",
                           shading="auto", rasterized=True)
        ax.scatter(tr[::cfg.subsample_vis, 0], tr[::cfg.subsample_vis, 1],
                   s=1, alpha=0.3, c="cyan", linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title="KDE Density", xlabel="x", ylabel="y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Density")

        # (0,2) train vs test
        ax = fig.add_subplot(gs[0, 2])
        ax.scatter(tr[:, 0], tr[:, 1], s=2, alpha=0.35, c=color,
                   linewidths=0, label="Train", rasterized=True)
        ax.scatter(te[:, 0], te[:, 1], s=2, alpha=0.35, c="gray",
                   marker="s", linewidths=0, label="Test", rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title="Train vs Test Overlay", xlabel="x", ylabel="y")
        ax.legend(markerscale=5)
        ax.grid(alpha=0.2)

        # (1,0) noise map
        ax = fig.add_subplot(gs[1, 0])
        sc = ax.scatter(tr[:, 0], tr[:, 1], s=5, alpha=0.7,
                        c=results["noise_map"], cmap="plasma",
                        linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title="Local Noise (k-NN)", xlabel="x", ylabel="y")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # (1,1) thickness
        ax = fig.add_subplot(gs[1, 1])
        sc = ax.scatter(tr[:, 0], tr[:, 1], s=5, alpha=0.7,
                        c=results["thickness_map"], cmap="viridis",
                        linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title="Manifold Thickness", xlabel="x", ylabel="y")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # (1,2) curvature
        ax = fig.add_subplot(gs[1, 2])
        sc = ax.scatter(tr[:, 0], tr[:, 1], s=5, alpha=0.7,
                        c=results["curvature_map"], cmap="coolwarm",
                        linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title="Local Curvature", xlabel="x", ylabel="y")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # (2,0) p(x)
        ax = fig.add_subplot(gs[2, 0])
        ax.hist(tr[:, 0], bins=60, density=True, alpha=0.7, color=color,
                edgecolor="white", linewidth=0.3, label="Train")
        ax.hist(te[:, 0], bins=60, density=True, alpha=0.4, color="gray",
                edgecolor="white", linewidth=0.3, label="Test")
        ax.set(title="p(x)", xlabel="x", ylabel="Density")
        ax.legend(fontsize=8)

        # (2,1) p(y)
        ax = fig.add_subplot(gs[2, 1])
        ax.hist(tr[:, 1], bins=60, density=True, alpha=0.7, color=color,
                edgecolor="white", linewidth=0.3)
        ax.hist(te[:, 1], bins=60, density=True, alpha=0.4, color="gray",
                edgecolor="white", linewidth=0.3)
        ax.set(title="p(y)", xlabel="y", ylabel="Density")

        # (2,2) pairwise distances
        ax = fig.add_subplot(gs[2, 2])
        sub = tr[np.random.default_rng(42).choice(
            len(tr), min(len(tr), cfg.n_subsample_pairwise), replace=False)]
        dists = spatial.distance.pdist(sub)
        ax.hist(dists, bins=80, density=True, alpha=0.7, color=color,
                edgecolor="white", linewidth=0.3)
        ax.axvline(dists.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean = {dists.mean():.3f}")
        ax.set(title="Pairwise Distances", xlabel="Distance", ylabel="Density")
        ax.legend(fontsize=8)

        fig.suptitle(f"Class {c}: {CLASS_NAMES[c]} — Data Analysis",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.savefig(out_dir / f"{file_stems[c]}_comprehensive.png", dpi=180)
        plt.close(fig)
        print(f"    Saved: {file_stems[c]}_comprehensive.png")


def plot_comparison_overview(train: np.ndarray, train_label: np.ndarray) -> None:
    """4×2 grid: scatter + KDE density for each class."""
    out = ensure_dir(ANALYSIS_DIR)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for c in range(N_CLASSES):
        pts = get_class_data(train, train_label, c)

        ax = axes[0, c]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5, c=CLASS_COLORS[c],
                   linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"{CLASS_NAMES[c]}\n(n={len(pts)})", xlabel="x", ylabel="y")
        ax.grid(alpha=0.2)

        ax = axes[1, c]
        kde = KernelDensity(bandwidth=0.2).fit(pts)
        g = np.linspace(-4, 4, 80)
        Xg, Yg = np.meshgrid(g, g)
        Z = np.exp(kde.score_samples(
            np.column_stack([Xg.ravel(), Yg.ravel()]))).reshape(Xg.shape)
        im = ax.pcolormesh(Xg, Yg, Z, cmap="hot", shading="auto", rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"KDE Density", xlabel="x", ylabel="y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Overview of Four 2D Distributions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "comparison_overview.png", dpi=180)
    plt.close(fig)
    print("  Saved: comparison_overview.png")


def plot_noise_and_structure(train: np.ndarray, train_label: np.ndarray) -> None:
    """3×4 grid: noise map, thickness map, k-NN distance distribution."""
    out = ensure_dir(ANALYSIS_DIR)
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))

    for c in range(N_CLASSES):
        pts = get_class_data(train, train_label, c)
        sub = pts[np.random.default_rng(42).choice(
            len(pts), min(len(pts), 600), replace=False)]

        noise, _ = estimate_local_noise(pts, 30)
        sc = axes[0, c].scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6,
                                c=noise, cmap="plasma", linewidths=0, rasterized=True)
        axes[0, c].set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
                       title=f"{CLASS_NAMES[c]}\nLocal Noise Map", xlabel="x", ylabel="y")
        plt.colorbar(sc, ax=axes[0, c], fraction=0.046, pad=0.04)

        thick = compute_thickness(pts, 20)
        sc = axes[1, c].scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6,
                                c=thick, cmap="viridis", linewidths=0, rasterized=True)
        axes[1, c].set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
                       title=f"Thickness (mean={thick.mean():.3f})", xlabel="x", ylabel="y")
        plt.colorbar(sc, ax=axes[1, c], fraction=0.046, pad=0.04)

        tree = spatial.cKDTree(pts)
        for k, sty, al in [(5, "dotted", 0.5), (15, "dashed", 0.6), (30, "solid", 0.7)]:
            d, _ = tree.query(sub, k=k + 1)
            axes[2, c].hist(d[:, -1], bins=40, density=True, alpha=al,
                            histtype="step", linewidth=1.5, linestyle=sty,
                            label=f"k={k}")
        axes[2, c].set(title="k-NN Distance Distribution",
                       xlabel="Distance", ylabel="Density")
        axes[2, c].legend(fontsize=7)

    fig.suptitle("Noise & Structure Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "noise_structure.png", dpi=180)
    plt.close(fig)
    print("  Saved: noise_structure.png")


def plot_pca_analysis(train: np.ndarray, train_label: np.ndarray) -> None:
    """2×4 grid: PCA directions overlaid + explained-variance bars."""
    from sklearn.decomposition import PCA
    out = ensure_dir(ANALYSIS_DIR)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for c in range(N_CLASSES):
        pts = get_class_data(train, train_label, c)
        pca = PCA().fit(pts)

        ax = axes[0, c]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, c=CLASS_COLORS[c],
                   linewidths=0, rasterized=True)
        mean = pts.mean(axis=0)
        for i, (ev, ratio) in enumerate(zip(
                pca.explained_variance_, pca.explained_variance_ratio_)):
            d = pca.components_[i] * np.sqrt(ev) * 2
            ax.arrow(*mean, *d, head_width=0.15, head_length=0.2,
                     fc=["red", "blue"][i], ec=["red", "blue"][i],
                     linewidth=2, alpha=0.8, label=f"PC{i+1} ({ratio:.1%})")
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"{CLASS_NAMES[c]}", xlabel="x", ylabel="y")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

        ax = axes[1, c]
        bars = ax.bar(["PC1", "PC2"], pca.explained_variance_ratio_,
                      color=["red", "blue"], alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set(ylim=(0, 1), title=f"Variance Ratio  "
               f"(anisotropy = {pca.explained_variance_ratio_[0]/max(pca.explained_variance_ratio_[1],1e-12):.2f})",
               ylabel="Ratio")
        for bar, val in zip(bars, pca.explained_variance_ratio_):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9)

    fig.suptitle("PCA Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "pca_analysis.png", dpi=180)
    plt.close(fig)
    print("  Saved: pca_analysis.png")


def plot_marginal_analysis(train: np.ndarray, train_label: np.ndarray,
                           test: np.ndarray, test_label: np.ndarray) -> None:
    """2×4: p(x) and p(y) marginals with KDE overlay."""
    out = ensure_dir(ANALYSIS_DIR)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for c in range(N_CLASSES):
        tr = get_class_data(train, train_label, c)
        te = get_class_data(test, test_label, c)

        for row, dim, lbl in [(0, 0, "x"), (1, 1, "y")]:
            ax = axes[row, c]
            x_rng = np.linspace(-4, 4, 200)[:, None]
            kde = KernelDensity(bandwidth=0.15).fit(tr[:, dim:dim+1])
            dens = np.exp(kde.score_samples(x_rng))
            ax.fill_between(x_rng.ravel(), dens, alpha=0.3, color=CLASS_COLORS[c])
            ax.hist(tr[:, dim], bins=50, density=True, alpha=0.5,
                    color=CLASS_COLORS[c], edgecolor="white", linewidth=0.3)
            ax.hist(te[:, dim], bins=50, density=True, alpha=0.3,
                    color="gray", edgecolor="white", linewidth=0.3)
            ax.set(title=f"{CLASS_NAMES[c]}\np({lbl})", xlabel=lbl, ylabel="Density")

    fig.suptitle("Marginal Distributions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "marginal_analysis.png", dpi=180)
    plt.close(fig)
    print("  Saved: marginal_analysis.png")


def plot_scatter_overview(train: np.ndarray, train_label: np.ndarray) -> None:
    """2×2 clean scatter plots with statistics."""
    out = ensure_dir(ANALYSIS_DIR)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for c, ax in enumerate(axes.ravel()):
        pts = get_class_data(train, train_label, c)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, c=CLASS_COLORS[c],
                   linewidths=0, rasterized=True)
        mean = pts.mean(axis=0)
        ax.scatter(*mean, s=80, c="black", marker="X", edgecolors="white",
                   linewidths=0.8, zorder=10, label=f"({mean[0]:.2f}, {mean[1]:.2f})")
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"Class {c}: {CLASS_NAMES[c]}", xlabel="x", ylabel="y")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
        text = f"σ_x={pts[:,0].std():.3f}\nσ_y={pts[:,1].std():.3f}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.suptitle("Scatter Overview with Statistics", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "scatter_overview.png", dpi=180)
    plt.close(fig)
    print("  Saved: scatter_overview.png")


def plot_polar_analysis(train: np.ndarray, train_label: np.ndarray) -> None:
    """Polar-coordinate analysis for Ring and Spiral."""
    out = ensure_dir(ANALYSIS_DIR)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, c in enumerate([1, 3]):
        pts = get_class_data(train, train_label, c)
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        theta = np.arctan2(pts[:, 1], pts[:, 0])

        ax = axes[0, idx]
        ax.scatter(theta, r, s=3, alpha=0.5, c=CLASS_COLORS[c],
                   linewidths=0, rasterized=True)
        ax.set(xlim=(-np.pi, np.pi), title=f"{CLASS_NAMES[c]}\n(r, θ)",
               xlabel="θ", ylabel="r")
        ax.grid(alpha=0.2)

        ax = axes[1, idx]
        ax.hist(r, bins=50, density=True, alpha=0.7, color=CLASS_COLORS[c],
                edgecolor="white", linewidth=0.3)
        ax.axvline(r.mean(), color="red", linestyle="--",
                   label=f"mean r = {r.mean():.3f}")
        ax.axvline(r.mean() - r.std(), color="orange", linestyle=":", linewidth=1)
        ax.axvline(r.mean() + r.std(), color="orange", linestyle=":", linewidth=1)
        ax.set(title="Radial Distribution p(r)", xlabel="r", ylabel="Density")
        ax.legend(fontsize=8)

    fig.suptitle("Polar Coordinate Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "polar_analysis.png", dpi=180)
    plt.close(fig)
    print("  Saved: polar_analysis.png")


def plot_mode_detection(train: np.ndarray, train_label: np.ndarray) -> None:
    """KDE peak finding across all classes."""
    from scipy.ndimage import maximum_filter, label as ndlabel
    out = ensure_dir(ANALYSIS_DIR)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for c, ax in enumerate(axes.ravel()):
        pts = get_class_data(train, train_label, c)
        kde = KernelDensity(bandwidth=0.2).fit(pts)
        g = np.linspace(-4, 4, 120)
        Xg, Yg = np.meshgrid(g, g)
        Z = np.exp(kde.score_samples(
            np.column_stack([Xg.ravel(), Yg.ravel()]))).reshape(Xg.shape)

        local_max = (Z == maximum_filter(Z, size=5)) & (Z > 0.005)
        _, n_modes = ndlabel(local_max)

        im = ax.pcolormesh(Xg, Yg, Z, cmap="hot", shading="auto", rasterized=True)
        ax.scatter(pts[::15, 0], pts[::15, 1], s=1, alpha=0.25,
                   c="cyan", linewidths=0, rasterized=True)

        for i in range(1, n_modes + 1):
            cy, cx = np.where((ndlabel(local_max)[0] == i) & local_max)
            ax.scatter(g[cx[0]], g[cy[0]], s=120, c="lime", marker="*",
                       edgecolors="black", linewidths=0.8, zorder=10)

        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"{CLASS_NAMES[c]}  (detected modes: {n_modes})",
               xlabel="x", ylabel="y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Mode Detection via KDE Peaks", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "mode_detection.png", dpi=180)
    plt.close(fig)
    print("  Saved: mode_detection.png")


def plot_effective_dimension(train: np.ndarray, train_label: np.ndarray) -> None:
    """Cross-scale local dimension estimation."""
    from sklearn.neighbors import NearestNeighbors
    out = ensure_dir(ANALYSIS_DIR)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    k_vals = [5, 10, 20, 40, 80, 160]

    for c, ax in enumerate(axes.ravel()):
        pts = get_class_data(train, train_label, c)
        nbrs = NearestNeighbors(n_neighbors=max(k_vals) + 1).fit(pts)
        dists, _ = nbrs.kneighbors(pts)

        est_dims = []
        for k in k_vals:
            r_k = dists[:, k - 1]
            r_2k = dists[:, min(2 * k - 1, max(k_vals) - 1)]
            ratio = np.clip(r_2k / (r_k + 1e-12), 1.01, 100)
            est_dims.append(np.log(2) / np.log(ratio).mean())

        ax.plot(k_vals, est_dims, "o-", color=CLASS_COLORS[c], linewidth=2, markersize=8,
                markerfacecolor="white")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="dim=1")
        ax.axhline(2.0, color="gray", ls="-.", alpha=0.5, label="dim=2")
        ax.set(title=CLASS_NAMES[c], xlabel="k", ylabel="Estimated Dim",
               ylim=(0.5, 2.5))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("Effective Dimensionality Across Scales", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "effective_dimension.png", dpi=180)
    plt.close(fig)
    print("  Saved: effective_dimension.png")


def plot_summary_radar(train: np.ndarray, train_label: np.ndarray,
                       cfg: AnalysisConfig = None) -> None:
    """Radar chart — 5 normalised metrics across classes."""
    cfg = cfg or AnalysisConfig()
    out = ensure_dir(ANALYSIS_DIR)

    metrics = {k: [] for k in ["noise", "curvature", "thickness",
                                "anisotropy", "spread"]}
    for c in range(N_CLASSES):
        r = analyse_distribution(get_class_data(train, train_label, c), cfg)
        metrics["noise"].append(r["local_noise_mean"])
        metrics["curvature"].append(r["curvature_mean"])
        metrics["thickness"].append(r["thickness_mean"])
        metrics["anisotropy"].append(min(r["global_anisotropy"], 10.0))
        metrics["spread"].append(r["pairwise"]["mean"])

    labels = ["Noise", "Curvature", "Thickness", "Anisotropy", "Spread"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for c in range(N_CLASSES):
        vals = [
            metrics["noise"][c] / max(metrics["noise"]),
            metrics["curvature"][c] / max(x + 1e-12 for x in metrics["curvature"]),
            metrics["thickness"][c] / max(metrics["thickness"]),
            metrics["anisotropy"][c] / max(x + 1e-12 for x in metrics["anisotropy"]),
            metrics["spread"][c] / max(metrics["spread"]),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, color=CLASS_COLORS[c],
                label=CLASS_NAMES[c])
        ax.fill(angles, vals, alpha=0.1, color=CLASS_COLORS[c])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set(ylim=(0, 1.1), title="Comparative Characteristics",
           )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(out / "summary_radar.png", dpi=180)
    plt.close(fig)
    print("  Saved: summary_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL-RELATED FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(all_models: Dict) -> None:
    """VAE and Diffusion loss curves for each class.

    Both model types use a consistent colour scheme across all four
    sub-panels: green = VAE (left y-axis), tomato-red = Diffusion
    (right y-axis).  The per-class colour is used only for the title
    accent, not for the curves themselves.
    """
    VAE_COLOR = "#1f77b4"      # muted blue
    DIFF_COLOR = "#d62728"     # red

    out = ensure_dir(GEN_DIR)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for c, ax in enumerate(axes.ravel()):
        vae_loss = all_models["VAE"][c].train_losses
        diff_loss = all_models["Diffusion"][c].train_losses

        ax.plot(vae_loss, alpha=0.7, lw=1.2, color=VAE_COLOR,
                label=f"VAE  ({len(vae_loss)} ep, final={vae_loss[-1]:.3f})")
        ax.set(title=CLASS_NAMES[c], xlabel="Epoch")
        ax.set_ylabel("VAE Loss  (ELBO)", color=VAE_COLOR)
        ax.tick_params(axis="y", labelcolor=VAE_COLOR)

        ax2 = ax.twinx()
        ax2.plot(diff_loss, alpha=0.7, lw=1.2, color=DIFF_COLOR, ls="--",
                 label=f"Diffusion  ({len(diff_loss)} ep, final={diff_loss[-1]:.4f})")
        ax2.set_ylabel("Diffusion Loss  (MSE)", color=DIFF_COLOR)
        ax2.tick_params(axis="y", labelcolor=DIFF_COLOR)

        # Unified legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle("Training Curves: VAE & Diffusion", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved: training_curves.png")


def plot_generation_comparison(all_models: Dict, train: np.ndarray,
                               train_label: np.ndarray,
                               n_samples: int = 2000) -> None:
    """Real data vs generated samples — all models, all classes."""
    out = ensure_dir(GEN_DIR)
    model_names = list(all_models.keys())
    fig, axes = plt.subplots(N_CLASSES, len(model_names) + 1,
                             figsize=(18, 14))

    for c in range(N_CLASSES):
        real = get_class_data(train, train_label, c)

        ax = axes[c, 0]
        ax.scatter(real[:, 0], real[:, 1], s=2, alpha=0.5,
                   c=CLASS_COLORS[c], linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
        if c == 0:
            ax.set_title("Real Data", fontweight="bold")
        ax.set_ylabel(CLASS_NAMES[c], fontweight="bold")
        ax.grid(alpha=0.15)

        for m_idx, m_name in enumerate(model_names):
            ax = axes[c, m_idx + 1]
            samples = all_models[m_name][c].sample(n_samples)
            ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5,
                       c=CLASS_COLORS[c], linewidths=0, rasterized=True)
            ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
            if c == 0:
                ax.set_title(m_name, fontweight="bold")
            ax.grid(alpha=0.15)

    fig.suptitle("Generated Samples vs Real Data", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "generation_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: generation_comparison.png")


def plot_density_comparison(all_models: Dict, train: np.ndarray,
                            train_label: np.ndarray,
                            n_samples: int = 2000) -> None:
    """KDE density grids — real vs generated."""
    out = ensure_dir(GEN_DIR)
    model_names = list(all_models.keys())
    fig, axes = plt.subplots(N_CLASSES, len(model_names) + 1,
                             figsize=(18, 14))
    g = np.linspace(-4, 4, 60)
    Xg, Yg = np.meshgrid(g, g)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])

    for c in range(N_CLASSES):
        real = get_class_data(train, train_label, c)

        ax = axes[c, 0]
        Z = np.exp(KernelDensity(bandwidth=0.2).fit(real).score_samples(grid))
        ax.pcolormesh(Xg, Yg, Z.reshape(Xg.shape), cmap="hot",
                      shading="auto", rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
        if c == 0:
            ax.set_title("Real", fontweight="bold")
        ax.set_ylabel(CLASS_NAMES[c], fontweight="bold")

        for m_idx, m_name in enumerate(model_names):
            ax = axes[c, m_idx + 1]
            samples = all_models[m_name][c].sample(n_samples)
            Z = np.exp(KernelDensity(bandwidth=0.2).fit(samples).score_samples(grid))
            ax.pcolormesh(Xg, Yg, Z.reshape(Xg.shape), cmap="hot",
                          shading="auto", rasterized=True)
            ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
            if c == 0:
                ax.set_title(m_name, fontweight="bold")

    fig.suptitle("Density Comparison: Real vs Generated", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "density_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: density_comparison.png")


def plot_gmm_bic(train: np.ndarray, train_label: np.ndarray) -> None:
    """BIC curves for GMM component selection."""
    out = ensure_dir(GEN_DIR)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ks = range(1, 21)

    for c, ax in enumerate(axes.ravel()):
        data = get_class_data(train, train_label, c)
        bics = []
        for k in ks:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=42, max_iter=500, n_init=5)
            gmm.fit(data)
            bics.append(gmm.bic(data))
        best_k = ks[np.argmin(bics)]
        ax.plot(list(ks), bics, "o-", color=CLASS_COLORS[c], lw=2, ms=6)
        ax.axvline(best_k, color="red", ls="--", alpha=0.7, label=f"Best K={best_k}")
        ax.set(title=CLASS_NAMES[c], xlabel="K", ylabel="BIC")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle("GMM Component Selection (BIC)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "gmm_bic.png", dpi=150)
    plt.close(fig)
    print("  Saved: gmm_bic.png")


def plot_diffusion_process(train: np.ndarray, train_label: np.ndarray,
                           diff_model=None) -> None:
    """Forward & reverse diffusion process for Spiral."""
    from code.models.diffusion_model import DiffusionModel
    out = ensure_dir(GEN_DIR)

    if diff_model is None:
        # Try loading trained model
        diff_model = DiffusionModel.load(
            Path("models/diffusion_class3.pt"), map_location="cpu")

    import torch
    device = diff_model.device
    data = torch.FloatTensor(
        get_class_data(train, train_label, 3)[:500]).to(device)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    timesteps = [0, 100, 300, 600, 999]

    for i, t_val in enumerate(timesteps):
        ax = axes[0, i]
        if t_val == 0:
            pts = data.cpu().numpy()
        else:
            sqrt_a = diff_model._sqrt_alphas_cumprod[t_val]
            sqrt_1m = diff_model._sqrt_one_minus[t_val]
            noise = torch.randn_like(data)
            pts = (sqrt_a * data + sqrt_1m * noise).cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5,
                   c="#984ea3", linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"t={t_val} (forward)")
        ax.grid(alpha=0.15)

    # Reverse
    diff_model.denoiser.eval()
    x = torch.randn(500, 2, device=device)
    for i, t_val in enumerate([999, 600, 300, 100, 0]):
        ax = axes[1, i]
        if i == 0:
            pts = x.cpu().numpy()
        else:
            for t_idx in range(diff_model.n_steps - 1, -1, -1):
                t_t = torch.full((500,), t_idx, dtype=torch.long, device=device)
                pred = diff_model.denoiser(x, t_t)
                a_t = diff_model._alphas[t_idx]
                b_t = diff_model._betas[t_idx]
                coef = b_t / diff_model._sqrt_one_minus[t_idx]
                mean = (x - coef * pred) / np.sqrt(a_t)
                x = mean + (np.sqrt(b_t) * torch.randn_like(x) if t_idx > 0 else 0)
                if t_idx == t_val:
                    pts = x.detach().cpu().numpy()
                    break
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5,
                   c="#984ea3", linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"t={t_val} (reverse)")
        ax.grid(alpha=0.15)

    fig.suptitle("Diffusion Process — Spiral", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "diffusion_process.png", dpi=150)
    plt.close(fig)
    print("  Saved: diffusion_process.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def make_all_analysis_figures(cfg: AnalysisConfig = None) -> None:
    """Run *all* data-analysis visualisations."""
    cfg = cfg or AnalysisConfig()
    train, train_label = load_data("train")
    test, test_label = load_data("test")

    print("\n[1/7] Comprehensive per-class analysis ...")
    plot_comprehensive_per_class(train, train_label, test, test_label, cfg)

    print("\n[2/7] Comparison overview ...")
    plot_comparison_overview(train, train_label)

    print("\n[3/7] Noise & structure ...")
    plot_noise_and_structure(train, train_label)

    print("\n[4/7] PCA analysis ...")
    plot_pca_analysis(train, train_label)

    print("\n[5/7] Marginal analysis + scatter + polar + mode + dimension ...")
    plot_marginal_analysis(train, train_label, test, test_label)
    plot_scatter_overview(train, train_label)
    plot_polar_analysis(train, train_label)
    plot_mode_detection(train, train_label)
    plot_effective_dimension(train, train_label)

    print("\n[6/7] Summary radar ...")
    plot_summary_radar(train, train_label, cfg)

    print(f"\nAnalysis figures saved to {ANALYSIS_DIR.resolve()}")


def make_all_generation_figures(all_models: Dict, train: np.ndarray,
                                train_label: np.ndarray,
                                n_samples: int = 2000) -> None:
    """Run all model-comparison visualisations."""
    print("\n[1/4] Training curves ...")
    plot_training_curves(all_models)

    print("\n[2/4] Generation comparison ...")
    plot_generation_comparison(all_models, train, train_label, n_samples)

    print("\n[3/4] Density comparison ...")
    plot_density_comparison(all_models, train, train_label, n_samples)

    print("\n[4/4] GMM BIC + Diffusion process ...")
    plot_gmm_bic(train, train_label)
    plot_diffusion_process(train, train_label, all_models["Diffusion"][3])

    print(f"\nGeneration figures saved to {GEN_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION FIGURES  (Task 4)
# ═══════════════════════════════════════════════════════════════════════════════

EVAL_DIR = FIGURES_DIR / "evaluation"


def plot_metrics_heatmap(metrics_table: dict, out_path: Path = None) -> None:
    """Heatmap: models (rows) × metrics (columns) for each class."""
    if out_path is None:
        out_path = ensure_dir(EVAL_DIR) / "metrics_heatmap.png"

    # metrics_table: class_name → model_type → {metric: value}
    model_names = ["KDE", "GMM", "VAE", "Diffusion"]
    metric_names = ["MMD", "Wasserstein", "Precision", "Coverage",
                    "NLL", "ModeCoverage"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14),
                              constrained_layout=True)
    for c, (cls_name, cls_data) in enumerate(metrics_table.items()):
        ax = axes[c // 2, c % 2]
        # Build matrix: models × metrics
        mat = np.zeros((len(model_names), len(metric_names)))
        for i, mn in enumerate(model_names):
            for j, met in enumerate(metric_names):
                mat[i, j] = cls_data[mn].get(met, np.nan)

        # Normalise each column (lower is better) — but not ModeCoverage (higher)
        mat_disp = mat.copy()
        for j, met in enumerate(metric_names):
            col = mat[:, j]
            if met == "ModeCoverage" or met == "Precision" or met == "Coverage":
                continue  # keep raw, higher is better
            rng = col.max() - col.min()
            if rng > 1e-8:
                mat_disp[:, j] = (col - col.min()) / rng
            else:
                mat_disp[:, j] = 0.5

        im = ax.imshow(mat_disp.T, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels(metric_names, fontsize=10)
        ax.set_title(cls_name, fontsize=13, fontweight="bold")

        # Annotate with raw values
        for i in range(len(model_names)):
            for j in range(len(metric_names)):
                val = mat[i, j]
                txt = f"{val:.3f}" if abs(val) < 10 else f"{val:.2f}"
                ax.text(i, j, txt, ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if 0.2 < mat_disp[i, j] < 0.8 else "black")

    # Colour bar — placed outside the rightmost column, sharing figure space
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Normalised  (lower → green   higher → red)", fontsize=10)

    fig.suptitle("Generation Quality Metrics", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_metrics_bars(metrics_table: dict, out_path: Path = None) -> None:
    """Grouped bar chart — one metric per subplot, all models + classes."""
    if out_path is None:
        out_path = ensure_dir(EVAL_DIR) / "metrics_bars.png"

    model_names = ["KDE", "GMM", "VAE", "Diffusion"]
    metric_names = ["MMD", "Wasserstein", "Precision", "Coverage",
                    "NLL", "ModeCoverage"]
    class_names = list(metrics_table.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    bar_width = 0.18
    x = np.arange(len(class_names))

    for m_idx, met in enumerate(metric_names):
        ax = axes[m_idx // 3, m_idx % 3]
        for i, mn in enumerate(model_names):
            vals = [metrics_table[cn][mn][met] for cn in class_names]
            offset = (i - len(model_names) / 2 + 0.5) * bar_width
            ax.bar(x + offset, vals, bar_width, label=mn,
                   color=CLASS_COLORS[i % 4], alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_title(met, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        if m_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Metrics by Class and Model", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_metrics_radar(metrics_table: dict, out_path: Path = None) -> None:
    """Radar chart per class — one polygon per model."""
    if out_path is None:
        out_path = ensure_dir(EVAL_DIR) / "metrics_radar.png"

    model_names = ["KDE", "GMM", "VAE", "Diffusion"]
    radar_metrics = ["MMD", "Wasserstein", "1-Precision", "1-Coverage", "NLL"]
    radar_labels = ["MMD↓", "Wass↓", "1-Prec↓", "1-Cov↓", "NLL↓"]
    class_names = list(metrics_table.keys())

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                              subplot_kw=dict(polar=True))

    for c, (cn, ax) in enumerate(zip(class_names, axes.ravel())):
        # Normalise across models
        raw = {}
        for mn in model_names:
            raw[mn] = [
                metrics_table[cn][mn]["MMD"],
                metrics_table[cn][mn]["Wasserstein"],
                1.0 - metrics_table[cn][mn]["Precision"],
                1.0 - metrics_table[cn][mn]["Coverage"],
                metrics_table[cn][mn]["NLL"],
            ]

        max_vals = [max(raw[mn][j] for mn in model_names) + 1e-12
                    for j in range(len(radar_metrics))]

        for mn in model_names:
            vals = [raw[mn][j] / max_vals[j] for j in range(len(radar_metrics))]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", lw=2, label=mn, markersize=5)
            ax.fill(angles, vals, alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(cn, fontsize=12, fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.suptitle("Model Comparison Radar  (lower area = better)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def print_metrics_table(metrics_table: dict) -> None:
    """Pretty-print a summary table to stdout."""
    model_names = ["KDE", "GMM", "VAE", "Diffusion"]
    metric_names = ["MMD", "Wasserstein", "Precision", "Coverage",
                    "NLL", "ModeCoverage"]

    print("\n" + "=" * 110)
    print("GENERATION QUALITY METRICS")
    print("=" * 110)

    for met in metric_names:
        print(f"\n── {met} ──")
        header = f"{'Class':<20} " + " ".join(f"{m:>14}" for m in model_names)
        print(header)
        print("-" * (20 + 14 * len(model_names)))
        for cn in metrics_table:
            vals = " ".join(f"{metrics_table[cn][mn][met]:>14.4f}"
                            for mn in model_names)
            print(f"{cn:<20} {vals}")

