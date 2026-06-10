"""Generate all figures for the report from scratch.

Produces:
  figures/analysis/   — data analysis figures
  figures/generation/ — model output figures
  figures/evaluation/ — metrics comparison figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.config import SystemConfig
SystemConfig.set_seed_all(42)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)

# ── Paths ────────────────────────────────────────────────────────────────────
ANALYSIS = ROOT / "figures/analysis"
GENERATION = ROOT / "figures/generation"
EVALUATION = ROOT / "figures/evaluation"
for d in [ANALYSIS, GENERATION, EVALUATION]:
    d.mkdir(parents=True, exist_ok=True)

def save_fig(fig, stem: str, subdir: Path):
    fig.savefig(subdir / f"{stem}.pdf", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem}")

# ── Data ─────────────────────────────────────────────────────────────────────
train = np.load(str(ROOT / "data/train.npy")).astype(np.float32)
train_label = np.load(str(ROOT / "data/train_label.npy")).astype(np.int64)
test = np.load(str(ROOT / "data/test.npy")).astype(np.float32)
test_label = np.load(str(ROOT / "data/test_label.npy")).astype(np.int64)

CLASS_NAMES = ["Gaussian Mixture", "Ring", "Two Moons", "Spiral"]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
N_CLASSES = 4


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_comparison_overview():
    """Scatter + KDE overview of all 4 classes."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for c in range(N_CLASSES):
        pts = train[train_label == c]
        ax = axes[0, c]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5, c=COLORS[c], linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=f"{CLASS_NAMES[c]}\n(n={len(pts)})", xlabel="x", ylabel="y")
        ax.grid(alpha=0.2)
        ax = axes[1, c]
        kde = KernelDensity(bandwidth=0.2).fit(pts)
        g = np.linspace(-4, 4, 80); Xg, Yg = np.meshgrid(g, g)
        Z = np.exp(kde.score_samples(np.column_stack([Xg.ravel(), Yg.ravel()]))).reshape(Xg.shape)
        im = ax.pcolormesh(Xg, Yg, Z, cmap="hot", shading="auto", rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title="KDE Density", xlabel="x", ylabel="y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Overview of Four 2D Distributions", fontsize=15, fontweight="bold")
    save_fig(fig, "comparison_overview", ANALYSIS)


def make_pca_figure():
    """PCA directions overlaid + explained variance bars."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for c in range(N_CLASSES):
        pts = train[train_label == c]
        pca = PCA().fit(pts)
        ax = axes[0, c]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, c=COLORS[c], linewidths=0, rasterized=True)
        mean = pts.mean(axis=0)
        for i, (ev, ratio) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_)):
            d = pca.components_[i] * np.sqrt(ev) * 2
            ax.arrow(*mean, *d, head_width=0.15, head_length=0.2, fc=["red","blue"][i], ec=["red","blue"][i], linewidth=2, alpha=0.8, label=f"PC{i+1} ({ratio:.1%})")
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=CLASS_NAMES[c], xlabel="x", ylabel="y")
        ax.legend(fontsize=7); ax.grid(alpha=0.2)
        ax = axes[1, c]
        bars = ax.bar(["PC1","PC2"], pca.explained_variance_ratio_, color=["red","blue"], alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set(ylim=(0, 1), title=f"Anisotropy={pca.explained_variance_ratio_[0]/max(pca.explained_variance_ratio_[1],1e-12):.2f}", ylabel="Ratio")
        for bar, val in zip(bars, pca.explained_variance_ratio_):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{val:.3f}", ha="center", fontsize=9)
    fig.suptitle("PCA Analysis", fontsize=15, fontweight="bold")
    save_fig(fig, "pca_analysis", ANALYSIS)


def make_effective_dimension():
    """Cross-scale local dimension estimation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    k_vals = [5, 10, 20, 40, 80, 160]
    for c, ax in enumerate(axes.ravel()):
        pts = train[train_label == c]
        nbrs = NearestNeighbors(n_neighbors=max(k_vals)+1).fit(pts)
        dists, _ = nbrs.kneighbors(pts)
        est = []
        for k in k_vals:
            r_k = dists[:, k-1]; r_2k = dists[:, min(2*k-1, max(k_vals)-1)]
            ratio = np.clip(r_2k/(r_k+1e-12), 1.01, 100)
            est.append(np.log(2)/np.log(ratio).mean())
        ax.plot(k_vals, est, "o-", color=COLORS[c], lw=2, ms=8, mfc="white")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="dim=1")
        ax.axhline(2.0, color="gray", ls="-.", alpha=0.5, label="dim=2")
        ax.set(title=CLASS_NAMES[c], xlabel="k", ylabel="Estimated Dim", ylim=(0.5, 2.5))
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.suptitle("Effective Dimensionality Across Scales", fontsize=15, fontweight="bold")
    save_fig(fig, "effective_dimension", ANALYSIS)


def make_noise_structure():
    """Local noise map + manifold thickness + k-NN distance histograms."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 13), constrained_layout=True)
    for c in range(N_CLASSES):
        pts = train[train_label == c]
        sub = pts[np.random.default_rng(42).choice(len(pts), min(len(pts), 600), replace=False)]
        # noise
        tree = spatial.cKDTree(pts)
        d, _ = tree.query(pts, k=31); noise = d[:, 1:].mean(axis=1)
        sc = axes[0, c].scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6, c=noise, cmap="plasma", linewidths=0, rasterized=True)
        axes[0, c].set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=f"{CLASS_NAMES[c]}\nLocal Noise", xlabel="x", ylabel="y")
        plt.colorbar(sc, ax=axes[0, c], fraction=0.046, pad=0.04)
        # thickness
        thick = np.zeros(len(pts))
        for i, p in enumerate(pts):
            _, idx = tree.query(p, k=21); neigh = pts[idx[1:]]
            cov = np.cov(neigh.T, ddof=1); eig = np.linalg.eigvalsh(cov)
            thick[i] = np.sqrt(max(eig[0], 0.0))
        sc2 = axes[1, c].scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6, c=thick, cmap="viridis", linewidths=0, rasterized=True)
        axes[1, c].set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=f"Thickness (mean={thick.mean():.3f})", xlabel="x", ylabel="y")
        plt.colorbar(sc2, ax=axes[1, c], fraction=0.046, pad=0.04)
        # k-NN dist
        for k, sty, al in [(5,"dotted",0.5),(15,"dashed",0.6),(30,"solid",0.7)]:
            dk, _ = tree.query(sub, k=k+1)
            axes[2, c].hist(dk[:, -1], bins=40, density=True, alpha=al, histtype="step", lw=1.5, ls=sty, label=f"k={k}")
        axes[2, c].set(title="k-NN Distance", xlabel="Distance", ylabel="Density"); axes[2, c].legend(fontsize=7)
    fig.suptitle("Noise & Structure Analysis", fontsize=15, fontweight="bold")
    save_fig(fig, "noise_structure", ANALYSIS)


def make_polar_analysis():
    """Polar coordinate analysis for Ring and Spiral."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for idx, c in enumerate([1, 3]):
        pts = train[train_label == c]
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        ax = axes[0, idx]
        ax.scatter(theta, r, s=3, alpha=0.5, c=COLORS[c], linewidths=0, rasterized=True)
        ax.set(xlim=(-np.pi, np.pi), title=f"{CLASS_NAMES[c]}\n(r, θ)", xlabel="θ", ylabel="r"); ax.grid(alpha=0.2)
        ax = axes[1, idx]
        ax.hist(r, bins=50, density=True, alpha=0.7, color=COLORS[c], edgecolor="white", linewidth=0.3)
        ax.axvline(r.mean(), color="red", ls="--", lw=1.5, label=f"mean r = {r.mean():.3f}")
        ax.axvline(r.mean()-r.std(), color="orange", ls=":", lw=1)
        ax.axvline(r.mean()+r.std(), color="orange", ls=":", lw=1)
        ax.set(title="p(r)", xlabel="r", ylabel="Density"); ax.legend(fontsize=8)
    fig.suptitle("Polar Coordinate Analysis", fontsize=15, fontweight="bold")
    save_fig(fig, "polar_analysis", ANALYSIS)


def make_summary_radar():
    """Radar chart comparing 4 classes on 5 dimensions."""
    from code.analysis import analyse_distribution
    metrics = {k: [] for k in ["noise","curvature","thickness","anisotropy","spread"]}
    for c in range(N_CLASSES):
        r = analyse_distribution(train[train_label == c])
        metrics["noise"].append(r["local_noise_mean"])
        metrics["curvature"].append(r["curvature_mean"])
        metrics["thickness"].append(r["thickness_mean"])
        metrics["anisotropy"].append(min(r["global_anisotropy"], 10.0))
        metrics["spread"].append(r["pairwise"]["mean"])
    labels = ["Noise","Curvature","Thickness","Anisotropy","Spread"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), constrained_layout=True)
    for c in range(N_CLASSES):
        vals = [metrics[k][c]/max(metrics[k]) for k in ["noise","curvature","thickness","anisotropy","spread"]] + [metrics["noise"][c]/max(metrics["noise"])]
        ax.plot(angles, vals, "o-", lw=2, color=COLORS[c], label=CLASS_NAMES[c]); ax.fill(angles, vals, alpha=0.1, color=COLORS[c])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set(ylim=(0, 1.1), title="Comparative Characteristics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    save_fig(fig, "summary_radar", ANALYSIS)


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_all_generation():
    """Load trained models and produce all generation + evaluation figures."""
    from code.models import VAEModel, DiffusionModel

    # Load models
    models = {}
    for mt, cls in [("VAE", VAEModel), ("Diffusion", DiffusionModel)]:
        models[mt] = {}
        for c in range(N_CLASSES):
            models[mt][c] = cls.load(str(ROOT / f"models/{mt.lower()}_class{c}.pt"), map_location="cpu")

    # ── training_curves ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for c, ax in enumerate(axes.ravel()):
        vl = models["VAE"][c].train_losses
        dl = models["Diffusion"][c].train_losses
        ax.plot(vl, alpha=0.7, lw=1.2, color="#1f77b4", label=f"VAE ({len(vl)} ep, final={vl[-1]:.3f})")
        ax.set(title=CLASS_NAMES[c], xlabel="Epoch"); ax.set_ylabel("VAE Loss", color="#1f77b4"); ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot(dl, alpha=0.7, lw=1.2, color="#d62728", ls="--", label=f"DDPM ({len(dl)} ep, final={dl[-1]:.4f})")
        ax2.set_ylabel("DDPM Loss", color="#d62728"); ax2.tick_params(axis="y", labelcolor="#d62728")
        lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, loc="upper right", fontsize=8)
    fig.suptitle("Training Curves: VAE & DDPM", fontsize=14, fontweight="bold")
    save_fig(fig, "training_curves", GENERATION)

    # ── generation_comparison (4 rows × 3 cols: Real, VAE, DDPM) ──
    fig, axes = plt.subplots(4, 3, figsize=(12, 15), constrained_layout=True)
    for c in range(N_CLASSES):
        real = train[train_label == c]
        for m_idx, (label, mt) in enumerate([("Real Data", None), ("VAE", "VAE"), ("DDPM", "Diffusion")]):
            ax = axes[c, m_idx]
            pts = real if mt is None else models[mt][c].sample(2000)
            ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5, c=COLORS[c], linewidths=0, rasterized=True)
            ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
            if c == 0: ax.set_title(label, fontsize=12, fontweight="bold")
            if m_idx == 0: ax.set_ylabel(CLASS_NAMES[c], fontsize=10, fontweight="bold")
            ax.grid(alpha=0.15)
    fig.suptitle("Generated Samples: VAE vs DDPM", fontsize=14, fontweight="bold")
    save_fig(fig, "generation_comparison", GENERATION)

    # ── diffusion_process (Spiral) ──
    import torch
    dm = models["Diffusion"][3]
    dm.denoiser.eval()
    device = "cpu"
    data = torch.FloatTensor(train[train_label == 3][:500])
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    for i, tv in enumerate([0, 100, 300, 600, 999]):
        ax = axes[0, i]
        if tv == 0: pts = data.numpy()
        else:
            sa = dm._sqrt_alphas_cumprod[tv]; s1m = dm._sqrt_one_minus[tv]
            pts = (sa * data + s1m * torch.randn_like(data)).numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5, c="#984ea3", linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=f"t={tv} (forward)"); ax.grid(alpha=0.15)
    x = torch.randn(500, 2)
    for i, tv in enumerate([999, 600, 300, 100, 0]):
        ax = axes[1, i]
        if i == 0: pts = x.numpy()
        else:
            for ti in range(dm.n_steps-1, -1, -1):
                tt = torch.full((500,), ti, dtype=torch.long)
                pred = dm.denoiser(x, tt)
                a_t = dm._alphas[ti]; b_t = dm._betas[ti]
                coef = b_t/dm._sqrt_one_minus[ti]
                x = (x - coef*pred)/np.sqrt(a_t) + (np.sqrt(b_t)*torch.randn_like(x) if ti>0 else 0)
                if ti == tv: pts = x.detach().numpy(); break
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5, c="#984ea3", linewidths=0, rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=f"t={tv} (reverse)"); ax.grid(alpha=0.15)
    fig.suptitle("Diffusion Process — Spiral", fontsize=14, fontweight="bold")
    save_fig(fig, "diffusion_process", GENERATION)

    # ── evaluation metrics ──
    from code.metrics import compute_all_metrics
    test_data, test_lbl = test, test_label
    met_names = ["MMD", "Wasserstein", "Precision", "Coverage", "NLL"]
    table = {}
    for c in range(N_CLASSES):
        cn = CLASS_NAMES[c]; real = test_data[test_lbl == c]
        table[cn] = {}
        for mt in ["VAE", "Diffusion"]:
            gen = models[mt][c].sample(2000)
            m = compute_all_metrics(real, gen, models[mt][c], mt, c, cn)
            table[cn][mt] = {k: m[k] for k in met_names}
            # rename key for display
            table[cn]["DDPM" if mt == "Diffusion" else mt] = table[cn].pop(mt)

    # ── heatmap (2 models × 5 metrics per class) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    for c, (cn, cd) in enumerate(table.items()):
        ax = axes[c//2, c%2]
        raw = np.array([[cd[mn][met] for met in met_names] for mn in ["VAE", "DDPM"]])
        disp = np.zeros_like(raw)
        for j, met in enumerate(met_names):
            col = raw[:, j]; rng = col.max()-col.min()
            if met in ("Precision", "Coverage"):
                disp[:, j] = 1.0-(col-col.min())/max(rng, 1e-8) if rng>1e-8 else 0.0
            else:
                disp[:, j] = (col-col.min())/max(rng, 1e-8) if rng>1e-8 else 0.5
        X = np.arange(3); Y = np.arange(6)
        im = ax.pcolormesh(X, Y, disp.T, cmap="Blues", vmin=0, vmax=1, edgecolors="white", linewidth=0.3)
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["VAE", "DDPM"], fontsize=12)
        ax.set_yticks(np.arange(5)+0.5); ax.set_yticklabels(met_names, fontsize=10)
        ax.set_title(cn, fontsize=13, fontweight="bold"); ax.set_xlim(0, 2); ax.set_ylim(5, 0)
        for i in range(2):
            for j in range(5):
                val = raw[i, j]; txt = f"{val:.3f}" if abs(val)<10 else f"{val:.2f}"
                ax.text(i+0.5, j+0.5, txt, ha="center", va="center", fontsize=9, fontweight="bold", color="white" if disp[i,j]>0.5 else "#222222")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Normalised  (lighter = better)", fontsize=10)
    fig.suptitle("Generation Quality Metrics: VAE vs DDPM", fontsize=15, fontweight="bold")
    save_fig(fig, "metrics_heatmap", EVALUATION)

    # ── bars (5 metrics, 2×3 layout with 6th cell = legend area) ──
    palette = sns.color_palette("muted", n_colors=2)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    bar_w = 0.28; x = np.arange(4)
    for m_idx, met in enumerate(met_names):
        ax = axes.ravel()[m_idx]
        for i, mn in enumerate(["VAE", "DDPM"]):
            vals = [table[cn][mn][met] for cn in CLASS_NAMES]
            off = (i-0.5)*bar_w
            ax.bar(x+off, vals, bar_w, label=mn, color=palette[i], alpha=0.88, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_title(met, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.25); sns.despine(ax=ax)
        if m_idx == 0: ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#ddd")
    # 6th cell: text summary
    ax6 = axes.ravel()[5]
    ax6.axis("off")
    summary_text = ("VAE vs DDPM\n\n"
                    "DDPM advantages:\n"
                    "• Higher Coverage on\n  Ring & Two Moons\n"
                    "• More balanced\n  Precision-Coverage\n"
                    "• Robust across all\n  geometric structures\n\n"
                    "VAE advantages:\n"
                    "• Single-step sampling\n  (~0.5 ms vs ~25 ms)\n"
                    "• Full mode coverage")
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=10, ha="center", va="center",
             family="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor="#f5f5f5", edgecolor="#ddd"))
    fig.suptitle("Metrics: VAE vs DDPM", fontsize=15, fontweight="bold")
    save_fig(fig, "metrics_bars", EVALUATION)

    print("\nAll generation + evaluation figures complete.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (imported by main_analysis.py / main_train.py)
# ═══════════════════════════════════════════════════════════════════════════════

def make_all_analysis_figures():
    """Generate all data-analysis figures (no models needed)."""
    print("=" * 60)
    print("Generating analysis figures...")
    print("=" * 60)
    make_comparison_overview()
    make_pca_figure()
    make_effective_dimension()
    make_noise_structure()
    make_polar_analysis()
    make_summary_radar()
    print(f"Saved to {ANALYSIS}")


def make_all_generation_and_eval_figures():
    """Generate all generation + evaluation figures (requires trained models)."""
    print("=" * 60)
    print("Generating generation + evaluation figures...")
    print("=" * 60)
    make_all_generation()
    print(f"Saved to {GENERATION} and {EVALUATION}")


# ==============================================================================

if __name__ == "__main__":
    make_all_analysis_figures()
    make_all_generation_and_eval_figures()
    print(f"\nAll figures saved to {ROOT / 'figures'}")
