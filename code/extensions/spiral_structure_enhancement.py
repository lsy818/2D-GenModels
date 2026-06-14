"""Spiral structural-prior enhancement experiments.

This script compares three ways to generate the Spiral class:

1. The best diagnostic beta-VAE checkpoint, as a VAE reference.
2. The original DDPM checkpoint trained in Cartesian coordinates.
3. A DDPM trained in fitted spiral coordinates.
4. A spiral-aware residual bootstrap generator.

The last two methods test whether making the arm/phase/local-noise structure
explicit fixes the failure mode observed in plain VAE/DDPM samples.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import spatial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.config import CLASS_COLORS, DATA_DIR, FIGURES_DIR, MODEL_DIR, VAEConfig, get_device, set_seed
from code.extensions.diffusion_spiral_experiments import DDPMConfig, SpiralDDPM
from code.metrics import coverage_precision, mmd_rbf
from code.models.diffusion_model import DiffusionModel
from code.models.vae_model import VAE, VAEModel


DEVICE = get_device()
OUT_DIR = FIGURES_DIR / "extensions"
MODEL_OUT = MODEL_DIR / "extensions"


@dataclass
class Projection:
    arm: np.ndarray
    t: np.ndarray
    center: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    tangent_resid: np.ndarray
    normal_resid: np.ndarray
    distance: np.ndarray


@dataclass
class SpiralFit:
    alpha: float
    t_min: float
    t_max: float
    normal_scale: float
    tangent_scale: float
    arm_probs: np.ndarray
    residual_pairs: np.ndarray

    def center(self, t: np.ndarray, arm: np.ndarray) -> np.ndarray:
        phase = arm.astype(np.float64) * np.pi
        angle = t + phase
        r = self.alpha * t
        return np.column_stack([r * np.cos(angle), r * np.sin(angle)])

    def frame(self, t: np.ndarray, arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phase = arm.astype(np.float64) * np.pi
        angle = t + phase
        deriv = self.alpha * np.column_stack(
            [np.cos(angle) - t * np.sin(angle), np.sin(angle) + t * np.cos(angle)]
        )
        tangent = deriv / np.linalg.norm(deriv, axis=1, keepdims=True).clip(1e-12)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        return tangent, normal

    def to_xy(
        self,
        arm: np.ndarray,
        t: np.ndarray,
        tangent_resid: np.ndarray | None = None,
        normal_resid: np.ndarray | None = None,
    ) -> np.ndarray:
        if tangent_resid is None:
            tangent_resid = np.zeros_like(t)
        if normal_resid is None:
            normal_resid = np.zeros_like(t)
        center = self.center(t, arm)
        tangent, normal = self.frame(t, arm)
        xy = center + tangent * tangent_resid[:, None] + normal * normal_resid[:, None]
        return xy.astype(np.float32)

    def project(self, points: np.ndarray, grid_size: int = 6000) -> Projection:
        t_grid = np.linspace(self.t_min, self.t_max, grid_size // 2)
        arms = np.concatenate(
            [
                np.zeros_like(t_grid, dtype=np.int64),
                np.ones_like(t_grid, dtype=np.int64),
            ]
        )
        t_values = np.concatenate([t_grid, t_grid])
        refs = self.center(t_values, arms)
        tree = spatial.cKDTree(refs)
        dist, idx = tree.query(points, k=1)
        arm = arms[idx]
        t = t_values[idx]
        center = refs[idx]
        tangent, normal = self.frame(t, arm)
        resid = points - center
        tangent_resid = np.sum(resid * tangent, axis=1)
        normal_resid = np.sum(resid * normal, axis=1)
        return Projection(
            arm=arm,
            t=t,
            center=center,
            tangent=tangent,
            normal=normal,
            tangent_resid=tangent_resid,
            normal_resid=normal_resid,
            distance=dist,
        )

    def to_ddpm_coords(self, points: np.ndarray) -> np.ndarray:
        proj = self.project(points)
        phase = 2.0 * (proj.t - self.t_min) / (self.t_max - self.t_min) - 1.0
        normal = proj.normal_resid / self.normal_scale
        coords = np.column_stack([phase, normal])
        return coords.astype(np.float32)

    def ddpm_coords_to_xy(
        self, coords: np.ndarray, rng: np.random.Generator, arm: np.ndarray | None = None
    ) -> np.ndarray:
        phase = np.clip(coords[:, 0], -1.0, 1.0)
        normal = np.clip(coords[:, 1], -4.0, 4.0) * self.normal_scale
        t = self.t_min + 0.5 * (phase + 1.0) * (self.t_max - self.t_min)
        if arm is None:
            arm = rng.choice(2, size=len(coords), p=self.arm_probs)
        tangent = np.zeros(len(coords), dtype=np.float64)
        return self.to_xy(arm.astype(np.int64), t, tangent, normal)


@dataclass
class Result:
    name: str
    mmd: float
    coverage: float
    precision: float
    support_off: float
    curve_mean: float
    curve_off: float
    phase_coverage: float
    phase_js: float
    arm_error: float
    sample_time_ms: float


def _reference_points(alpha: float, t_min: float, t_max: float, n: int = 5000) -> np.ndarray:
    t = np.linspace(t_min, t_max, n // 2)
    r = alpha * t
    arm0 = np.column_stack([r * np.cos(t), r * np.sin(t)])
    arm1 = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)])
    return np.concatenate([arm0, arm1], axis=0)


def fit_spiral(points: np.ndarray) -> SpiralFit:
    t_min0 = 0.20
    t_max0 = 4.0 * np.pi + 0.12
    best_alpha = 0.22
    best_score = float("inf")
    for alpha in np.linspace(0.16, 0.28, 121):
        refs = _reference_points(float(alpha), t_min0, t_max0, n=5000)
        dist, _ = spatial.cKDTree(refs).query(points, k=1)
        score = float(np.mean(np.sort(dist)[: int(0.95 * len(dist))]))
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)

    provisional = SpiralFit(
        alpha=best_alpha,
        t_min=t_min0,
        t_max=t_max0,
        normal_scale=1.0,
        tangent_scale=1.0,
        arm_probs=np.array([0.5, 0.5], dtype=np.float64),
        residual_pairs=np.zeros((1, 2), dtype=np.float64),
    )
    proj = provisional.project(points)
    t_min = max(0.05, float(np.quantile(proj.t, 0.005)) - 0.04)
    t_max = min(4.0 * np.pi + 0.3, float(np.quantile(proj.t, 0.995)) + 0.04)

    fitted = SpiralFit(
        alpha=best_alpha,
        t_min=t_min,
        t_max=t_max,
        normal_scale=1.0,
        tangent_scale=1.0,
        arm_probs=np.array([0.5, 0.5], dtype=np.float64),
        residual_pairs=np.zeros((1, 2), dtype=np.float64),
    )
    proj = fitted.project(points)
    arm_counts = np.bincount(proj.arm, minlength=2).astype(np.float64)
    arm_probs = arm_counts / arm_counts.sum()
    normal_scale = float(max(np.std(proj.normal_resid), 1e-3))
    tangent_scale = float(max(np.std(proj.tangent_resid), 1e-3))
    residual_pairs = np.column_stack([proj.tangent_resid, proj.normal_resid]).astype(np.float64)

    return SpiralFit(
        alpha=best_alpha,
        t_min=t_min,
        t_max=t_max,
        normal_scale=normal_scale,
        tangent_scale=tangent_scale,
        arm_probs=arm_probs,
        residual_pairs=residual_pairs,
    )


def sample_spiral_prior(fit: SpiralFit, n: int, rng: np.random.Generator) -> np.ndarray:
    arm = rng.choice(2, size=n, p=fit.arm_probs)
    t = rng.uniform(fit.t_min, fit.t_max, size=n)
    resid_idx = rng.integers(0, len(fit.residual_pairs), size=n)
    residuals = fit.residual_pairs[resid_idx]
    return fit.to_xy(arm.astype(np.int64), t, residuals[:, 0], residuals[:, 1])


def _load_vae_reference(n: int) -> np.ndarray:
    path = MODEL_OUT / "vae_beta_sweep_b0p3_class3.pt"
    if path.exists():
        cfg = VAEConfig()
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = VAE(input_dim=2, hidden_dims=cfg.hidden_dims, latent_dim=cfg.latent_dim).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        torch.manual_seed(8300)
        with torch.no_grad():
            z = torch.randn(n, cfg.latent_dim, device=DEVICE)
            return model.decoder(z).cpu().numpy().astype(np.float32)

    model = VAEModel.load(MODEL_DIR / "vae_class3.pt", map_location=DEVICE)
    return model.sample(n)


def _load_ddpm_reference(n: int) -> tuple[np.ndarray, float]:
    import time

    model = DiffusionModel.load(MODEL_DIR / "diffusion_class3.pt", map_location=DEVICE)
    start = time.perf_counter()
    sample = model.sample(n)
    return sample, (time.perf_counter() - start) * 1000.0


def _load_or_train_coord_ddpm(fit: SpiralFit, train: np.ndarray) -> SpiralDDPM:
    path = MODEL_OUT / "spiral_coord_ddpm_T100.pt"
    cfg = DDPMConfig(
        name="coord_T100",
        n_steps=100,
        schedule="linear",
        hidden_dim=160,
        num_layers=3,
        feature_mode="xy",
        epochs=700,
        lr=2e-4,
        batch_size=256,
    )
    if path.exists():
        print(f"  loading Coord-DDPM: {path.name}")
        return SpiralDDPM.load(path)
    print("  training Coord-DDPM in fitted spiral coordinates")
    coords = fit.to_ddpm_coords(train)
    model = SpiralDDPM(cfg).fit(coords)
    model.save(path)
    return model


def _support_off(real: np.ndarray, gen: np.ndarray) -> float:
    tree = spatial.cKDTree(real)
    knn, _ = tree.query(real, k=6)
    threshold = float(np.percentile(knn[:, -1], 95))
    dist, _ = tree.query(gen, k=1)
    return float(np.mean(dist > threshold))


def _phase_hist(fit: SpiralFit, points: np.ndarray, bins: int = 28) -> np.ndarray:
    proj = fit.project(points)
    edges = np.linspace(fit.t_min, fit.t_max, bins + 1)
    hist = np.zeros((2, bins), dtype=np.float64)
    for arm in (0, 1):
        hist[arm], _ = np.histogram(proj.t[proj.arm == arm], bins=edges)
    return hist


def _phase_coverage_and_js(fit: SpiralFit, real: np.ndarray, gen: np.ndarray) -> tuple[float, float]:
    real_hist = _phase_hist(fit, real)
    gen_hist = _phase_hist(fit, gen)
    valid = real_hist >= max(2, int(0.0015 * len(real)))
    covered = np.logical_and(valid, gen_hist > 0)
    phase_coverage = float(covered.sum() / max(valid.sum(), 1))

    eps = 1e-12
    p = (real_hist.ravel() + eps) / (real_hist.sum() + eps * real_hist.size)
    q = (gen_hist.ravel() + eps) / (gen_hist.sum() + eps * gen_hist.size)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return phase_coverage, float(js)


def _curve_metrics(fit: SpiralFit, real: np.ndarray, gen: np.ndarray) -> tuple[float, float]:
    real_proj = fit.project(real)
    gen_proj = fit.project(gen)
    threshold = float(np.percentile(real_proj.distance, 95))
    return float(np.mean(gen_proj.distance)), float(np.mean(gen_proj.distance > threshold))


def _arm_error(fit: SpiralFit, real: np.ndarray, gen: np.ndarray) -> float:
    real_proj = fit.project(real)
    gen_proj = fit.project(gen)
    real_p = np.bincount(real_proj.arm, minlength=2).astype(np.float64)
    gen_p = np.bincount(gen_proj.arm, minlength=2).astype(np.float64)
    real_p /= real_p.sum()
    gen_p /= gen_p.sum()
    return float(np.abs(real_p - gen_p).sum() / 2.0)


def evaluate(name: str, fit: SpiralFit, real: np.ndarray, gen: np.ndarray, sample_time_ms: float) -> Result:
    cp = coverage_precision(real, gen, k=5, max_subsample=1500)
    curve_mean, curve_off = _curve_metrics(fit, real, gen)
    phase_coverage, phase_js = _phase_coverage_and_js(fit, real, gen)
    return Result(
        name=name,
        mmd=mmd_rbf(real, gen, max_subsample=1500),
        coverage=cp["coverage"],
        precision=cp["precision"],
        support_off=_support_off(real, gen),
        curve_mean=curve_mean,
        curve_off=curve_off,
        phase_coverage=phase_coverage,
        phase_js=phase_js,
        arm_error=_arm_error(fit, real, gen),
        sample_time_ms=sample_time_ms,
    )


def run_experiment() -> tuple[SpiralFit, list[Result], dict[str, np.ndarray], np.ndarray]:
    import time

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    rng = np.random.default_rng(42)

    train = np.load(DATA_DIR / "train.npy").astype(np.float32)
    train_label = np.load(DATA_DIR / "train_label.npy").astype(np.int64)
    test = np.load(DATA_DIR / "test.npy").astype(np.float32)
    test_label = np.load(DATA_DIR / "test_label.npy").astype(np.int64)
    spiral_train = train[train_label == 3]
    spiral_test = test[test_label == 3]

    print(f"Device: {DEVICE}")
    print("Fitting spiral structural coordinates")
    fit = fit_spiral(spiral_train)
    print(
        f"  alpha={fit.alpha:.4f}, t=[{fit.t_min:.3f}, {fit.t_max:.3f}], "
        f"normal_scale={fit.normal_scale:.4f}, arm_probs={fit.arm_probs}"
    )

    samples: dict[str, np.ndarray] = {"Data": spiral_test}
    results: list[Result] = []

    print("Sampling VAE beta=0.3 reference")
    start = time.perf_counter()
    samples["VAE beta=0.3"] = _load_vae_reference(len(spiral_test))
    results.append(
        evaluate(
            "VAE beta=0.3",
            fit,
            spiral_test,
            samples["VAE beta=0.3"],
            (time.perf_counter() - start) * 1000.0,
        )
    )

    print("Sampling DDPM baseline reference")
    samples["DDPM baseline"], ddpm_time = _load_ddpm_reference(len(spiral_test))
    results.append(evaluate("DDPM baseline", fit, spiral_test, samples["DDPM baseline"], ddpm_time))

    coord_model = _load_or_train_coord_ddpm(fit, spiral_train)
    coords, coord_time = coord_model.sample(len(spiral_test), seed=4242)
    samples["Spiral-coordinate DDPM"] = fit.ddpm_coords_to_xy(coords, rng)
    results.append(
        evaluate(
            "Spiral-coordinate DDPM",
            fit,
            spiral_test,
            samples["Spiral-coordinate DDPM"],
            coord_time,
        )
    )

    print("Sampling spiral structural prior")
    start = time.perf_counter()
    samples["Spiral structural prior"] = sample_spiral_prior(fit, len(spiral_test), rng)
    prior_time = (time.perf_counter() - start) * 1000.0
    results.append(
        evaluate(
            "Spiral structural prior",
            fit,
            spiral_test,
            samples["Spiral structural prior"],
            prior_time,
        )
    )

    for row in results:
        print(
            f"  {row.name}: MMD={row.mmd:.4f}, Cov={row.coverage:.3f}, "
            f"Prec={row.precision:.3f}, CurveOff={row.curve_off:.3f}, "
            f"PhaseJS={row.phase_js:.4f}"
        )

    return fit, results, samples, spiral_test


def _format_xy(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.25, 3.25)
    ax.set_ylim(-3.25, 3.25)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.grid(color="#e7eaf0", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#ccd1dc")
        spine.set_linewidth(0.7)


def _subsample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[rng.choice(len(points), n, replace=False)]


def _plot_centerline(ax: plt.Axes, fit: SpiralFit) -> None:
    t = np.linspace(fit.t_min, fit.t_max, 600)
    for arm in (0, 1):
        xy = fit.center(t, np.full(len(t), arm, dtype=np.int64))
        ax.plot(xy[:, 0], xy[:, 1], color="#111827", linewidth=0.8, alpha=0.75, zorder=3)


def figure_samples(fit: SpiralFit, samples: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(7)
    keys = ["Data", "VAE beta=0.3", "DDPM baseline", "Spiral-coordinate DDPM", "Spiral structural prior"]
    titles = ["Data", "VAE beta=0.3", "DDPM", "Coord-DDPM", "Structural prior"]
    fig, axes = plt.subplots(1, len(keys), figsize=(12.2, 2.9), constrained_layout=True)
    for ax, key, title in zip(axes, keys, titles):
        pts = _subsample(samples[key], 1300, rng)
        color = "#30343b" if key == "Data" else CLASS_COLORS[3]
        ax.scatter(pts[:, 0], pts[:, 1], s=4.4, c=color, alpha=0.56, linewidths=0, rasterized=True)
        _plot_centerline(ax, fit)
        ax.set_title(title, fontsize=10.5, weight="bold")
        _format_xy(ax)
    fig.suptitle("Spiral structure enhancement: samples with fitted centerline", fontsize=12, weight="bold")
    fig.savefig(OUT_DIR / "spiral_structure_samples.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_metrics(results: list[Result]) -> None:
    labels = ["VAE", "DDPM", "Coord", "Prior"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0), constrained_layout=True)
    panels = [
        ("curve_off", "Off spiral curve", False),
        ("phase_js", "Phase JS", False),
        ("coverage", "Coverage", True),
        ("mmd", "MMD", False),
    ]
    for ax, (attr, title, higher_better) in zip(axes, panels):
        vals = [getattr(r, attr) for r in results]
        colors = ["#9aa1ad", "#687386", "#4c78a8", "#984ea3"]
        ax.bar(x, vals, color=colors, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title, weight="bold", fontsize=10.5)
        if higher_better:
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e7eaf0", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("#ccd1dc")
            spine.set_linewidth(0.7)
    fig.suptitle("Spiral structure metrics", fontsize=12, weight="bold")
    fig.savefig(OUT_DIR / "spiral_structure_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fit, results, samples, _ = run_experiment()
    figure_samples(fit, samples)
    figure_metrics(results)
    print(f"\nSaved figures to {OUT_DIR}")
    print(f"Computed {len(results)} Spiral structure rows.")


if __name__ == "__main__":
    main()
