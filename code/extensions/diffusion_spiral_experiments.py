"""Targeted DDPM experiments on the Spiral class.

The goal is to test whether DDPM can recover the two-arm spiral better after
adjusting the diffusion process and adding a lightweight geometric inductive
bias.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.config import CLASS_COLORS, DATA_DIR, FIGURES_DIR, MODEL_DIR, get_device, set_seed
from code.metrics import coverage_precision, mmd_rbf
from code.models.diffusion_model import DiffusionModel


OUT_DIR = FIGURES_DIR / "extensions"
MODEL_OUT = MODEL_DIR / "extensions"
DEVICE = get_device()


@dataclass
class DDPMConfig:
    name: str
    n_steps: int = 400
    schedule: str = "linear"
    hidden_dim: int = 256
    num_layers: int = 4
    feature_mode: str = "xy"
    epochs: int = 1200
    lr: float = 2e-4
    batch_size: int = 256


@dataclass
class Result:
    name: str
    mmd: float
    coverage: float
    precision: float
    support_off: float
    curve_mean: float
    curve_off: float
    train_loss: float
    sample_time_ms: float


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        base = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-np.log(10000.0) / max(half - 1, 1))
        )
        emb = t[:, None].float() * base[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


def _augment(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "xy":
        return x
    if mode == "polar":
        r = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-8)
        theta = torch.atan2(x[:, 1:2], x[:, 0:1])
        return torch.cat([x, r / 3.0, torch.sin(theta), torch.cos(theta)], dim=1)
    raise ValueError(f"unknown feature mode: {mode}")


class Denoiser(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, feature_mode: str, time_dim: int = 64):
        super().__init__()
        self.feature_mode = feature_mode
        self.time_emb = TimeEmbedding(time_dim)
        input_dim = 2 if feature_mode == "xy" else 5
        layers: list[nn.Module] = []
        prev = input_dim + time_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(prev, hidden_dim), nn.SiLU()])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([_augment(x, self.feature_mode), self.time_emb(t)], dim=-1))


def make_schedule(n_steps: int, schedule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if schedule == "linear":
        betas = np.linspace(1e-4, 0.02, n_steps, dtype=np.float32)
    elif schedule == "cosine":
        s = 0.008
        steps = np.arange(n_steps + 1, dtype=np.float64)
        acp = np.cos(((steps / n_steps) + s) / (1 + s) * np.pi / 2) ** 2
        acp = acp / acp[0]
        betas = 1.0 - acp[1:] / acp[:-1]
        betas = np.clip(betas, 1e-5, 0.02).astype(np.float32)
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    alphas = 1.0 - betas
    acp = np.cumprod(alphas).astype(np.float32)
    return betas, alphas.astype(np.float32), acp


class SpiralDDPM:
    def __init__(self, cfg: DDPMConfig):
        self.cfg = cfg
        self.betas, self.alphas, self.acp = make_schedule(cfg.n_steps, cfg.schedule)
        self.sqrt_acp = np.sqrt(self.acp).astype(np.float32)
        self.sqrt_one_minus = np.sqrt(1.0 - self.acp).astype(np.float32)
        self.denoiser = Denoiser(cfg.hidden_dim, cfg.num_layers, cfg.feature_mode).to(DEVICE)
        self.train_losses: list[float] = []

    def fit(self, data: np.ndarray) -> "SpiralDDPM":
        loader = DataLoader(
            TensorDataset(torch.tensor(data, dtype=torch.float32)),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        opt = torch.optim.AdamW(self.denoiser.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        sacp = torch.tensor(self.sqrt_acp, device=DEVICE)
        som = torch.tensor(self.sqrt_one_minus, device=DEVICE)

        for epoch in range(1, self.cfg.epochs + 1):
            self.denoiser.train()
            running = 0.0
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                t = torch.randint(0, self.cfg.n_steps, (batch.size(0),), device=DEVICE)
                noise = torch.randn_like(batch)
                xt = sacp[t].view(-1, 1) * batch + som[t].view(-1, 1) * noise
                loss = F.mse_loss(self.denoiser(xt, t), noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += float(loss.item())
            self.train_losses.append(running / len(loader))
            if epoch % 400 == 0:
                print(f"    {self.cfg.name} epoch {epoch}/{self.cfg.epochs}: loss={self.train_losses[-1]:.5f}")
        return self

    @torch.no_grad()
    def sample(self, n: int, seed: int = 42) -> tuple[np.ndarray, float]:
        import time

        self.denoiser.eval()
        torch.manual_seed(seed)
        x = torch.randn(n, 2, device=DEVICE)
        start = time.perf_counter()

        for ti in range(self.cfg.n_steps - 1, -1, -1):
            t = torch.full((n,), ti, device=DEVICE, dtype=torch.long)
            eps = self.denoiser(x, t)
            alpha_t = self.alphas[ti]
            acp_t = self.acp[ti]
            acp_prev = self.acp[ti - 1] if ti > 0 else 1.0
            beta_t = self.betas[ti]

            x0_pred = (x - np.sqrt(1.0 - acp_t) * eps) / np.sqrt(acp_t)
            coef_x0 = np.sqrt(acp_prev) * beta_t / (1.0 - acp_t)
            coef_xt = np.sqrt(alpha_t) * (1.0 - acp_prev) / (1.0 - acp_t)
            mean = coef_x0 * x0_pred + coef_xt * x

            if ti > 0:
                posterior_var = beta_t * (1.0 - acp_prev) / (1.0 - acp_t)
                x = mean + np.sqrt(max(posterior_var, 1e-12)) * torch.randn_like(x)
            else:
                x = mean

        elapsed = (time.perf_counter() - start) * 1000.0
        return x.cpu().numpy().astype(np.float32), elapsed

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "state": self.denoiser.state_dict(),
                "losses": self.train_losses,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SpiralDDPM":
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = cls(DDPMConfig(**ckpt["cfg"]))
        model.denoiser.load_state_dict(ckpt["state"])
        model.denoiser.eval()
        model.train_losses = list(ckpt["losses"])
        return model


def _support_off(real: np.ndarray, gen: np.ndarray) -> float:
    tree = spatial.cKDTree(real)
    knn, _ = tree.query(real, k=6)
    threshold = float(np.percentile(knn[:, -1], 95))
    dist, _ = tree.query(gen, k=1)
    return float(np.mean(dist > threshold))


def _spiral_reference(n: int = 3000) -> np.ndarray:
    t = np.linspace(0.25, 4.0 * np.pi, n // 2)
    r = 0.22 * t
    arm0 = np.column_stack([r * np.cos(t), r * np.sin(t)])
    arm1 = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)])
    return np.concatenate([arm0, arm1], axis=0).astype(np.float32)


def _curve_metrics(real: np.ndarray, gen: np.ndarray) -> tuple[float, float]:
    ref = _spiral_reference()
    tree = spatial.cKDTree(ref)
    real_dist, _ = tree.query(real, k=1)
    threshold = float(np.percentile(real_dist, 95))
    gen_dist, _ = tree.query(gen, k=1)
    return float(np.mean(gen_dist)), float(np.mean(gen_dist > threshold))


def _evaluate(name: str, real: np.ndarray, gen: np.ndarray, train_loss: float, sample_time_ms: float) -> Result:
    cp = coverage_precision(real, gen, k=5, max_subsample=1000)
    curve_mean, curve_off = _curve_metrics(real, gen)
    return Result(
        name=name,
        mmd=mmd_rbf(real, gen, max_subsample=1000),
        coverage=cp["coverage"],
        precision=cp["precision"],
        support_off=_support_off(real, gen),
        curve_mean=curve_mean,
        curve_off=curve_off,
        train_loss=train_loss,
        sample_time_ms=sample_time_ms,
    )


def _load_or_train(cfg: DDPMConfig, train: np.ndarray) -> SpiralDDPM:
    path = MODEL_OUT / f"spiral_ddpm_{cfg.name}.pt"
    if path.exists():
        print(f"  loading {cfg.name}: {path.name}")
        return SpiralDDPM.load(path)
    print(f"  training {cfg.name}")
    model = SpiralDDPM(cfg).fit(train)
    model.save(path)
    return model


def _sample_existing_baseline(n: int) -> tuple[np.ndarray, float, float]:
    import time

    model = DiffusionModel.load(MODEL_DIR / "diffusion_class3.pt", map_location=DEVICE)
    start = time.perf_counter()
    gen = model.sample(n)
    elapsed = (time.perf_counter() - start) * 1000.0
    train_loss = float(model.train_losses[-1]) if model.train_losses else float("nan")
    return gen, train_loss, elapsed


def run_experiments() -> tuple[list[Result], dict[str, np.ndarray], np.ndarray]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    train = np.load(DATA_DIR / "train.npy").astype(np.float32)
    train_label = np.load(DATA_DIR / "train_label.npy").astype(np.int64)
    test = np.load(DATA_DIR / "test.npy").astype(np.float32)
    test_label = np.load(DATA_DIR / "test_label.npy").astype(np.int64)
    spiral_train = train[train_label == 3]
    spiral_test = test[test_label == 3]

    configs = [
        DDPMConfig(name="linear_T100", n_steps=100, schedule="linear", feature_mode="xy", epochs=1200),
        DDPMConfig(name="cosine_T400", n_steps=400, schedule="cosine", feature_mode="xy", epochs=1200),
        DDPMConfig(name="polar_T400", n_steps=400, schedule="cosine", feature_mode="polar", epochs=1200),
    ]

    samples: dict[str, np.ndarray] = {"Data": spiral_test}
    results: list[Result] = []

    print(f"Device: {DEVICE}")
    print("\nBaseline existing DDPM")
    gen, loss, sample_time = _sample_existing_baseline(len(spiral_test))
    samples["baseline"] = gen
    row = _evaluate("baseline", spiral_test, gen, loss, sample_time)
    results.append(row)
    print(
        f"  baseline MMD={row.mmd:.4f} Cov={row.coverage:.3f} Prec={row.precision:.3f} "
        f"CurveOff={row.curve_off:.3f}"
    )

    for cfg in configs:
        model = _load_or_train(cfg, spiral_train)
        gen, sample_time = model.sample(len(spiral_test), seed=200 + len(results))
        samples[cfg.name] = gen
        row = _evaluate(cfg.name, spiral_test, gen, model.train_losses[-1], sample_time)
        results.append(row)
        print(
            f"  {cfg.name} MMD={row.mmd:.4f} Cov={row.coverage:.3f} Prec={row.precision:.3f} "
            f"CurveOff={row.curve_off:.3f}"
        )

    return results, samples, spiral_test


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


def figure_samples(samples: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(42)
    keys = ["Data", "baseline", "linear_T100", "cosine_T400", "polar_T400"]
    titles = ["Data", "Baseline", "Linear T=100", "Cosine T=400", "Polar feat. T=400"]
    fig, axes = plt.subplots(1, len(keys), figsize=(11.4, 2.8), constrained_layout=True)
    for ax, key, title in zip(axes, keys, titles):
        pts = _subsample(samples[key], 1100, rng)
        color = "#30343b" if key == "Data" else CLASS_COLORS[3]
        ax.scatter(pts[:, 0], pts[:, 1], s=4.0, c=color, alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=10, weight="bold")
        _format_xy(ax)
    fig.suptitle("Spiral DDPM variants: prior samples", fontsize=12, weight="bold")
    fig.savefig(OUT_DIR / "diffusion_spiral_samples.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results, samples, _ = run_experiments()
    figure_samples(samples)
    print(f"\nSaved figures to {OUT_DIR}")
    print(f"Computed {len(results)} DDPM Spiral variant rows.")


if __name__ == "__main__":
    main()
