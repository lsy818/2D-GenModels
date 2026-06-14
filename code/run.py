"""Entry point — train models, run experiments, generate all figures.

Usage:
    python -m code.run              # train + generate all figures
    python -m code.run --figs-only  # only generate figures (models must exist)
    python -m code.run --ext-only   # only run extension experiments
"""

from __future__ import annotations

import json
import sys

from code.config import CLASS_NAMES, MODEL_DIR, OUTPUT_DIR, get_device, set_seed
from code.data import get_class, load_data
from code.metrics import evaluate_generation_metrics

set_seed(42)


def main():
    args = set(sys.argv[1:])
    figs_only = "--figs-only" in args
    ext_only = "--ext-only" in args

    print(f"Device: {get_device()}")
    print("=" * 60)

    if not figs_only and not ext_only:
        print("STEP 1/3: Training main models ...")
        from code.experiments import train_main
        train_main()
        write_main_results()

        print("\nSTEP 2/3: Running extension experiments ...")
        run_extensions()

    elif ext_only:
        print("Running extension experiments only ...")
        run_extensions()

    if not ext_only:
        print("\nSTEP 3/3: Generating all figures ...")
        from code.visualization import make_all
        make_all()

    print("\nDone.")


def run_extensions():
    from code.experiments import ablation, conditional, robustness
    results = {
        "ablation": ablation(),
        "conditional": conditional(),
        "robustness": robustness(),
    }
    _write_json(OUTPUT_DIR / "conditional_results_with_wasserstein.json", results["conditional"])
    _write_json(OUTPUT_DIR / "robustness_extended_results.json", results["robustness"])

    out = OUTPUT_DIR / "extension_results.json"
    _write_json(out, results)
    print(f"\nExtension results saved to {out}")
    return results


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_main_results() -> dict:
    """Evaluate saved main VAE/DDPM models and write the result table."""
    from code.models.diffusion_model import DiffusionModel
    from code.models.vae_model import VAEModel

    test, test_label = load_data("test")
    results = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        real = get_class(test, test_label, class_id)

        vae = VAEModel.load(MODEL_DIR / f"vae_class{class_id}.pt", map_location="cpu")
        ddpm = DiffusionModel.load(MODEL_DIR / f"diffusion_class{class_id}.pt", map_location="cpu")

        set_seed(5100 + class_id)
        vae_metrics = evaluate_generation_metrics(real, vae.sample(2000))
        set_seed(5200 + class_id)
        ddpm_metrics = evaluate_generation_metrics(real, ddpm.sample(2000))

        results[class_name] = {"VAE": vae_metrics, "DDPM": ddpm_metrics}
        print(
            f"  {class_name}: "
            f"VAE Cov={vae_metrics['Coverage']:.3f}, "
            f"DDPM Cov={ddpm_metrics['Coverage']:.3f}"
        )

    out = OUTPUT_DIR / "main_results.json"
    _write_json(out, results)
    print(f"\nMain results saved to {out}")
    return results


if __name__ == "__main__":
    main()
