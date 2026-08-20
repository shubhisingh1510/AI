"""
Sweeps qubit count and circuit depth for the quantum hybrid head, retraining the head each
time (the CNN backbone stays frozen and reused across every run — only PCA dims and the
quantum circuit change). Saves results/ablation_results.csv and
figures/ablation_qubits_vs_accuracy.png.

Run: python src/ablation.py --config configs/config.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from classical_baseline import load_config, set_seed
from quantum_hybrid import run_hybrid_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qubit_counts = cfg["ablation"]["qubit_counts"]
    depths = cfg["ablation"]["circuit_depths"]
    if args.smoke_test:
        qubit_counts = qubit_counts[:1]
        depths = depths[:1]

    rows = []
    for n_qubits in qubit_counts:
        for depth in depths:
            print(f"\n########## Ablation: qubits={n_qubits}, depth={depth} ##########")
            result = run_hybrid_pipeline(
                cfg, research_root, device,
                num_qubits=n_qubits, circuit_depth=depth, smoke_test=args.smoke_test,
            )
            rows.append({
                "num_qubits": n_qubits,
                "circuit_depth": depth,
                "n_params": result["n_params"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "test_f1_macro": result["test_metrics"]["f1_macro"],
                "cv_mean_accuracy": result.get("cv_mean_accuracy"),
                "cv_std_accuracy": result.get("cv_std_accuracy"),
                "train_time_s": result["train_time_s"],
                "inference_time_ms_per_image": result["inference_time_ms_per_image"],
            })

    results_dir = research_root / cfg["paths"]["results_dir"]
    figures_dir = research_root / cfg["paths"]["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "ablation_results.csv", index=False)
    print("\n=== Ablation results ===")
    print(df.to_string(index=False))

    if not args.smoke_test:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for depth in sorted(df["circuit_depth"].unique()):
            sub = df[df["circuit_depth"] == depth].sort_values("num_qubits")
            ax.plot(sub["num_qubits"], sub["test_accuracy"], marker="o", label=f"depth={depth}")
        ax.set_xlabel("Number of qubits (= PCA dimensions)")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Ablation: qubit count / circuit depth vs. test accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "ablation_qubits_vs_accuracy.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {figures_dir/'ablation_qubits_vs_accuracy.png'}")

    print(f"Saved: {results_dir/'ablation_results.csv'}")


if __name__ == "__main__":
    main()
