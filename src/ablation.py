"""
Sweeps qubit count, circuit depth, and data re-uploading (single-injection vs. re-injecting the
input before every entangling layer, see quantum_hybrid.py's build_quantum_layer docstring) for
the quantum hybrid head, retraining the head each time, across BOTH feature encodings:
  - "pca":    generic PCA-of-frozen-CNN-features, swept across every
              (qubit_count, depth, data_reuploading) triple.
  - "domain": the six clinically-motivated wound features (src/domain_features.py), which are
              fixed-dimensional (see N_DOMAIN_FEATURES) so only depth and reuploading are swept,
              at num_qubits fixed to that dimensionality.
The reuploading axis roughly doubles this sweep's runtime -- pass --no-reuploading-axis to run
only the original single-injection sweep (e.g. to reproduce a prior run's grid exactly).
Saves results/ablation_results.csv (tagged by feature_encoding and data_reuploading columns) and
figures/ablation_qubits_vs_accuracy.png (pca sweep, single-injection only, for readability) and
figures/ablation_reuploading_vs_accuracy.png (reuploading vs. not, at each depth, qubits fixed to
the config's default num_qubits -- the comparison this axis exists to make).

Run: python src/ablation.py --config configs/config.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from classical_baseline import load_config, set_seed
from domain_features import N_DOMAIN_FEATURES
from quantum_hybrid import run_hybrid_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-reuploading-axis", action="store_true",
                         help="Sweep only data_reuploading=False (the original grid), skipping "
                              "the added reuploading=True runs.")
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
    reuploading_options = (False,) if args.no_reuploading_axis else (False, True)
    if args.smoke_test:
        qubit_counts = qubit_counts[:1]
        depths = depths[:1]

    rows = []
    for encoding in ("pca", "domain"):
        for n_qubits in qubit_counts:
            if encoding == "domain" and n_qubits != N_DOMAIN_FEATURES:
                continue  # domain features are fixed-dimensional; only pca varies qubit count
            for depth in depths:
                for reuploading in reuploading_options:
                    print(f"\n########## Ablation: encoding={encoding}, qubits={n_qubits}, "
                          f"depth={depth}, data_reuploading={reuploading} ##########")
                    result = run_hybrid_pipeline(
                        cfg, research_root, device,
                        num_qubits=n_qubits, circuit_depth=depth, feature_encoding=encoding,
                        smoke_test=args.smoke_test, data_reuploading=reuploading,
                    )
                    rows.append({
                        "feature_encoding": encoding,
                        "num_qubits": n_qubits,
                        "circuit_depth": depth,
                        "data_reuploading": reuploading,
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

        # Main plot: qubit count / depth / encoding, single-injection only (data_reuploading has
        # its own comparison plot below) so this one stays directly comparable to a prior run's.
        pca_df = df[(df["feature_encoding"] == "pca") & (~df["data_reuploading"])]
        fig, ax = plt.subplots(figsize=(7, 5))
        for depth in sorted(pca_df["circuit_depth"].unique()):
            sub = pca_df[pca_df["circuit_depth"] == depth].sort_values("num_qubits")
            ax.plot(sub["num_qubits"], sub["test_accuracy"], marker="o", label=f"pca, depth={depth}")
        domain_df = df[(df["feature_encoding"] == "domain") & (~df["data_reuploading"])]
        if not domain_df.empty:
            ax.axhline(domain_df["test_accuracy"].max(), color="grey", linestyle="--",
                       label="domain encoding (best depth)")
        ax.set_xlabel("Number of qubits (= PCA dimensions)")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Ablation: qubit count / circuit depth / feature encoding vs. test accuracy\n"
                      "(single-injection; see ablation_reuploading_vs_accuracy.png for re-uploading)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "ablation_qubits_vs_accuracy.png", dpi=150)
        plt.close(fig)

        if not args.no_reuploading_axis:
            # The comparison this axis exists to make: at the config's own default qubit count,
            # does re-uploading beat single-injection at each depth?
            default_qubits = cfg["quantum"]["num_qubits"]
            reup_df = df[(df["feature_encoding"] == "pca") & (df["num_qubits"] == default_qubits)]
            fig, ax = plt.subplots(figsize=(7, 5))
            width = 0.35
            depths_sorted = sorted(reup_df["circuit_depth"].unique())
            x = range(len(depths_sorted))
            for offset, reuploading, label in ((-width / 2, False, "single-injection"),
                                                 (width / 2, True, "data re-uploading")):
                accs = [
                    reup_df[(reup_df["circuit_depth"] == d) & (reup_df["data_reuploading"] == reuploading)]
                    ["test_accuracy"].mean()
                    for d in depths_sorted
                ]
                ax.bar([xi + offset for xi in x], accs, width=width, label=label)
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(d) for d in depths_sorted])
            ax.set_xlabel("Circuit depth")
            ax.set_ylabel("Test accuracy")
            ax.set_title(f"Data re-uploading vs. single-injection (pca, {default_qubits} qubits)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / "ablation_reuploading_vs_accuracy.png", dpi=150)
            plt.close(fig)
            print(f"Saved: {figures_dir/'ablation_reuploading_vs_accuracy.png'}")
        print(f"Saved: {figures_dir/'ablation_qubits_vs_accuracy.png'}")

    print(f"Saved: {results_dir/'ablation_results.csv'}")


if __name__ == "__main__":
    main()
