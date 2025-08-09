# main.py
from src.__00__paths import (
    raw_data_dir, processed_data_dir, model_dir, figures_dir, docs_dir,
    data_dir_list, output_dir_list
)
from src.__01__data_setup import download_data, preprocess
from src.__02__MediCluster_modeling import train


def main():
    # Create needed directories
    print("Creating needed directories...")
    for p in data_dir_list + output_dir_list:
        p.mkdir(parents=True, exist_ok=True)
    print("✔️ Directories ready.")

    # Download
    print("Downloading dataset...")
    raw_csv = download_data()
    print(f"✔️ Dataset available at {'/'.join(raw_csv.parts[-3:])}")

    # Preprocess
    print("Preprocessing dataset...")
    clean_path, processed_path = preprocess(raw_csv)
    print(f"✔️ Clean (pre-scaled) saved → {'/'.join(clean_path.parts[-3:])}")
    print(f"✔️ Processed (scaled+weighted) saved → {'/'.join(processed_path.parts[-3:])}")

    # Train + artifacts
    print("Training K-Means model & saving artifacts...")
    res = train()
    print(f"✔️ Best k: {res['best_k']}")
    print(f"✔️ Metrics saved → {'/'.join(res['metrics_path'].parts[-3:])}")
    print(f"✔️ Elbow plot → {'/'.join(res['elbow_plot'].parts[-3:])}")
    print(f"✔️ PCA plot → {'/'.join(res['pca_plot'].parts[-3:])}")
    print(f"✔️ Clustered CSV → {'/'.join(res['clustered_csv'].parts[-3:])}")
    print(f"✔️ Inference pipeline → {'/'.join(res['pipeline_path'].parts[-3:])}")

    print("All done!")


if __name__ == "__main__":
    main()