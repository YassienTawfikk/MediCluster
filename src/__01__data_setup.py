from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.__00__paths import (
    data_dir_list, output_dir_list,
    raw_data_dir, processed_data_dir,
    figures_dir, docs_dir,
)

# ---- Config ----
DATASET_SLUG = "arjunnsharma/patient-dataset-for-clustering-raw-data"
RAW_FILENAME = "patient_dataset.csv"

WEIGHTS = {
    # High importance
    "plasma_glucose": 1.8,
    "blood_pressure": 1.8,
    "cholesterol": 1.8,
    "bmi": 1.8,
    "hypertension": 1.8,
    "heart_disease": 1.8,

    # Medium importance
    "chest_pain_type": 1.3,
    "exercise_angina": 1.3,
    "max_heart_rate": 1.3,
    "insulin": 1.3,
}


def ensure_dirs():
    for p in data_dir_list + output_dir_list:
        p.mkdir(parents=True, exist_ok=True)


def download_data():
    """Download the raw CSV into data/raw/ if missing. Returns the file path."""
    ensure_dirs()
    raw_path = raw_data_dir / RAW_FILENAME
    if raw_path.exists():
        print("✔️ Dataset is already downloaded.")
        return raw_path

    try:
        import kagglehub
    except Exception as e:
        raise RuntimeError(
            "kagglehub is required to download the dataset. "
            "Install it or place the CSV manually in data/raw/."
        ) from e

    ds_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    if not ds_dir.exists():
        raise FileNotFoundError("⚠ Dataset download failed.")

    data_root = ds_dir / "Data" if (ds_dir / "Data").exists() else ds_dir
    for item in data_root.iterdir():
        if item.is_file():
            shutil.copy2(item, raw_data_dir / item.name)

    if not raw_path.exists():
        # Fall back: pick first CSV if naming differs
        csvs = sorted(raw_data_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"⚠ Expected {RAW_FILENAME} not found after copy.")
        raw_path = csvs[0]

    print("✔️ Dataset successfully downloaded →", "/".join(raw_path.parts[-3:]))
    return raw_path


def preprocess(raw_csv):
    """
    Preprocess and save:
    - clean_preScaled_data.csv (original units, readable)
    - processed_patients_dataset.csv (scaled + weighted)
    Returns (clean_path, processed_path).
    """
    ensure_dirs()
    # ---- Load ----
    raw_df = pd.read_csv(raw_csv)
    print("Raw shape:", raw_df.shape)

    # ---- Gap summary -> docs ----
    nan_cnt = raw_df.isna().sum()
    unknown_cnt = pd.Series(0, index=raw_df.columns)
    if "smoking_status" in raw_df.columns:
        unknown_cnt["smoking_status"] = (
            raw_df["smoking_status"].astype(str).str.strip().str.lower().eq("unknown").sum()
        )
    gaps = pd.DataFrame({"NaN": nan_cnt, "'Unknown' (smoking_status)": unknown_cnt})
    gaps["Total gaps"] = gaps["NaN"] + gaps["'Unknown' (smoking_status)"]
    gaps["% of rows (total)"] = (gaps["Total gaps"] / len(raw_df) * 100).round(2)
    gaps = gaps.sort_values("Total gaps", ascending=False).drop(columns=["NaN", "'Unknown' (smoking_status)"])
    (docs_dir / "gap_summary.csv").parent.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(docs_dir / "gap_summary.csv")

    # ---- Copy for processing ----
    df = raw_df.copy()

    # Smoking -> numeric 'smoker'
    if "smoking_status" in df.columns:
        df["smoker"] = df["smoking_status"].map({"Smoker": 1, "Non-Smoker": -1, "Unknown": 0})
        df.drop(columns=["smoking_status"], inplace=True)

    # Gender/residence: mode fill + map to ints
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna(df["gender"].mode(dropna=True)[0]).astype(int)
    if "residence_type" in df.columns:
        df["residence_type"] = df["residence_type"].fillna(df["residence_type"].mode(dropna=True)[0])
        df["residence_type"] = df["residence_type"].map({"Urban": 0, "Rural": 1})

    # Median impute for numeric trio
    for col in ["skin_thickness", "plasma_glucose", "insulin"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ---- Save clean (original units, readable labels) ----
    clean_prescaled = df.copy()
    if "residence_type" in clean_prescaled.columns:
        clean_prescaled["residence_type"] = clean_prescaled["residence_type"].map({0: "Urban", 1: "Rural"})
    if "gender" in clean_prescaled.columns:
        clean_prescaled["gender"] = clean_prescaled["gender"].map({0: "Female", 1: "Male"})
    if "smoker" in clean_prescaled.columns:
        clean_prescaled["smoking_status"] = clean_prescaled["smoker"].map(
            {1: "Smoker", -1: "Non-Smoker", 0: "Unknown"}
        )
        clean_prescaled.drop(columns=["smoker"], inplace=True)
    if "bmi" in clean_prescaled.columns:
        clean_prescaled["bmi"] = clean_prescaled["bmi"].round(2)
    if "diabetes_pedigree" in clean_prescaled.columns:
        clean_prescaled["diabetes_pedigree"] = clean_prescaled["diabetes_pedigree"].round(3)

    clean_path = processed_data_dir / "clean_preScaled_data.csv"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_prescaled.to_csv(clean_path, index=False)
    print("✔️ Clean (pre-scaled) saved →", "/".join(clean_path.parts[-3:]))

    # ---- Scale then weight ----
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    for col, w in WEIGHTS.items():
        if col in df_scaled.columns:
            df_scaled[col] = df_scaled[col] * w

    # Optional PCA variance plot
    pca = PCA().fit(df_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained) + 1), explained, marker="o")
    plt.axhline(y=0.95, linestyle="--")
    plt.xlabel("Number of Components");
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA - Cumulative Variance Explained");
    plt.grid(True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pca_plot = figures_dir / "pca_cumulative_variance.png"
    plt.tight_layout();
    plt.savefig(pca_plot, dpi=160);
    plt.close()

    processed_path = processed_data_dir / "processed_patients_dataset.csv"
    df_scaled.to_csv(processed_path, index=False)
    print("✔️ Processed data saved →", "/".join(processed_path.parts[-3:]))

    return clean_path, processed_path
