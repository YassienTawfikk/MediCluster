import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from src.__00__paths import (
    processed_data_dir, figures_dir, docs_dir, model_dir
)

# =========================
# Config
# =========================
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


# =========================
# Transformers (top-level → picklable)
# =========================
class ColumnWeighter(BaseEstimator, TransformerMixin):
    """Multiply columns by provided weights (expects DataFrame or ndarray)."""

    def __init__(self, columns, weights: dict):
        self.columns = list(columns)
        self.weights = np.array([weights.get(c, 1.0) for c in self.columns], dtype=float)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_nd = X.values if isinstance(X, pd.DataFrame) else X
        return X_nd * self.weights


class MapCategories(BaseEstimator, TransformerMixin):
    """
    Map human-readable categories to numeric encodings:
      - smoking_status -> smoker {-1, 0, 1}
      - gender {'Female':0, 'Male':1}
      - residence_type {'Urban':0, 'Rural':1}
    Then reindex to training feature columns.
    """

    def __init__(self, feature_cols):
        self.feature_cols = list(feature_cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X_df):
        X = X_df.copy()

        if "smoking_status" in X.columns:
            X["smoker"] = X["smoking_status"].map({"Smoker": 1, "Non-Smoker": -1, "Unknown": 0})
            X.drop(columns=["smoking_status"], inplace=True)

        if "gender" in X.columns and X["gender"].dtype == object:
            X["gender"] = X["gender"].map({"Female": 0, "Male": 1}).astype(int)

        if "residence_type" in X.columns and X["residence_type"].dtype == object:
            X["residence_type"] = X["residence_type"].map({"Urban": 0, "Rural": 1}).astype(int)

        # Ensure exact training order (missing columns would become NaN → caller must provide all required fields)
        return X.reindex(columns=self.feature_cols)


# =========================
# Training entrypoint
# =========================
def train():
    """
    Runs k selection, trains final KMeans, writes artifacts, and saves a picklable inference pipeline.

    Returns:
        dict: paths + best_k
    """
    # --- Load datasets ---
    # df_proc: scaled + weighted features (no 'cluster' column)
    df_proc = pd.read_csv(processed_data_dir / "processed_patients_dataset.csv")
    # df_original: original units (human-readable strings allowed)
    df_original = pd.read_csv(processed_data_dir / "clean_preScaled_data.csv")

    # --- Choose K via elbow + silhouette on the modeling space ---
    inertia, sil = [], []
    K_range = list(range(2, 11))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_proc)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(df_proc, km.labels_))

    # Save elbow/silhouette plot
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertia, marker="o")
    plt.xlabel("k");
    plt.ylabel("Inertia");
    plt.title("K-Means Elbow");
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(K_range, sil, marker="o")
    plt.xlabel("k");
    plt.ylabel("Silhouette");
    plt.title("Silhouette Scores");
    plt.grid(True)
    plt.tight_layout()
    elbow_path = figures_dir / "kmeans_elbow.png"
    plt.savefig(elbow_path, dpi=160)
    plt.close()
    print("✔️ Plot saved →", "/".join(elbow_path.parts[-3:]))

    # --- Final fit with best_k (maximize silhouette) ---
    best_k = K_range[sil.index(max(sil))]
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(df_proc)

    # --- Metrics (on features only) ---
    ch = calinski_harabasz_score(df_proc, labels)
    db = davies_bouldin_score(df_proc, labels)
    seeds = range(5)
    lbls = [KMeans(n_clusters=best_k, random_state=rs, n_init=10).fit(df_proc).labels_ for rs in seeds]
    ari = np.mean([
        adjusted_rand_score(lbls[i], lbls[j])
        for i in range(len(seeds)) for j in range(i + 1, len(seeds))
    ])

    # --- Save clustered original-units CSV ---
    docs_dir.mkdir(parents=True, exist_ok=True)
    df_out = df_original.copy()
    df_out["cluster"] = labels
    clustered_path = docs_dir / "clustered_patients_dataset_original_units.csv"
    df_out.to_csv(clustered_path, index=False)
    print("✔️ Saved →", "/".join(clustered_path.parts[-3:]))

    # --- PCA 2D visualization (on modeling space) ---
    pca_vis = PCA(n_components=2).fit(df_proc)  # fit with DataFrame
    X_2d = pca_vis.transform(df_proc)
    # Avoid sklearn "feature names" warning by converting centers to DataFrame with same columns
    centers_df = pd.DataFrame(kmeans_final.cluster_centers_, columns=df_proc.columns)
    centers_2d = pca_vis.transform(centers_df)

    plt.figure(figsize=(8, 6))
    for c in np.unique(labels):
        idx = (labels == c)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=12, alpha=0.6, label=f"Cluster {c}")
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], s=200, marker="X", edgecolor="k", linewidth=1.5, label="Centers")
    var = pca_vis.explained_variance_ratio_ * 100
    plt.xlabel(f"PC1 ({var[0]:.1f}% var)")
    plt.ylabel(f"PC2 ({var[1]:.1f}% var)")
    plt.title("K-Means Clusters (PCA 2D projection)")
    plt.legend()
    plt.tight_layout()
    pca_fig_path = figures_dir / "kmeans_pca2d.png"
    plt.savefig(pca_fig_path, dpi=160)
    plt.close()
    print("✔️ Figure saved →", "/".join(pca_fig_path.parts[-3:]))

    # --- Metrics file (clean format) ---
    metrics_lines = [f"Best k: {best_k}", "Cluster counts:"]
    for cid, cnt in pd.Series(labels).value_counts().sort_index().items():
        metrics_lines.append(f"{cid}    {cnt}")
    metrics_lines.append(f"Calinski–Harabasz: {ch:.2f}, Davies–Bouldin: {db:.2f}")
    metrics_lines.append(f"KMeans stability (ARI): {ari:.3f}")
    metrics_path = docs_dir / "kmeans_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("\n".join(metrics_lines))
    print("✔️ Metrics saved →", "/".join(metrics_path.parts[-3:]))

    # --- Build & save inference pipeline (raw strings → map → scale → weight → KMeans) ---
    feature_cols = list(df_proc.columns)  # training feature order (no 'cluster')
    pipe = Pipeline(steps=[
        ("map_categories", MapCategories(feature_cols)),
        ("scale", StandardScaler()),
        ("weight", ColumnWeighter(feature_cols, WEIGHTS)),
        ("kmeans", KMeans(
            n_clusters=int(best_k),
            random_state=42,
            n_init=1,  # reuse learned centers
            init=kmeans_final.cluster_centers_
        )),
    ])

    # Fit on original-units data (strings allowed for gender/residence/smoking)
    pipe.fit(df_original)

    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = model_dir / "medicluster_kmeans_pipeline.joblib"
    joblib.dump({"pipeline": pipe, "feature_cols": feature_cols}, bundle)
    print("✔️ Inference pipeline saved →", "/".join(bundle.parts[-3:]))

    return {
        "best_k": best_k,
        "metrics_path": metrics_path,
        "pipeline_path": bundle,
        "elbow_plot": elbow_path,
        "pca_plot": pca_fig_path,
        "clustered_csv": clustered_path,
    }
