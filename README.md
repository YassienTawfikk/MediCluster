# MediCluster

> Patient Dataset Clustering Using K-Means and PCA

<p align='center'>
   <img width="500" alt="medicluster_poster" src="https://github.com/user-attachments/assets/bd1b9599-6aa1-4bc4-8483-9645eb71598c" />
</p>

---

## Problem Statement

Medical data clustering enables healthcare providers to identify patient subgroups for better-targeted interventions and resource allocation. This project focuses on segmenting patients into meaningful groups using:

* **K-Means** with optimized cluster count.
* **PCA** for dimensionality reduction and visualization.

---

## Dataset Overview

We used the [Patient Dataset for Clustering](https://www.kaggle.com/datasets/arjunnsharma/patient-dataset-for-clustering-raw-data), containing:

* Demographics (age, gender, residence type, smoking status)
* Health indicators (blood pressure, cholesterol, BMI, glucose levels)
* Disease history (hypertension, heart disease)

---

## Data Preprocessing Steps

1. **Filling Gaps (Missing Values)**

   * Categorical (`gender`, `residence_type`) → filled with mode.
   * Continuous (`skin_thickness`, `plasma_glucose`, `insulin`) → filled with median.
   * `smoking_status` mapped to numeric: Smoker=1, Non-Smoker=-1, Unknown=0.

2. **Label Encoding**

   * `gender`: Female=0, Male=1
   * `residence_type`: Urban=0, Rural=1

3. **Scaling & Weighting**

   * Standardized features.
   * Weighted important medical indicators (high ×1.8, medium ×1.3).

4. **Dimensionality Reduction (PCA)**

   * Used for visualization; 95% variance at 15 components.

---

## Model Outputs

### Figure (saved under `outputs/figures/`)

`kmeans_pca2d.png` — 2D PCA projection showing 4 clusters with centroids.

<img width="800" height="600" alt="kmeans_pca2d" src="https://github.com/user-attachments/assets/6355d65e-602e-4ca4-9f11-cba5984e24bf" />


### Documents (saved under `outputs/docs/`)


| Metric                      | Value  |
| --------------------------- | ------ |
| **Best k**                  | 4      |
| **Cluster 0 size**          | 1,525  |
| **Cluster 1 size**          | 1,467  |
| **Cluster 2 size**          | 1,522  |
| **Cluster 3 size**          | 1,486  |
| **Calinski–Harabasz Score** | 602.07 |
| **Davies–Bouldin Score**    | 2.68   |
| **KMeans Stability (ARI)**  | 1.000  |

**`clustered_patients_dataset_original_units.csv`** — same as clean dataset plus `cluster` column (0–3).

---

## Inference Pipeline

**`medicluster_kmeans_pipeline.joblib`** — pipeline that:

1. Maps categories → numeric.
2. Scales & weights features.
3. Predicts cluster directly from raw patient input.

---

## Project Structure

```
MediCluster/
│
├── documents/
│   └── project_structure.txt
│
├── data/
│   ├── raw/
│   │   └── patient_dataset.csv
│   │
│   └── processed/
│       ├── clean_preScaled_data.csv
│       └── processed_patients_dataset.csv
│
├── notebooks/
│   ├── 00_init.ipynb
│   ├── 01_data_setup.ipynb
│   └── 02_MediCluster_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── __00__paths.py
│   ├── 01_data_setup.ipynb
│   └── 02_MediCluster_Modeling.ipynb
│
├── outputs/
│   ├── model/
│   │   └── medicluster_kmeans_pipeline.joblib
│   │
│   ├── figures/
│   │   └── kmeans_pca2d.png
│   │
│   └── docs/
│       ├── clustered_patients_dataset_original_units.csv
│       └── kmeans_metrics.txt
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Submission

Part of the **MediCluster Series**, applying unsupervised learning to patient segmentation.

---

## Author

<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br>
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>




