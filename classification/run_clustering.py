import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = "."
TOP_N = 5

K_MIN = 2
K_MAX = 23

RANDOM_STATE = 42

# =========================================================
# LOAD SELECTED K CSV
# =========================================================

selected_k_path = "selected_k.csv"

if not os.path.exists(selected_k_path):

    raise FileNotFoundError(
        "selected_k.csv not found"
    )

selected_k_df = pd.read_csv(selected_k_path)

print("\nLoaded selected_k.csv")

# =========================================================
# GET ALL EXERCISES
# =========================================================

exercise_folders = [
    f for f in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, f))
]

print(f"\nFound {len(exercise_folders)} exercises\n")

# =========================================================
# MAIN LOOP
# =========================================================

for exercise_name in exercise_folders:

    print("\n" + "=" * 80)
    print(f"PROCESSING : {exercise_name}")
    print("=" * 80)

    # -----------------------------------------------------
    # PATH
    # -----------------------------------------------------

    run_path = os.path.join(
        BASE_DIR,
        exercise_name,
        "fold0",
        "run0"
    )

    if not os.path.exists(run_path):

        print(f"[WARNING] Missing : {run_path}")
        continue

    # -----------------------------------------------------
    # FILES
    # -----------------------------------------------------

    latent_train_path = os.path.join(
        run_path,
        "latent_space_train.npy"
    )

    latent_test_path = os.path.join(
        run_path,
        "latent_space_test.npy"
    )

    y_train_path = os.path.join(
        run_path,
        "y_train_fold0.npy"
    )

    y_test_path = os.path.join(
        run_path,
        "y_test_fold0.npy"
    )

    # -----------------------------------------------------
    # CHECK FILES
    # -----------------------------------------------------

    required_files = [
        latent_train_path,
        latent_test_path,
        y_train_path,
        y_test_path
    ]

    missing = [
        f for f in required_files
        if not os.path.exists(f)
    ]

    if len(missing) > 0:

        print("[WARNING] Missing files :")

        for m in missing:
            print(m)

        continue

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    latent_train = np.load(
        latent_train_path,
        allow_pickle=True
    )

    latent_test = np.load(
        latent_test_path,
        allow_pickle=True
    )

    y_train = np.load(
        y_train_path,
        allow_pickle=True
    )

    y_test = np.load(
        y_test_path,
        allow_pickle=True
    )

    # -----------------------------------------------------
    # CONCAT
    # -----------------------------------------------------

    latent_all = np.concatenate(
        [latent_train, latent_test],
        axis=0
    )

    y_all = np.concatenate(
        [y_train, y_test],
        axis=0
    )

    print(f"Latent shape : {latent_all.shape}")

    # =====================================================
    # KEEP ORIGINAL LABELS
    # =====================================================

    classes = np.unique(y_all)

    print(f"Classes : {classes}")

    # =====================================================
    # LOOP OVER CLASSES
    # =====================================================

    for cls in classes:

        print("\n" + "-" * 60)
        print(f"CLASS : {cls}")
        print("-" * 60)

        # -------------------------------------------------
        # FILTER CLASS
        # -------------------------------------------------

        class_indices = np.where(
            y_all == cls
        )[0]

        latent_cls = latent_all[
            class_indices
        ]

        print(f"Samples : {len(latent_cls)}")

        if len(latent_cls) < 3:

            print("[WARNING] Too few samples")
            continue

        # -------------------------------------------------
        # OUTPUT DIR
        # -------------------------------------------------

        class_output_dir = os.path.join(
            run_path,
            f"class_{str(cls)}"
        )

        os.makedirs(
            class_output_dir,
            exist_ok=True
        )

        # =================================================
        # METRICS FOR ALL K
        # =================================================

        metrics = []

        max_k = min(
            K_MAX,
            len(latent_cls) - 1
        )

        for k in range(K_MIN, max_k + 1):

            kmeans = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=10
            )

            labels = kmeans.fit_predict(
                latent_cls
            )

            inertia = kmeans.inertia_

            silhouette = silhouette_score(
                latent_cls,
                labels
            )

            metrics.append({
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette
            })

        # -------------------------------------------------
        # SAVE METRICS
        # -------------------------------------------------

        metrics_df = pd.DataFrame(metrics)

        metrics_csv_path = os.path.join(
            class_output_dir,
            "metrics.csv"
        )

        metrics_df.to_csv(
            metrics_csv_path,
            index=False
        )

        print(f"Saved : {metrics_csv_path}")

        # =================================================
        # ELBOW PLOT
        # =================================================

        plt.figure(figsize=(8, 5))

        plt.plot(
            metrics_df["k"],
            metrics_df["inertia"],
            marker='o'
        )

        plt.title(
            f"Elbow - {exercise_name} - Class {cls}"
        )

        plt.xlabel("K")
        plt.ylabel("Inertia")

        plt.grid(True)

        elbow_path = os.path.join(
            class_output_dir,
            "elbow.png"
        )

        plt.savefig(
            elbow_path,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved : {elbow_path}")

        # =================================================
        # GET SELECTED K
        # =================================================

        selected_row = selected_k_df[
            (selected_k_df["exercise"] == exercise_name)
            &
            (selected_k_df["class"].astype(str) == str(cls))
        ]

        if len(selected_row) == 0:

            print(
                f"[WARNING] No selected K for "
                f"{exercise_name} class {cls}"
            )

            continue

        best_k = int(
            selected_row.iloc[0]["selected_k"]
        )

        print(f"Selected K : {best_k}")

        # =================================================
        # FINAL KMEANS
        # =================================================

        final_kmeans = KMeans(
            n_clusters=best_k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        final_labels = final_kmeans.fit_predict(
            latent_cls
        )

        centroids = final_kmeans.cluster_centers_

        # =================================================
        # SAVE CLUSTERS
        # =================================================

        clusters_df = pd.DataFrame({
            "sequence_index": class_indices,
            "cluster": final_labels
        })

        clusters_csv_path = os.path.join(
            class_output_dir,
            "clusters.csv"
        )

        clusters_df.to_csv(
            clusters_csv_path,
            index=False
        )

        print(f"Saved : {clusters_csv_path}")

        # =================================================
        # REPRESENTATIVES
        # =================================================

        representatives = []

        for cluster_id in range(best_k):

            cluster_mask = (
                final_labels == cluster_id
            )

            cluster_local_indices = np.where(
                cluster_mask
            )[0]

            cluster_points = latent_cls[
                cluster_local_indices
            ]

            centroid = centroids[
                cluster_id
            ].reshape(1, -1)

            distances = cdist(
                cluster_points,
                centroid
            )

            sorted_idx = np.argsort(
            distances[:, 0]
            )

            n_representatives = min(
            TOP_N,
            len(cluster_points)
            )

            top_idx = sorted_idx[:n_representatives]

            top_local_indices = (
                cluster_local_indices[top_idx]
            )

            top_global_indices = class_indices[
                top_local_indices
            ]

            top_distances = distances[
                top_idx,
                0
            ]

            for seq_idx, dist in zip(
                top_global_indices,
                top_distances
            ):

                representatives.append({
                    "cluster": cluster_id,
                    "sequence_index": int(seq_idx),
                    "distance_to_centroid": float(dist)
                })

        # -------------------------------------------------
        # SAVE REPRESENTATIVES
        # -------------------------------------------------

        representatives_df = pd.DataFrame(
            representatives
        )

        representatives_csv_path = os.path.join(
            class_output_dir,
            "representatives.csv"
        )

        representatives_df.to_csv(
            representatives_csv_path,
            index=False
        )

        print(f"Saved : {representatives_csv_path}")

        # =================================================
        # PCA VISUALIZATION
        # =================================================

        pca = PCA(n_components=2)

        latent_2d = pca.fit_transform(
            latent_cls
        )

        centroids_2d = pca.transform(
            centroids
        )

        plt.figure(figsize=(10, 8))

        # -------------------------------------------------
        # CLUSTERS
        # -------------------------------------------------

        for cluster_id in range(best_k):

            cluster_points = latent_2d[
                final_labels == cluster_id
            ]

            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.7
            )

        # -------------------------------------------------
        # CENTROIDS
        # -------------------------------------------------

        plt.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c='black',
            marker='X',
            s=250,
            label='Centroids'
        )

        # -------------------------------------------------
        # REPRESENTATIVES
        # -------------------------------------------------

        for cluster_id in range(best_k):

            reps_cluster = representatives_df[
                representatives_df["cluster"]
                == cluster_id
            ]

            rep_indices = reps_cluster[
                "sequence_index"
            ].values

            local_mask = np.isin(
                class_indices,
                rep_indices
            )

            rep_points = latent_2d[
                local_mask
            ]

            plt.scatter(
                rep_points[:, 0],
                rep_points[:, 1],
                c='red',
                marker='*',
                s=300
            )

        # -------------------------------------------------
        # FINAL FIGURE
        # -------------------------------------------------

        plt.title(
            f"{exercise_name} "
            f"- Class {cls} "
            f"(K={best_k})"
        )

        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")

        plt.legend()
        plt.grid(True)

        pca_path = os.path.join(
            class_output_dir,
            "pca_clusters.png"
        )

        plt.savefig(
            pca_path,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved : {pca_path}")

print("\nDONE")