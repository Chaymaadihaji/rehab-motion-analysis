import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist



EXERCISE = "UCDHE_CLF_MC_Rowing"
TARGET_CLASS = 'rb'
K = 8
TOP_N = 5





run_path = os.path.join(
    EXERCISE,
    "fold0",
    "run0"
) 

latent_train = np.load(
    os.path.join(run_path, "latent_space_train.npy"),
    allow_pickle=True
)

latent_test = np.load(
    os.path.join(run_path, "latent_space_test.npy"),
    allow_pickle=True
)

y_train = np.load(
    os.path.join(run_path, "y_train_fold0.npy"),
    allow_pickle=True
)

y_test = np.load(
    os.path.join(run_path, "y_test_fold0.npy"),
    allow_pickle=True
)



latent_all = np.concatenate(
    [latent_train, latent_test],
    axis=0
)

y_all = np.concatenate(
    [y_train, y_test],
    axis=0
)



indices = np.where(
    y_all == TARGET_CLASS
)[0]

latent_cls = latent_all[indices]

print(f"Samples : {len(latent_cls)}")



kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(latent_cls)

centroids = kmeans.cluster_centers_



pca = PCA(n_components=2)

latent_2d = pca.fit_transform(latent_cls)

centroids_2d = pca.transform(centroids)



representatives = []

for cluster_id in range(K):

    cluster_mask = labels == cluster_id

    cluster_local_idx = np.where(
        cluster_mask
    )[0]

    cluster_points = latent_cls[
        cluster_local_idx
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

    top_idx = sorted_idx[:TOP_N]

    top_local_idx = cluster_local_idx[
        top_idx
    ]

    rep_points = latent_2d[
        top_local_idx
    ]

    representatives.append(rep_points)

plt.figure(figsize=(10, 8))

for cluster_id in range(K):

    points = latent_2d[
        labels == cluster_id
    ]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.7
    )

# centroids
plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    c='black',
    marker='X',
    s=250,
    label='Centroids'
)



# representatives
for rep in representatives:

    plt.scatter(
        rep[:, 0],
        rep[:, 1],
        c='red',
        marker='*',
        s=300
    )

plt.title(
    f"{EXERCISE} - class {TARGET_CLASS} - K={K}"
)

plt.grid(True)
plt.legend()

plt.show()