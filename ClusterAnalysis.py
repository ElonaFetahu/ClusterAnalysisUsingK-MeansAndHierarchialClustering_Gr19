import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
from tkinter import Tk, simpledialog, messagebox

# K-Means Clustering
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return labels, centers

# Hierarchical Clustering
def hierarchical_clustering(data, n_clusters, method="ward", metric="euclidean"):
    if metric != "euclidean":
        distance_matrix = pairwise_distances(data, metric=metric)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method, affinity="precomputed")
        labels = clustering.fit_predict(distance_matrix)
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = clustering.fit_predict(data)
    return labels
# Plot K-Means Clusters
def plot_kmeans_clusters(data, labels, centers):
    plt.figure(figsize=(8, 6))

    # Define symbols and colors for each cluster
    markers = ['+', '*', 'o']
    colors = ['red', 'blue', 'green']

    for cluster in range(len(np.unique(labels))):
        cluster_points = data[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[cluster % len(colors)], 
                    marker=markers[cluster % len(markers)], 
                    label=f'Cluster {cluster + 1}', edgecolor='k')

    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, marker="X", label="Centroids")
    plt.title("K-Means Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="upper right")
    plt.show()

# Plot Hierarchical Clusters and Dendrogram
def plot_hierarchical_clusters(data, labels, method="ward", metric="euclidean"):
    plt.figure(figsize=(12, 6))

    # Plot the dendrogram
    if metric != "euclidean":
        distance_matrix = pairwise_distances(data, metric=metric)
        Z = linkage(distance_matrix, method=method)
    else:
        Z = linkage(data, method=method)

    plt.subplot(1, 2, 1)
    dendrogram(Z, truncate_mode="level", p=4)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")

    # Plot the clusters
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", marker="o", edgecolor="k")
    plt.title("Hierarchical Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.show()