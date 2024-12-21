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
def hierarchical_clustering(data, n_clusters=None, method="ward", metric="euclidean"):
    if metric != "euclidean":
        distance_matrix = pairwise_distances(data, metric=metric)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method, affinity="precomputed")
        labels = clustering.fit_predict(distance_matrix) if n_clusters else None
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = clustering.fit_predict(data) if n_clusters else None
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
def plot_hierarchical_clusters(data, labels=None, method="ward", metric="euclidean"):
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

    # Plot the clusters if labels are provided
    if labels is not None:
        plt.subplot(1, 2, 2)
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", marker="o", edgecolor="k")
        plt.title("Hierarchical Clusters")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.show()

# Main Application
def main():
    Tk().withdraw()  # Hide root tkinter window

    # Input data points
    n_points = simpledialog.askinteger("Input", "Enter the number of data points:", minvalue=3, maxvalue=300)
    data = []
    for i in range(n_points):
        point = simpledialog.askstring("Input", f"Enter coordinates for point {i+1} (e.g., x,y):")
        try:
            x, y = map(float, point.split(","))
            data.append([x, y])
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter coordinates as x,y.")
            return

    data = np.array(data)

    # Select clustering method
    method = simpledialog.askstring("Clustering Method", "Choose method: kmeans or hierarchical:")
    if method not in ["kmeans", "hierarchical"]:
        messagebox.showerror("Error", "Invalid method. Please choose 'kmeans' or 'hierarchical'.")
        return

    if method == "kmeans":
        # Number of clusters
        n_clusters = simpledialog.askinteger("Clusters", "Enter the number of clusters:", minvalue=2, maxvalue=n_points)
        labels, centers = kmeans_clustering(data, n_clusters)
        plot_kmeans_clusters(data, labels, centers)

    elif method == "hierarchical":
        # Select distance metric
        metric = simpledialog.askstring("Metric", "Enter distance metric (euclidean/manhattan):")
        if metric not in ["euclidean", "manhattan"]:
            messagebox.showerror("Error", "Invalid metric. Please choose 'euclidean' or 'manhattan'.")
            return

        # Select linkage method
        linkage_method = simpledialog.askstring("Linkage", "Enter linkage method (ward/single/complete/average):")
        if linkage_method not in ["ward", "single", "complete", "average"]:
            messagebox.showerror("Error", "Invalid linkage method.")
            return

        labels = hierarchical_clustering(data, method=linkage_method, metric=metric)
        plot_hierarchical_clusters(data, labels, linkage_method, metric)

if __name__ == "__main__":
    main() 
