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
