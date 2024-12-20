# ClusterAnalysisUsingK-MeansAndHierarchialClustering
- Cluster Analysis Using K-Means and Hierarchical Clustering
This project was developed as part of a semester course at the University "Hasan Prishtina" - Faculty of Electrical and Computer Engineering, in the course "Algorithm Design and Analysis". It implements two popular clustering algorithms, K-Means and Hierarchical Clustering, for analyzing and grouping data points into clusters based on similarity.

# Introduction
Clustering is a fundamental technique in unsupervised machine learning, used to group similar data points into clusters. This project provides an intuitive tool for performing clustering using two methods:
K-Means Clustering: A partition-based algorithm that iteratively assigns data points to clusters based on their proximity to cluster centroids.
Hierarchical Clustering: A tree-based approach that creates a hierarchy of clusters, visualized using a dendrogram.
This Python application includes both algorithms and offers visualizations to help understand the clustering results.

# How It Works
The application provides the following features:
Algorithm Selection: Users can choose between K-Means and Hierarchical Clustering.
Input Data:
Users can upload a dataset containing data points.
Alternatively, users can generate synthetic data for testing.
Cluster Configuration:
Specify the number of clusters for K-Means.
Choose the linkage method (e.g., single, complete, average) for Hierarchical Clustering.
Distance Metrics: Select distance metrics such as Euclidean or Manhattan.
Visualization: The application visualizes the resulting clusters and, for hierarchical clustering, generates a dendrogram.
Save Results: Users can save the cluster assignments and visualizations.

# Graphical User Interface (GUI)
The application uses tkinter for a user-friendly interface, allowing users to:
Select clustering options via dropdown menus.
Browse and upload data files or generate synthetic data.
Specify clustering parameters through dialog boxes.
View interactive visualizations of clusters and dendrograms.

# Functions
- K-Means Clustering
kmeans_clustering: Implements the K-Means algorithm to assign data points to clusters based on their proximity to centroids.
- Hierarchical Clustering
hierarchical_clustering: Performs hierarchical clustering and generates a dendrogram for visualizing the cluster hierarchy.
Visualization
plot_clusters: Displays the clustered data points with different colors for each cluster.
plot_dendrogram: Visualizes the hierarchical clustering structure.

# How to Use
Clone this repository to your local machine:
git clone https://github.com/username/ClusterAnalysisTool.git
Install Python and the required libraries:
pip install -r requirements.txt
Run the application:
python main.py
Follow the GUI prompts to select your clustering algorithm, upload data, and visualize the results.

# Contributors
Agnesa Mani
Elona Fetahu
Venesa Fejza
Rona Tasholli

# Technical Documentation
K-Means Algorithm
Initialization: Choose initial centroids randomly.
Assignment Step: Assign each data point to the nearest centroid.
Update Step: Recalculate centroids as the mean of assigned points.
Repeat until centroids stabilize or maximum iterations are reached.
Hierarchical Clustering Algorithm
Step 1: Compute the distance matrix for all data points.
Step 2: Merge the closest points or clusters iteratively.
Step 3: Generate a dendrogram to visualize the hierarchy.

# Security Considerations
While the clustering algorithms implemented are robust for educational and exploratory purposes, it is essential to validate the results when applied to real-world datasets. Ensure the integrity and privacy of sensitive data during processing.
