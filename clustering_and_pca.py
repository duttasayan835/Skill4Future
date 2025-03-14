import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import os

def create_clustering_visualizations(X, output_dir='visualizations'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. K-Means Clustering
    # Elbow method to find optimal K
    inertias = []
    silhouette_scores = []
    K = range(2, 11)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kmeans_elbow_method.png')
    plt.close()
    
    # Perform K-means with optimal K (let's use 4 for visualization)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 2. PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot K-means clusters in PCA space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('K-means Clusters (PCA visualization)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.savefig(f'{output_dir}/kmeans_clusters_pca.png')
    plt.close()
    
    # 3. Hierarchical Clustering
    # Generate linkage matrix
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig(f'{output_dir}/hierarchical_clustering.png')
    plt.close()
    
    # 4. DBSCAN
    # Find optimal eps using k-distance graph
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances)
    plt.title('K-distance Graph')
    plt.xlabel('Points')
    plt.ylabel('Distance')
    plt.savefig(f'{output_dir}/dbscan_kdistance.png')
    plt.close()
    
    # Perform DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Plot DBSCAN clusters in PCA space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis')
    plt.title('DBSCAN Clusters (PCA visualization)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.savefig(f'{output_dir}/dbscan_clusters_pca.png')
    plt.close()
    
    # 5. PCA Analysis
    # Scree plot
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Scree Plot')
    plt.savefig(f'{output_dir}/pca_scree_plot.png')
    plt.close()
    
    # Feature importance in first two PCs
    pc_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=X.columns
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pc_df, annot=True, cmap='coolwarm', center=0)
    plt.title('PCA Components Feature Importance')
    plt.savefig(f'{output_dir}/pca_feature_importance.png')
    plt.close()
    
    return {
        'kmeans_labels': cluster_labels,
        'dbscan_labels': dbscan_labels,
        'pca_components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }
