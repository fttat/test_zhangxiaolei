"""
Analysis Core Module for CCGL Analytics
Handles machine learning analysis, clustering, and pattern recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from ..utils.logger import get_logger


class ClusteringAnalysis:
    """Clustering analysis for data segmentation"""
    
    def __init__(self):
        self.logger = get_logger("clustering_analysis")
        self.models = {}
        self.results = {}
    
    def perform_kmeans_clustering(self, df: pd.DataFrame, n_clusters: int = None, max_clusters: int = 10) -> Dict[str, Any]:
        """
        Perform K-Means clustering analysis
        
        Args:
            df: DataFrame for clustering
            n_clusters: Number of clusters (if None, will find optimal)
            max_clusters: Maximum clusters to test for optimal selection
            
        Returns:
            Dictionary containing clustering results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for clustering analysis")
        
        # Prepare data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for clustering")
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(scaled_data, max_clusters)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        inertia = kmeans.inertia_
        
        # Store model and results
        self.models['kmeans'] = kmeans
        self.results['kmeans'] = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'inertia': inertia,
            'cluster_centers': kmeans.cluster_centers_,
            'scaler': scaler
        }
        
        # Add cluster labels to original DataFrame
        result_df = df.copy()
        result_df['cluster'] = cluster_labels
        
        # Generate cluster summary
        cluster_summary = self._generate_cluster_summary(result_df, numeric_df.columns)
        
        self.logger.info(f"K-Means clustering completed with {n_clusters} clusters. Silhouette score: {silhouette_avg:.3f}")
        
        return {
            'algorithm': 'kmeans',
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'inertia': inertia,
            'cluster_labels': cluster_labels,
            'clustered_data': result_df,
            'cluster_summary': cluster_summary,
            'feature_importance': self._calculate_feature_importance(scaled_data, cluster_labels, numeric_df.columns)
        }
    
    def perform_dbscan_clustering(self, df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """
        Perform DBSCAN clustering analysis
        
        Args:
            df: DataFrame for clustering
            eps: Maximum distance between samples in same neighborhood
            min_samples: Minimum samples in neighborhood for core point
            
        Returns:
            Dictionary containing clustering results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for clustering analysis")
        
        # Prepare data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for clustering")
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_data)
        
        # Calculate metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate silhouette score (only if we have more than 1 cluster)
        silhouette_avg = None
        if n_clusters > 1:
            # Remove noise points for silhouette calculation
            non_noise_mask = cluster_labels != -1
            if non_noise_mask.sum() > 0:
                silhouette_avg = silhouette_score(scaled_data[non_noise_mask], cluster_labels[non_noise_mask])
        
        # Store model and results
        self.models['dbscan'] = dbscan
        self.results['dbscan'] = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'eps': eps,
            'min_samples': min_samples,
            'scaler': scaler
        }
        
        # Add cluster labels to original DataFrame
        result_df = df.copy()
        result_df['cluster'] = cluster_labels
        
        # Generate cluster summary
        cluster_summary = self._generate_cluster_summary(result_df, numeric_df.columns)
        
        self.logger.info(f"DBSCAN clustering completed. Found {n_clusters} clusters and {n_noise} noise points.")
        
        return {
            'algorithm': 'dbscan',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette_avg,
            'cluster_labels': cluster_labels,
            'clustered_data': result_df,
            'cluster_summary': cluster_summary,
            'eps': eps,
            'min_samples': min_samples
        }
    
    def _find_optimal_clusters(self, data: np.ndarray, max_clusters: int) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        
        K_range = range(2, min(max_clusters + 1, len(data)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, cluster_labels))
        
        # Use silhouette score to determine optimal clusters
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        self.logger.info(f"Optimal number of clusters determined: {optimal_k}")
        return optimal_k
    
    def _generate_cluster_summary(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """Generate summary statistics for each cluster"""
        summary = {}
        
        for cluster_id in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster_id]
            
            summary[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'numeric_means': cluster_data[numeric_columns].mean().to_dict(),
                'numeric_stds': cluster_data[numeric_columns].std().to_dict()
            }
        
        return summary
    
    def _calculate_feature_importance(self, data: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for clustering"""
        # Calculate variance ratio for each feature across clusters
        importance = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # Calculate between-cluster variance vs within-cluster variance
            overall_mean = np.mean(feature_data)
            overall_var = np.var(feature_data)
            
            if overall_var == 0:
                importance[feature] = 0
                continue
            
            between_cluster_var = 0
            within_cluster_var = 0
            
            for cluster_id in np.unique(labels):
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    continue
                    
                cluster_mask = labels == cluster_id
                cluster_data = feature_data[cluster_mask]
                
                if len(cluster_data) == 0:
                    continue
                
                cluster_mean = np.mean(cluster_data)
                cluster_size = len(cluster_data)
                
                between_cluster_var += cluster_size * (cluster_mean - overall_mean) ** 2
                within_cluster_var += np.sum((cluster_data - cluster_mean) ** 2)
            
            # Calculate F-ratio as importance measure
            if within_cluster_var > 0:
                f_ratio = between_cluster_var / within_cluster_var
                importance[feature] = f_ratio
            else:
                importance[feature] = 0
        
        # Normalize importance scores
        max_importance = max(importance.values()) if importance.values() else 1
        if max_importance > 0:
            importance = {k: v / max_importance for k, v in importance.items()}
        
        return importance


class AnomalyDetection:
    """Anomaly detection for identifying unusual patterns"""
    
    def __init__(self):
        self.logger = get_logger("anomaly_detection")
        self.models = {}
    
    def detect_anomalies(self, df: pd.DataFrame, method: str = 'isolation_forest', contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect anomalies in the dataset
        
        Args:
            df: DataFrame to analyze
            method: Anomaly detection method
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary containing anomaly detection results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for anomaly detection")
        
        # Prepare data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for anomaly detection")
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        if method == 'isolation_forest':
            return self._isolation_forest_detection(df, scaled_data, contamination, scaler)
        elif method == 'statistical':
            return self._statistical_detection(df, numeric_df)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
    
    def _isolation_forest_detection(self, df: pd.DataFrame, scaled_data: np.ndarray, 
                                  contamination: float, scaler) -> Dict[str, Any]:
        """Isolation Forest anomaly detection"""
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        
        # Convert labels: -1 (anomaly) to 1, 1 (normal) to 0
        is_anomaly = (anomaly_labels == -1)
        
        # Calculate anomaly scores
        anomaly_scores = iso_forest.decision_function(scaled_data)
        
        # Store model
        self.models['isolation_forest'] = {
            'model': iso_forest,
            'scaler': scaler
        }
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df['is_anomaly'] = is_anomaly
        result_df['anomaly_score'] = anomaly_scores
        
        # Generate anomaly summary
        n_anomalies = is_anomaly.sum()
        anomaly_percentage = n_anomalies / len(df) * 100
        
        anomaly_summary = {
            'total_anomalies': int(n_anomalies),
            'anomaly_percentage': anomaly_percentage,
            'contamination_used': contamination,
            'method': 'isolation_forest'
        }
        
        self.logger.info(f"Isolation Forest detected {n_anomalies} anomalies ({anomaly_percentage:.2f}%)")
        
        return {
            'method': 'isolation_forest',
            'anomalies_detected': int(n_anomalies),
            'anomaly_percentage': anomaly_percentage,
            'anomaly_data': result_df[is_anomaly],
            'normal_data': result_df[~is_anomaly],
            'full_data': result_df,
            'anomaly_summary': anomaly_summary,
            'anomaly_scores': anomaly_scores
        }
    
    def _statistical_detection(self, df: pd.DataFrame, numeric_df: pd.DataFrame) -> Dict[str, Any]:
        """Statistical anomaly detection using Z-score and IQR"""
        
        anomalies_z = pd.DataFrame(index=df.index)
        anomalies_iqr = pd.DataFrame(index=df.index)
        
        # Z-score based detection
        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        anomalies_z['is_anomaly_z'] = (z_scores > 3).any(axis=1)
        
        # IQR based detection
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies_iqr['is_anomaly_iqr'] = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)
        
        # Combine both methods
        combined_anomalies = anomalies_z['is_anomaly_z'] | anomalies_iqr['is_anomaly_iqr']
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df['is_anomaly'] = combined_anomalies
        result_df['is_anomaly_z'] = anomalies_z['is_anomaly_z']
        result_df['is_anomaly_iqr'] = anomalies_iqr['is_anomaly_iqr']
        
        # Generate summary
        n_anomalies = combined_anomalies.sum()
        anomaly_percentage = n_anomalies / len(df) * 100
        
        anomaly_summary = {
            'total_anomalies': int(n_anomalies),
            'anomaly_percentage': anomaly_percentage,
            'z_score_anomalies': int(anomalies_z['is_anomaly_z'].sum()),
            'iqr_anomalies': int(anomalies_iqr['is_anomaly_iqr'].sum()),
            'method': 'statistical'
        }
        
        self.logger.info(f"Statistical method detected {n_anomalies} anomalies ({anomaly_percentage:.2f}%)")
        
        return {
            'method': 'statistical',
            'anomalies_detected': int(n_anomalies),
            'anomaly_percentage': anomaly_percentage,
            'anomaly_data': result_df[combined_anomalies],
            'normal_data': result_df[~combined_anomalies],
            'full_data': result_df,
            'anomaly_summary': anomaly_summary
        }


class DimensionalityReduction:
    """Dimensionality reduction for data visualization and analysis"""
    
    def __init__(self):
        self.logger = get_logger("dimensionality_reduction")
        self.models = {}
    
    def perform_pca(self, df: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
        """
        Perform Principal Component Analysis
        
        Args:
            df: DataFrame to analyze
            n_components: Number of components to extract
            
        Returns:
            Dictionary containing PCA results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for PCA")
        
        # Prepare data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for PCA")
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Perform PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca_data = pca.fit_transform(scaled_data)
        
        # Store model
        self.models['pca'] = {
            'model': pca,
            'scaler': scaler
        }
        
        # Create results DataFrame
        pca_df = df.copy()
        for i in range(n_components):
            pca_df[f'PC{i+1}'] = pca_data[:, i]
        
        # Calculate metrics
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        self.logger.info(f"PCA completed. Explained variance: {explained_variance_ratio}")
        
        return {
            'method': 'pca',
            'n_components': n_components,
            'transformed_data': pca_df,
            'pca_components': pca_data,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'feature_loadings': pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=numeric_df.columns
            )
        }
    
    def perform_tsne(self, df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0) -> Dict[str, Any]:
        """
        Perform t-SNE analysis
        
        Args:
            df: DataFrame to analyze
            n_components: Number of components to extract
            perplexity: t-SNE perplexity parameter
            
        Returns:
            Dictionary containing t-SNE results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for t-SNE")
        
        # Prepare data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for t-SNE")
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Perform t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, max_iter=1000)
        tsne_data = tsne.fit_transform(scaled_data)
        
        # Store model
        self.models['tsne'] = {
            'model': tsne,
            'scaler': scaler
        }
        
        # Create results DataFrame
        tsne_df = df.copy()
        for i in range(n_components):
            tsne_df[f'tSNE{i+1}'] = tsne_data[:, i]
        
        self.logger.info(f"t-SNE completed with perplexity={perplexity}")
        
        return {
            'method': 'tsne',
            'n_components': n_components,
            'transformed_data': tsne_df,
            'tsne_components': tsne_data,
            'perplexity': perplexity
        }


class AnalysisCore:
    """
    Core analysis engine combining all analysis capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analysis core
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("analysis_core")
        
        # Initialize analysis components
        self.clustering = ClusteringAnalysis()
        self.anomaly_detection = AnomalyDetection()
        self.dimensionality_reduction = DimensionalityReduction()
        
        # Results storage
        self.analysis_results = {}
    
    def comprehensive_analysis(self, df: pd.DataFrame, 
                             include_clustering: bool = True,
                             include_anomaly_detection: bool = True,
                             include_dimensionality_reduction: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on the dataset
        
        Args:
            df: DataFrame to analyze
            include_clustering: Whether to perform clustering analysis
            include_anomaly_detection: Whether to perform anomaly detection
            include_dimensionality_reduction: Whether to perform dimensionality reduction
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting comprehensive analysis")
        
        results = {
            'dataset_info': self._get_dataset_info(df),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            # Clustering Analysis
            if include_clustering:
                self.logger.info("Performing clustering analysis")
                clustering_results = {}
                
                # K-Means clustering
                try:
                    kmeans_result = self.clustering.perform_kmeans_clustering(df)
                    clustering_results['kmeans'] = kmeans_result
                except Exception as e:
                    self.logger.error(f"K-Means clustering failed: {str(e)}")
                
                # DBSCAN clustering
                try:
                    dbscan_result = self.clustering.perform_dbscan_clustering(df)
                    clustering_results['dbscan'] = dbscan_result
                except Exception as e:
                    self.logger.error(f"DBSCAN clustering failed: {str(e)}")
                
                results['clustering'] = clustering_results
            
            # Anomaly Detection
            if include_anomaly_detection:
                self.logger.info("Performing anomaly detection")
                anomaly_results = {}
                
                # Isolation Forest
                try:
                    iso_result = self.anomaly_detection.detect_anomalies(df, method='isolation_forest')
                    anomaly_results['isolation_forest'] = iso_result
                except Exception as e:
                    self.logger.error(f"Isolation Forest anomaly detection failed: {str(e)}")
                
                # Statistical methods
                try:
                    stat_result = self.anomaly_detection.detect_anomalies(df, method='statistical')
                    anomaly_results['statistical'] = stat_result
                except Exception as e:
                    self.logger.error(f"Statistical anomaly detection failed: {str(e)}")
                
                results['anomaly_detection'] = anomaly_results
            
            # Dimensionality Reduction
            if include_dimensionality_reduction:
                self.logger.info("Performing dimensionality reduction")
                dim_reduction_results = {}
                
                # PCA
                try:
                    pca_result = self.dimensionality_reduction.perform_pca(df)
                    dim_reduction_results['pca'] = pca_result
                except Exception as e:
                    self.logger.error(f"PCA failed: {str(e)}")
                
                # t-SNE
                try:
                    tsne_result = self.dimensionality_reduction.perform_tsne(df)
                    dim_reduction_results['tsne'] = tsne_result
                except Exception as e:
                    self.logger.error(f"t-SNE failed: {str(e)}")
                
                results['dimensionality_reduction'] = dim_reduction_results
            
            # Store results
            self.analysis_results = results
            
            self.logger.info("Comprehensive analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise
        
        return results
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'missing_values': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all analysis results
        
        Returns:
            Dictionary containing analysis summary
        """
        if not self.analysis_results:
            return {'message': 'No analysis results available'}
        
        summary = {
            'dataset_overview': self.analysis_results.get('dataset_info', {}),
            'analyses_performed': list(self.analysis_results.keys())
        }
        
        # Clustering summary
        if 'clustering' in self.analysis_results:
            clustering_data = self.analysis_results['clustering']
            summary['clustering_summary'] = {}
            
            for method, results in clustering_data.items():
                if 'n_clusters' in results:
                    summary['clustering_summary'][method] = {
                        'n_clusters': results['n_clusters'],
                        'silhouette_score': results.get('silhouette_score'),
                        'anomalies_detected': results.get('n_noise', 0)  # For DBSCAN
                    }
        
        # Anomaly detection summary
        if 'anomaly_detection' in self.analysis_results:
            anomaly_data = self.analysis_results['anomaly_detection']
            summary['anomaly_summary'] = {}
            
            for method, results in anomaly_data.items():
                summary['anomaly_summary'][method] = {
                    'anomalies_detected': results.get('anomalies_detected', 0),
                    'anomaly_percentage': results.get('anomaly_percentage', 0)
                }
        
        # Dimensionality reduction summary
        if 'dimensionality_reduction' in self.analysis_results:
            dim_data = self.analysis_results['dimensionality_reduction']
            summary['dimensionality_reduction_summary'] = {}
            
            if 'pca' in dim_data:
                pca_results = dim_data['pca']
                summary['dimensionality_reduction_summary']['pca'] = {
                    'explained_variance': pca_results.get('explained_variance_ratio', []),
                    'cumulative_variance': pca_results.get('cumulative_variance', [])
                }
        
        return summary