"""
CCGL Analytics - Analysis Core Module
Comprehensive machine learning analysis including clustering, anomaly detection, and dimensionality reduction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import warnings

# Machine Learning imports
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Association rule mining
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

from ..utils.logger import get_logger, LoggerMixin

class ClusteringAnalyzer(LoggerMixin):
    """Clustering analysis with multiple algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize clustering analyzer.
        
        Args:
            config: Clustering configuration
        """
        self.config = config or {}
        self.clustering_config = self.config.get('analysis', {}).get('clustering', {})
        self.models = {}
        self.results = {}
    
    def perform_clustering(self, data: pd.DataFrame, algorithm: str = 'kmeans',
                         n_clusters: Union[int, str] = 'auto', **kwargs) -> Dict[str, Any]:
        """Perform clustering analysis.
        
        Args:
            data: Input DataFrame
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters or 'auto'
            **kwargs: Additional algorithm parameters
            
        Returns:
            Clustering results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for clustering analysis")
        
        self.logger.info(f"Performing clustering analysis using {algorithm}")
        
        # Prepare data
        numeric_data = self._prepare_clustering_data(data)
        
        if numeric_data.empty:
            return {'error': 'No numeric data available for clustering'}
        
        start_time = time.time()
        
        # Determine optimal number of clusters if auto
        if algorithm in ['kmeans', 'hierarchical'] and n_clusters == 'auto':
            n_clusters = self._determine_optimal_clusters(numeric_data, algorithm)
        
        # Perform clustering
        if algorithm == 'kmeans':
            results = self._kmeans_clustering(numeric_data, n_clusters, **kwargs)
        elif algorithm == 'dbscan':
            results = self._dbscan_clustering(numeric_data, **kwargs)
        elif algorithm == 'hierarchical':
            results = self._hierarchical_clustering(numeric_data, n_clusters, **kwargs)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        # Add metadata
        results.update({
            'algorithm': algorithm,
            'input_shape': data.shape,
            'processed_shape': numeric_data.shape,
            'processing_time': time.time() - start_time,
            'features_used': list(numeric_data.columns)
        })
        
        # Store results
        self.results[algorithm] = results
        
        self.logger.info(f"Clustering analysis completed in {results['processing_time']:.2f} seconds")
        
        return results
    
    def _prepare_clustering_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for clustering.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame ready for clustering
        """
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number]).copy()
        
        # Remove columns with all NaN values
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        # Remove rows with any NaN values
        numeric_data = numeric_data.dropna(axis=0, how='any')
        
        # Standardize the data
        if not numeric_data.empty:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(numeric_data)
            numeric_data = pd.DataFrame(
                scaled_values,
                columns=numeric_data.columns,
                index=numeric_data.index
            )
        
        return numeric_data
    
    def _determine_optimal_clusters(self, data: pd.DataFrame, algorithm: str) -> int:
        """Determine optimal number of clusters.
        
        Args:
            data: Prepared data
            algorithm: Clustering algorithm
            
        Returns:
            Optimal number of clusters
        """
        max_clusters = min(10, len(data) // 2)  # Reasonable upper bound
        
        if max_clusters < 2:
            return 2
        
        if algorithm == 'kmeans':
            return self._elbow_method(data, max_clusters)
        else:
            # Simple heuristic for other algorithms
            return min(8, max(2, len(data) // 50))
    
    def _elbow_method(self, data: pd.DataFrame, max_clusters: int) -> int:
        """Use elbow method to find optimal clusters for K-means.
        
        Args:
            data: Prepared data
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            except:
                break
        
        if len(inertias) < 2:
            return 3  # Default fallback
        
        # Find elbow point using difference of differences
        diffs = np.diff(inertias)
        if len(diffs) > 1:
            diff_diffs = np.diff(diffs)
            elbow_index = np.argmax(diff_diffs) + 2  # +2 because we start from k=2
            return min(elbow_index, max_clusters)
        
        return 3  # Default fallback
    
    def _kmeans_clustering(self, data: pd.DataFrame, n_clusters: int, **kwargs) -> Dict[str, Any]:
        """Perform K-means clustering.
        
        Args:
            data: Prepared data
            n_clusters: Number of clusters
            **kwargs: Additional parameters
            
        Returns:
            K-means results
        """
        # Default parameters
        params = {
            'n_clusters': n_clusters,
            'random_state': 42,
            'n_init': 10,
            'max_iter': 300
        }
        params.update(kwargs)
        
        # Fit model
        kmeans = KMeans(**params)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate metrics
        metrics = self._calculate_clustering_metrics(data, cluster_labels)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, cluster_labels)
        
        results = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': float(kmeans.inertia_),
            'metrics': metrics,
            'cluster_analysis': cluster_analysis
        }
        
        # Store model
        self.models['kmeans'] = kmeans
        
        return results
    
    def _dbscan_clustering(self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5, **kwargs) -> Dict[str, Any]:
        """Perform DBSCAN clustering.
        
        Args:
            data: Prepared data
            eps: Maximum distance between two samples
            min_samples: Minimum number of samples in a neighborhood
            **kwargs: Additional parameters
            
        Returns:
            DBSCAN results
        """
        # Default parameters
        params = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': 'euclidean'
        }
        params.update(kwargs)
        
        # Fit model
        dbscan = DBSCAN(**params)
        cluster_labels = dbscan.fit_predict(data)
        
        # Calculate number of clusters (excluding noise points)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate metrics (excluding noise points)
        if n_clusters > 1:
            valid_indices = cluster_labels != -1
            if valid_indices.sum() > 0:
                metrics = self._calculate_clustering_metrics(
                    data[valid_indices], cluster_labels[valid_indices]
                )
            else:
                metrics = {}
        else:
            metrics = {}
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, cluster_labels)
        
        results = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'cluster_labels': cluster_labels.tolist(),
            'eps': eps,
            'min_samples': min_samples,
            'metrics': metrics,
            'cluster_analysis': cluster_analysis
        }
        
        # Store model
        self.models['dbscan'] = dbscan
        
        return results
    
    def _hierarchical_clustering(self, data: pd.DataFrame, n_clusters: int, linkage: str = 'ward', **kwargs) -> Dict[str, Any]:
        """Perform hierarchical clustering.
        
        Args:
            data: Prepared data
            n_clusters: Number of clusters
            linkage: Linkage criterion
            **kwargs: Additional parameters
            
        Returns:
            Hierarchical clustering results
        """
        # Default parameters
        params = {
            'n_clusters': n_clusters,
            'linkage': linkage
        }
        params.update(kwargs)
        
        # Fit model
        hierarchical = AgglomerativeClustering(**params)
        cluster_labels = hierarchical.fit_predict(data)
        
        # Calculate metrics
        metrics = self._calculate_clustering_metrics(data, cluster_labels)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, cluster_labels)
        
        results = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'linkage': linkage,
            'metrics': metrics,
            'cluster_analysis': cluster_analysis
        }
        
        # Store model
        self.models['hierarchical'] = hierarchical
        
        return results
    
    def _calculate_clustering_metrics(self, data: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics.
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            Clustering metrics
        """
        metrics = {}
        
        try:
            # Silhouette score
            if len(set(labels)) > 1:
                metrics['silhouette_score'] = float(silhouette_score(data, labels))
                metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(data, labels))
                metrics['davies_bouldin_score'] = float(davies_bouldin_score(data, labels))
        except Exception as e:
            self.logger.warning(f"Failed to calculate some clustering metrics: {e}")
        
        return metrics
    
    def _analyze_clusters(self, data: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics.
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            Cluster analysis
        """
        analysis = {
            'cluster_sizes': {},
            'cluster_centers': {},
            'cluster_spreads': {}
        }
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
                
            cluster_data = data[labels == label]
            
            analysis['cluster_sizes'][str(label)] = len(cluster_data)
            analysis['cluster_centers'][str(label)] = cluster_data.mean().to_dict()
            analysis['cluster_spreads'][str(label)] = cluster_data.std().to_dict()
        
        return analysis

class AnomalyDetector(LoggerMixin):
    """Anomaly detection with multiple algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config or {}
        self.anomaly_config = self.config.get('analysis', {}).get('anomaly_detection', {})
        self.models = {}
        self.results = {}
    
    def detect_anomalies(self, data: pd.DataFrame, algorithm: str = 'isolation_forest',
                        contamination: float = 0.1, **kwargs) -> Dict[str, Any]:
        """Detect anomalies in the data.
        
        Args:
            data: Input DataFrame
            algorithm: Detection algorithm ('isolation_forest', 'local_outlier_factor', 'one_class_svm')
            contamination: Expected proportion of outliers
            **kwargs: Additional algorithm parameters
            
        Returns:
            Anomaly detection results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for anomaly detection")
        
        self.logger.info(f"Detecting anomalies using {algorithm}")
        
        # Prepare data
        numeric_data = self._prepare_anomaly_data(data)
        
        if numeric_data.empty:
            return {'error': 'No numeric data available for anomaly detection'}
        
        start_time = time.time()
        
        # Perform anomaly detection
        if algorithm == 'isolation_forest':
            results = self._isolation_forest_detection(numeric_data, contamination, **kwargs)
        elif algorithm == 'local_outlier_factor':
            results = self._lof_detection(numeric_data, contamination, **kwargs)
        elif algorithm == 'one_class_svm':
            results = self._one_class_svm_detection(numeric_data, **kwargs)
        else:
            raise ValueError(f"Unsupported anomaly detection algorithm: {algorithm}")
        
        # Add metadata
        results.update({
            'algorithm': algorithm,
            'input_shape': data.shape,
            'processed_shape': numeric_data.shape,
            'processing_time': time.time() - start_time,
            'features_used': list(numeric_data.columns),
            'contamination': contamination
        })
        
        # Store results
        self.results[algorithm] = results
        
        self.logger.info(f"Anomaly detection completed in {results['processing_time']:.2f} seconds")
        
        return results
    
    def _prepare_anomaly_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for anomaly detection.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number]).copy()
        
        # Remove columns with all NaN values
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        # Remove rows with any NaN values
        numeric_data = numeric_data.dropna(axis=0, how='any')
        
        # Standardize the data
        if not numeric_data.empty:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(numeric_data)
            numeric_data = pd.DataFrame(
                scaled_values,
                columns=numeric_data.columns,
                index=numeric_data.index
            )
        
        return numeric_data
    
    def _isolation_forest_detection(self, data: pd.DataFrame, contamination: float, **kwargs) -> Dict[str, Any]:
        """Perform anomaly detection using Isolation Forest.
        
        Args:
            data: Prepared data
            contamination: Expected proportion of outliers
            **kwargs: Additional parameters
            
        Returns:
            Detection results
        """
        # Default parameters
        params = {
            'contamination': contamination,
            'random_state': 42,
            'n_estimators': 100
        }
        params.update(kwargs)
        
        # Fit model
        iso_forest = IsolationForest(**params)
        predictions = iso_forest.fit_predict(data)
        scores = iso_forest.score_samples(data)
        
        # Analyze results
        anomalies = predictions == -1
        anomaly_count = anomalies.sum()
        
        results = {
            'predictions': predictions.tolist(),
            'anomaly_scores': scores.tolist(),
            'anomaly_labels': anomalies.tolist(),
            'anomaly_count': int(anomaly_count),
            'anomaly_percentage': float(anomaly_count / len(data) * 100),
            'anomaly_indices': data.index[anomalies].tolist(),
            'normal_count': int((predictions == 1).sum())
        }
        
        # Store model
        self.models['isolation_forest'] = iso_forest
        
        return results
    
    def _lof_detection(self, data: pd.DataFrame, contamination: float, n_neighbors: int = 20, **kwargs) -> Dict[str, Any]:
        """Perform anomaly detection using Local Outlier Factor.
        
        Args:
            data: Prepared data
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors
            **kwargs: Additional parameters
            
        Returns:
            Detection results
        """
        # Default parameters
        params = {
            'contamination': contamination,
            'n_neighbors': min(n_neighbors, len(data) - 1)
        }
        params.update(kwargs)
        
        # Fit model
        lof = LocalOutlierFactor(**params)
        predictions = lof.fit_predict(data)
        scores = lof.negative_outlier_factor_
        
        # Analyze results
        anomalies = predictions == -1
        anomaly_count = anomalies.sum()
        
        results = {
            'predictions': predictions.tolist(),
            'outlier_factors': scores.tolist(),
            'anomaly_labels': anomalies.tolist(),
            'anomaly_count': int(anomaly_count),
            'anomaly_percentage': float(anomaly_count / len(data) * 100),
            'anomaly_indices': data.index[anomalies].tolist(),
            'normal_count': int((predictions == 1).sum()),
            'n_neighbors': params['n_neighbors']
        }
        
        # Store model
        self.models['local_outlier_factor'] = lof
        
        return results
    
    def _one_class_svm_detection(self, data: pd.DataFrame, nu: float = 0.1, **kwargs) -> Dict[str, Any]:
        """Perform anomaly detection using One-Class SVM.
        
        Args:
            data: Prepared data
            nu: Upper bound on the fraction of training errors
            **kwargs: Additional parameters
            
        Returns:
            Detection results
        """
        # Default parameters
        params = {
            'nu': nu,
            'kernel': 'rbf',
            'gamma': 'scale'
        }
        params.update(kwargs)
        
        # Fit model
        one_class_svm = OneClassSVM(**params)
        predictions = one_class_svm.fit_predict(data)
        scores = one_class_svm.score_samples(data)
        
        # Analyze results
        anomalies = predictions == -1
        anomaly_count = anomalies.sum()
        
        results = {
            'predictions': predictions.tolist(),
            'decision_scores': scores.tolist(),
            'anomaly_labels': anomalies.tolist(),
            'anomaly_count': int(anomaly_count),
            'anomaly_percentage': float(anomaly_count / len(data) * 100),
            'anomaly_indices': data.index[anomalies].tolist(),
            'normal_count': int((predictions == 1).sum()),
            'nu': nu
        }
        
        # Store model
        self.models['one_class_svm'] = one_class_svm
        
        return results

class DimensionalityReducer(LoggerMixin):
    """Dimensionality reduction with multiple algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dimensionality reducer.
        
        Args:
            config: Dimensionality reduction configuration
        """
        self.config = config or {}
        self.reduction_config = self.config.get('analysis', {}).get('dimensionality_reduction', {})
        self.models = {}
        self.results = {}
    
    def reduce_dimensions(self, data: pd.DataFrame, algorithm: str = 'pca',
                         n_components: int = 2, **kwargs) -> Dict[str, Any]:
        """Perform dimensionality reduction.
        
        Args:
            data: Input DataFrame
            algorithm: Reduction algorithm ('pca', 'tsne', 'umap')
            n_components: Number of components
            **kwargs: Additional algorithm parameters
            
        Returns:
            Dimensionality reduction results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for dimensionality reduction")
        
        self.logger.info(f"Performing dimensionality reduction using {algorithm}")
        
        # Prepare data
        numeric_data = self._prepare_reduction_data(data)
        
        if numeric_data.empty:
            return {'error': 'No numeric data available for dimensionality reduction'}
        
        start_time = time.time()
        
        # Perform reduction
        if algorithm == 'pca':
            results = self._pca_reduction(numeric_data, n_components, **kwargs)
        elif algorithm == 'tsne':
            results = self._tsne_reduction(numeric_data, n_components, **kwargs)
        elif algorithm == 'umap':
            results = self._umap_reduction(numeric_data, n_components, **kwargs)
        else:
            raise ValueError(f"Unsupported dimensionality reduction algorithm: {algorithm}")
        
        # Add metadata
        results.update({
            'algorithm': algorithm,
            'input_shape': data.shape,
            'processed_shape': numeric_data.shape,
            'processing_time': time.time() - start_time,
            'features_used': list(numeric_data.columns),
            'n_components': n_components
        })
        
        # Store results
        self.results[algorithm] = results
        
        self.logger.info(f"Dimensionality reduction completed in {results['processing_time']:.2f} seconds")
        
        return results
    
    def _prepare_reduction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for dimensionality reduction.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number]).copy()
        
        # Remove columns with all NaN values
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        # Remove rows with any NaN values
        numeric_data = numeric_data.dropna(axis=0, how='any')
        
        # Standardize the data
        if not numeric_data.empty:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(numeric_data)
            numeric_data = pd.DataFrame(
                scaled_values,
                columns=numeric_data.columns,
                index=numeric_data.index
            )
        
        return numeric_data
    
    def _pca_reduction(self, data: pd.DataFrame, n_components: int, **kwargs) -> Dict[str, Any]:
        """Perform PCA dimensionality reduction.
        
        Args:
            data: Prepared data
            n_components: Number of components
            **kwargs: Additional parameters
            
        Returns:
            PCA results
        """
        # Default parameters
        params = {
            'n_components': min(n_components, min(data.shape) - 1),
            'random_state': 42
        }
        params.update(kwargs)
        
        # Fit model
        pca = PCA(**params)
        transformed_data = pca.fit_transform(data)
        
        # Create column names
        component_names = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
        
        results = {
            'transformed_data': transformed_data.tolist(),
            'component_names': component_names,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist(),
            'feature_importance': self._calculate_feature_importance_pca(pca, data.columns)
        }
        
        # Store model
        self.models['pca'] = pca
        
        return results
    
    def _tsne_reduction(self, data: pd.DataFrame, n_components: int, perplexity: float = 30, **kwargs) -> Dict[str, Any]:
        """Perform t-SNE dimensionality reduction.
        
        Args:
            data: Prepared data
            n_components: Number of components
            perplexity: Perplexity parameter
            **kwargs: Additional parameters
            
        Returns:
            t-SNE results
        """
        # Default parameters
        params = {
            'n_components': n_components,
            'perplexity': min(perplexity, (len(data) - 1) / 3),
            'random_state': 42,
            'max_iter': 1000
        }
        params.update(kwargs)
        
        # Fit model
        tsne = TSNE(**params)
        transformed_data = tsne.fit_transform(data)
        
        # Create column names
        component_names = [f'tSNE{i+1}' for i in range(transformed_data.shape[1])]
        
        results = {
            'transformed_data': transformed_data.tolist(),
            'component_names': component_names,
            'perplexity': params['perplexity'],
            'kl_divergence': float(tsne.kl_divergence_) if hasattr(tsne, 'kl_divergence_') else None
        }
        
        # Store model
        self.models['tsne'] = tsne
        
        return results
    
    def _umap_reduction(self, data: pd.DataFrame, n_components: int, **kwargs) -> Dict[str, Any]:
        """Perform UMAP dimensionality reduction.
        
        Args:
            data: Prepared data
            n_components: Number of components
            **kwargs: Additional parameters
            
        Returns:
            UMAP results
        """
        if not UMAP_AVAILABLE:
            self.logger.warning("UMAP not available, falling back to PCA")
            return self._pca_reduction(data, n_components)
        
        # Default parameters
        params = {
            'n_components': n_components,
            'n_neighbors': 15,
            'min_dist': 0.1,
            'random_state': 42
        }
        params.update(kwargs)
        
        # Fit model
        umap_model = umap.UMAP(**params)
        transformed_data = umap_model.fit_transform(data)
        
        # Create column names
        component_names = [f'UMAP{i+1}' for i in range(transformed_data.shape[1])]
        
        results = {
            'transformed_data': transformed_data.tolist(),
            'component_names': component_names,
            'n_neighbors': params['n_neighbors'],
            'min_dist': params['min_dist']
        }
        
        # Store model
        self.models['umap'] = umap_model
        
        return results
    
    def _calculate_feature_importance_pca(self, pca_model, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for PCA.
        
        Args:
            pca_model: Fitted PCA model
            feature_names: Original feature names
            
        Returns:
            Feature importance scores
        """
        # Calculate feature importance as the sum of absolute loadings weighted by explained variance
        components = np.abs(pca_model.components_)
        explained_variance = pca_model.explained_variance_ratio_
        
        feature_importance = np.sum(components * explained_variance.reshape(-1, 1), axis=0)
        
        return dict(zip(feature_names, feature_importance.tolist()))

class AnalysisCore(LoggerMixin):
    """Main analysis core combining all ML algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analysis core.
        
        Args:
            config: Analysis configuration
        """
        self.config = config or {}
        self.clustering_analyzer = ClusteringAnalyzer(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.dimensionality_reducer = DimensionalityReducer(config)
        self.analysis_results = {}
    
    def comprehensive_analysis(self, data: pd.DataFrame, analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive ML analysis.
        
        Args:
            data: Input DataFrame
            analysis_types: Types of analysis to perform
            
        Returns:
            Comprehensive analysis results
        """
        if analysis_types is None:
            analysis_types = ['clustering', 'anomaly_detection', 'dimensionality_reduction']
        
        self.logger.info(f"Starting comprehensive analysis with types: {analysis_types}")
        
        start_time = time.time()
        results = {
            'input_shape': data.shape,
            'analysis_types': analysis_types,
            'results': {},
            'summary': {}
        }
        
        # Clustering analysis
        if 'clustering' in analysis_types:
            try:
                clustering_results = self.clustering_analyzer.perform_clustering(data)
                results['results']['clustering'] = clustering_results
            except Exception as e:
                self.logger.error(f"Clustering analysis failed: {e}")
                results['results']['clustering'] = {'error': str(e)}
        
        # Anomaly detection
        if 'anomaly_detection' in analysis_types:
            try:
                anomaly_results = self.anomaly_detector.detect_anomalies(data)
                results['results']['anomaly_detection'] = anomaly_results
            except Exception as e:
                self.logger.error(f"Anomaly detection failed: {e}")
                results['results']['anomaly_detection'] = {'error': str(e)}
        
        # Dimensionality reduction
        if 'dimensionality_reduction' in analysis_types:
            try:
                reduction_results = self.dimensionality_reducer.reduce_dimensions(data)
                results['results']['dimensionality_reduction'] = reduction_results
            except Exception as e:
                self.logger.error(f"Dimensionality reduction failed: {e}")
                results['results']['dimensionality_reduction'] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_analysis_summary(results['results'])
        results['total_processing_time'] = time.time() - start_time
        
        # Store results
        self.analysis_results = results
        
        self.logger.info(f"Comprehensive analysis completed in {results['total_processing_time']:.2f} seconds")
        
        return results
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of analysis results.
        
        Args:
            results: Analysis results
            
        Returns:
            Analysis summary
        """
        summary = {
            'successful_analyses': [],
            'failed_analyses': [],
            'key_findings': []
        }
        
        for analysis_type, result in results.items():
            if 'error' in result:
                summary['failed_analyses'].append(analysis_type)
            else:
                summary['successful_analyses'].append(analysis_type)
                
                # Extract key findings
                if analysis_type == 'clustering' and 'n_clusters' in result:
                    summary['key_findings'].append(
                        f"Found {result['n_clusters']} clusters in the data"
                    )
                
                if analysis_type == 'anomaly_detection' and 'anomaly_count' in result:
                    summary['key_findings'].append(
                        f"Detected {result['anomaly_count']} anomalies ({result['anomaly_percentage']:.1f}%)"
                    )
                
                if analysis_type == 'dimensionality_reduction' and 'cumulative_variance_ratio' in result:
                    var_explained = result['cumulative_variance_ratio'][-1] if result['cumulative_variance_ratio'] else 0
                    summary['key_findings'].append(
                        f"Dimensionality reduction explained {var_explained:.1%} of variance"
                    )
        
        return summary