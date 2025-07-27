"""
机器学习分析核心模块

提供聚类分析、异常检测、降维、关联规则挖掘等机器学习功能
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from mlxtend.frequent_patterns import apriori, association_rules
from typing import Dict, Any, Optional, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')


class AnalysisCore:
    """机器学习分析核心类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化分析核心"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 聚类配置
        self.clustering_config = config.get('clustering', {})
        self.default_algorithm = self.clustering_config.get('default_algorithm', 'kmeans')
        self.max_clusters = self.clustering_config.get('max_clusters', 10)
        
        # 异常检测配置
        self.anomaly_config = config.get('anomaly_detection', {})
        self.anomaly_algorithms = self.anomaly_config.get('algorithms', ['isolation_forest'])
        self.contamination = self.anomaly_config.get('contamination', 0.1)
        
        # 降维配置
        self.dimred_config = config.get('dimensionality_reduction', {})
        self.dimred_methods = self.dimred_config.get('methods', ['pca'])
        self.n_components = self.dimred_config.get('n_components', 2)
        
        # 已训练的模型
        self.trained_models = {}
        
        self.logger.info("机器学习分析核心初始化完成")
    
    async def cluster_analysis(self, data: pd.DataFrame, 
                             algorithm: Optional[str] = None) -> Dict[str, Any]:
        """聚类分析"""
        try:
            self.logger.info(f"开始聚类分析，数据形状: {data.shape}")
            
            # 选择数值列
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ValueError("数据中没有数值列可用于聚类")
            
            algorithm = algorithm or self.default_algorithm
            
            # 确定最优聚类数
            optimal_clusters = self._find_optimal_clusters(numeric_data)
            
            # 执行聚类
            if algorithm == 'kmeans':
                result = self._kmeans_clustering(numeric_data, optimal_clusters)
            elif algorithm == 'dbscan':
                result = self._dbscan_clustering(numeric_data)
            elif algorithm == 'hierarchical':
                result = self._hierarchical_clustering(numeric_data, optimal_clusters)
            else:
                raise ValueError(f"不支持的聚类算法: {algorithm}")
            
            # 添加聚类结果到原始数据
            data_with_clusters = data.copy()
            data_with_clusters['cluster'] = result['labels']
            
            # 计算聚类统计
            cluster_stats = self._calculate_cluster_statistics(data_with_clusters, numeric_data.columns)
            
            result.update({
                'cluster_statistics': cluster_stats,
                'algorithm': algorithm,
                'n_clusters': len(np.unique(result['labels'])),
                'data_shape': data.shape
            })
            
            self.logger.info(f"聚类分析完成，发现 {result['n_clusters']} 个簇")
            return result
            
        except Exception as e:
            self.logger.error(f"聚类分析失败: {e}")
            raise
    
    def _find_optimal_clusters(self, data: pd.DataFrame) -> int:
        """使用肘部法则和轮廓系数找到最优聚类数"""
        if len(data) < 4:
            return 2
        
        max_k = min(self.max_clusters, len(data) // 2)
        inertias = []
        silhouette_scores = []
        
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                
                inertias.append(kmeans.inertia_)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(data, labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(0)
                    
            except Exception:
                silhouette_scores.append(0)
        
        # 找到轮廓系数最大的k值
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3
        
        self.logger.debug(f"最优聚类数: {optimal_k}")
        return optimal_k
    
    def _kmeans_clustering(self, data: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
        """K-Means聚类"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # 计算评估指标
        silhouette_avg = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else 0
        calinski_harabasz = calinski_harabasz_score(data, labels) if len(np.unique(labels)) > 1 else 0
        
        self.trained_models['kmeans'] = kmeans
        
        return {
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'model': kmeans
        }
    
    def _dbscan_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """DBSCAN聚类"""
        # 自动选择参数
        eps = self._estimate_dbscan_eps(data)
        min_samples = max(2, len(data) // 50)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # 计算评估指标
        if n_clusters > 1:
            silhouette_avg = silhouette_score(data, labels)
        else:
            silhouette_avg = 0
        
        self.trained_models['dbscan'] = dbscan
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples,
            'silhouette_score': silhouette_avg,
            'model': dbscan
        }
    
    def _estimate_dbscan_eps(self, data: pd.DataFrame) -> float:
        """估计DBSCAN的eps参数"""
        from sklearn.neighbors import NearestNeighbors
        
        k = min(4, len(data) // 10)
        if k < 2:
            k = 2
            
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(data)
        distances, indices = nbrs.kneighbors(data)
        
        distances = np.sort(distances[:, k-1], axis=0)
        
        # 使用第95百分位数作为eps
        eps = np.percentile(distances, 95)
        return eps
    
    def _hierarchical_clustering(self, data: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
        """层次聚类"""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(data)
        
        # 计算评估指标
        silhouette_avg = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else 0
        
        self.trained_models['hierarchical'] = hierarchical
        
        return {
            'labels': labels,
            'silhouette_score': silhouette_avg,
            'model': hierarchical
        }
    
    def _calculate_cluster_statistics(self, data: pd.DataFrame, 
                                    numeric_columns: List[str]) -> Dict[str, Any]:
        """计算聚类统计信息"""
        stats = {}
        
        for cluster_id in data['cluster'].unique():
            cluster_data = data[data['cluster'] == cluster_id]
            
            stats[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
                'numeric_stats': cluster_data[numeric_columns].describe().to_dict()
            }
        
        return stats
    
    async def anomaly_detection(self, data: pd.DataFrame, 
                              algorithms: Optional[List[str]] = None) -> Dict[str, Any]:
        """异常检测"""
        try:
            self.logger.info(f"开始异常检测，数据形状: {data.shape}")
            
            # 选择数值列
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ValueError("数据中没有数值列可用于异常检测")
            
            algorithms = algorithms or self.anomaly_algorithms
            results = {}
            
            for algorithm in algorithms:
                if algorithm == 'isolation_forest':
                    result = self._isolation_forest_detection(numeric_data)
                elif algorithm == 'one_class_svm':
                    result = self._one_class_svm_detection(numeric_data)
                else:
                    self.logger.warning(f"不支持的异常检测算法: {algorithm}")
                    continue
                
                results[algorithm] = result
            
            # 合并异常检测结果
            combined_result = self._combine_anomaly_results(results, data)
            
            self.logger.info(f"异常检测完成，发现 {combined_result['n_anomalies']} 个异常点")
            return combined_result
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            raise
    
    def _isolation_forest_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """孤立森林异常检测"""
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = iso_forest.fit_predict(data)
        anomaly_scores = iso_forest.decision_function(data)
        
        self.trained_models['isolation_forest'] = iso_forest
        
        return {
            'labels': anomaly_labels,
            'scores': anomaly_scores,
            'n_anomalies': np.sum(anomaly_labels == -1),
            'model': iso_forest
        }
    
    def _one_class_svm_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """One-Class SVM异常检测"""
        try:
            # 对于大数据集，使用采样
            if len(data) > 10000:
                sample_data = data.sample(n=10000, random_state=42)
            else:
                sample_data = data
            
            svm = OneClassSVM(nu=self.contamination, kernel='rbf', gamma='scale')
            svm.fit(sample_data)
            
            anomaly_labels = svm.predict(data)
            anomaly_scores = svm.decision_function(data)
            
            self.trained_models['one_class_svm'] = svm
            
            return {
                'labels': anomaly_labels,
                'scores': anomaly_scores,
                'n_anomalies': np.sum(anomaly_labels == -1),
                'model': svm
            }
            
        except Exception as e:
            self.logger.warning(f"One-Class SVM检测失败: {e}")
            return {
                'labels': np.ones(len(data)),
                'scores': np.zeros(len(data)),
                'n_anomalies': 0,
                'error': str(e)
            }
    
    def _combine_anomaly_results(self, results: Dict[str, Any], 
                               original_data: pd.DataFrame) -> Dict[str, Any]:
        """合并异常检测结果"""
        n_samples = len(original_data)
        combined_scores = np.zeros(n_samples)
        combined_labels = np.ones(n_samples)
        
        # 投票机制
        for algorithm, result in results.items():
            if 'error' not in result:
                combined_scores += result['scores']
                combined_labels += (result['labels'] == -1).astype(int)
        
        # 归一化分数
        if len(results) > 0:
            combined_scores /= len(results)
            
        # 确定最终异常点（超过一半算法认为是异常）
        threshold = len(results) / 2
        final_anomalies = combined_labels > threshold
        
        # 创建结果数据框
        anomaly_data = original_data.copy()
        anomaly_data['anomaly_score'] = combined_scores
        anomaly_data['is_anomaly'] = final_anomalies
        
        return {
            'individual_results': results,
            'combined_scores': combined_scores,
            'final_labels': final_anomalies,
            'n_anomalies': np.sum(final_anomalies),
            'anomaly_percentage': np.sum(final_anomalies) / n_samples * 100,
            'anomaly_data': anomaly_data[final_anomalies] if np.any(final_anomalies) else pd.DataFrame()
        }
    
    async def dimensionality_reduction(self, data: pd.DataFrame, 
                                     methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """降维分析"""
        try:
            self.logger.info(f"开始降维分析，数据形状: {data.shape}")
            
            # 选择数值列
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ValueError("数据中没有数值列可用于降维")
            
            methods = methods or self.dimred_methods
            results = {}
            
            for method in methods:
                if method == 'pca':
                    result = self._pca_reduction(numeric_data)
                elif method == 'tsne':
                    result = self._tsne_reduction(numeric_data)
                else:
                    self.logger.warning(f"不支持的降维方法: {method}")
                    continue
                
                results[method] = result
            
            self.logger.info("降维分析完成")
            return results
            
        except Exception as e:
            self.logger.error(f"降维分析失败: {e}")
            raise
    
    def _pca_reduction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """PCA降维"""
        n_components = min(self.n_components, data.shape[1], data.shape[0])
        
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        
        # 计算解释方差比
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        self.trained_models['pca'] = pca
        
        return {
            'transformed_data': transformed_data,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'components': pca.components_,
            'feature_names': [f'PC{i+1}' for i in range(n_components)],
            'model': pca
        }
    
    def _tsne_reduction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """t-SNE降维"""
        # 对于大数据集，先用PCA降维
        if data.shape[1] > 50:
            pca = PCA(n_components=50)
            data = pca.fit_transform(data)
        
        # 限制样本数量以提高性能
        if len(data) > 5000:
            indices = np.random.choice(len(data), 5000, replace=False)
            sample_data = data.iloc[indices] if isinstance(data, pd.DataFrame) else data[indices]
        else:
            sample_data = data
            indices = np.arange(len(data))
        
        tsne = TSNE(
            n_components=min(self.n_components, 3),
            random_state=42,
            perplexity=min(30, len(sample_data) // 4)
        )
        
        transformed_sample = tsne.fit_transform(sample_data)
        
        # 为所有数据创建结果（对于未采样的数据使用最近邻插值）
        if len(indices) < len(data):
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(sample_data)
            _, neighbor_indices = nn.kneighbors(data)
            transformed_data = transformed_sample[neighbor_indices.flatten()]
        else:
            transformed_data = transformed_sample
        
        return {
            'transformed_data': transformed_data,
            'feature_names': [f'tSNE{i+1}' for i in range(transformed_data.shape[1])],
            'sample_indices': indices,
            'model': tsne
        }
    
    async def association_rules(self, data: pd.DataFrame, 
                              min_support: float = 0.01,
                              min_confidence: float = 0.5) -> Dict[str, Any]:
        """关联规则挖掘"""
        try:
            self.logger.info(f"开始关联规则挖掘，数据形状: {data.shape}")
            
            # 将数据转换为事务格式
            transaction_data = self._prepare_transaction_data(data)
            
            if transaction_data.empty:
                return {'error': '无法生成事务数据'}
            
            # 找到频繁项集
            frequent_itemsets = apriori(
                transaction_data, 
                min_support=min_support, 
                use_colnames=True
            )
            
            if frequent_itemsets.empty:
                return {'frequent_itemsets': pd.DataFrame(), 'rules': pd.DataFrame()}
            
            # 生成关联规则
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence
            )
            
            # 计算额外的度量
            if not rules.empty:
                rules = self._calculate_additional_metrics(rules)
            
            result = {
                'frequent_itemsets': frequent_itemsets,
                'rules': rules,
                'n_frequent_itemsets': len(frequent_itemsets),
                'n_rules': len(rules),
                'parameters': {
                    'min_support': min_support,
                    'min_confidence': min_confidence
                }
            }
            
            self.logger.info(f"关联规则挖掘完成，发现 {len(rules)} 条规则")
            return result
            
        except Exception as e:
            self.logger.error(f"关联规则挖掘失败: {e}")
            return {'error': str(e)}
    
    def _prepare_transaction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备事务数据"""
        # 将分类列转换为独热编码
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            # 如果没有分类列，将数值列离散化
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            discrete_data = pd.DataFrame()
            
            for col in numeric_cols:
                # 使用分位数离散化
                discrete_data[f'{col}_low'] = data[col] <= data[col].quantile(0.33)
                discrete_data[f'{col}_medium'] = (data[col] > data[col].quantile(0.33)) & (data[col] <= data[col].quantile(0.67))
                discrete_data[f'{col}_high'] = data[col] > data[col].quantile(0.67)
            
            return discrete_data.astype(bool)
        else:
            # 使用独热编码处理分类列
            encoded_data = pd.get_dummies(data[categorical_cols], prefix_sep='_')
            return encoded_data.astype(bool)
    
    def _calculate_additional_metrics(self, rules: pd.DataFrame) -> pd.DataFrame:
        """计算额外的关联规则度量"""
        # 计算提升度的置信区间等额外指标
        rules['leverage'] = rules['support'] - (rules['antecedent support'] * rules['consequent support'])
        rules['conviction'] = (1 - rules['consequent support']) / (1 - rules['confidence'])
        
        # 排序规则
        rules = rules.sort_values(['confidence', 'lift', 'support'], ascending=[False, False, False])
        
        return rules