"""
分析核心模块

提供机器学习分析功能，包括聚类分析、异常检测、降维分析和关联规则挖掘。
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class AnalysisCore:
    """分析核心引擎 - 提供各种机器学习分析功能"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化分析核心
        
        Args:
            config: 分析配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.analysis_results = {}
        
    def clustering_analysis(self, df: pd.DataFrame, 
                          methods: List[str] = ['kmeans', 'dbscan', 'hierarchical'],
                          n_clusters: int = None) -> Dict[str, Any]:
        """
        聚类分析
        
        Args:
            df: 输入数据
            methods: 聚类方法列表
            n_clusters: 聚类数量（用于需要指定的算法）
            
        Returns:
            聚类分析结果
        """
        self.logger.info("开始聚类分析")
        
        # 数据预处理
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("没有数值型数据可用于聚类分析")
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df.fillna(numeric_df.mean()))
        
        results = {
            'data_shape': scaled_data.shape,
            'methods_used': methods,
            'clustering_results': {},
            'evaluation_metrics': {},
            'visualizations': {}
        }
        
        # 自动确定最优聚类数
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(scaled_data)
        
        results['optimal_clusters'] = n_clusters
        
        # 执行不同的聚类方法
        for method in methods:
            try:
                cluster_result = self._perform_clustering(scaled_data, method, n_clusters)
                results['clustering_results'][method] = cluster_result
                
                # 评估聚类质量
                if len(set(cluster_result['labels'])) > 1:
                    silhouette = silhouette_score(scaled_data, cluster_result['labels'])
                    calinski = calinski_harabasz_score(scaled_data, cluster_result['labels'])
                    
                    results['evaluation_metrics'][method] = {
                        'silhouette_score': float(silhouette),
                        'calinski_harabasz_score': float(calinski),
                        'n_clusters_found': len(set(cluster_result['labels']))
                    }
                
            except Exception as e:
                self.logger.error(f"聚类方法 {method} 执行失败: {e}")
                results['clustering_results'][method] = {'error': str(e)}
        
        # 生成可视化
        results['visualizations'] = self._generate_clustering_visualizations(
            scaled_data, results['clustering_results'], numeric_df.columns.tolist()
        )
        
        self.analysis_results['clustering'] = results
        return results
    
    def _determine_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """使用肘部法则确定最优聚类数"""
        if len(data) < max_k:
            max_k = len(data) - 1
        
        inertias = []
        k_range = range(2, min(max_k + 1, len(data)))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            except:
                continue
        
        if len(inertias) < 2:
            return 3
        
        # 计算肘部
        differences = np.diff(inertias)
        if len(differences) > 1:
            second_diff = np.diff(differences)
            optimal_idx = np.argmax(second_diff) + 2
            return min(k_range[optimal_idx] if optimal_idx < len(k_range) else 3, max_k)
        
        return 3
    
    def _perform_clustering(self, data: np.ndarray, method: str, n_clusters: int) -> Dict[str, Any]:
        """执行具体的聚类算法"""
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(data)
            return {
                'labels': labels.tolist(),
                'model': model,
                'cluster_centers': model.cluster_centers_.tolist() if hasattr(model, 'cluster_centers_') else None
            }
        
        elif method == 'dbscan':
            # 自动选择合适的eps参数
            eps = self._estimate_eps(data)
            model = DBSCAN(eps=eps, min_samples=5)
            labels = model.fit_predict(data)
            return {
                'labels': labels.tolist(),
                'model': model,
                'eps_used': eps,
                'n_noise_points': int(np.sum(labels == -1))
            }
        
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(data)
            return {
                'labels': labels.tolist(),
                'model': model
            }
        
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
    
    def _estimate_eps(self, data: np.ndarray) -> float:
        """估算DBSCAN的eps参数"""
        from sklearn.neighbors import NearestNeighbors
        
        try:
            k = min(4, len(data) - 1)
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(data)
            distances, indices = neighbors_fit.kneighbors(data)
            distances = np.sort(distances[:, k-1], axis=0)
            
            # 找到距离的急剧变化点
            diff = np.diff(distances)
            knee_point = np.argmax(diff)
            return distances[knee_point]
        except:
            return 0.5
    
    def anomaly_detection(self, df: pd.DataFrame, 
                         methods: List[str] = ['isolation_forest', 'one_class_svm']) -> Dict[str, Any]:
        """
        异常检测分析
        
        Args:
            df: 输入数据
            methods: 异常检测方法列表
            
        Returns:
            异常检测结果
        """
        self.logger.info("开始异常检测分析")
        
        # 数据预处理
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("没有数值型数据可用于异常检测")
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df.fillna(numeric_df.mean()))
        
        results = {
            'data_shape': scaled_data.shape,
            'methods_used': methods,
            'anomaly_results': {},
            'summary': {}
        }
        
        all_anomalies = {}
        
        # 执行不同的异常检测方法
        for method in methods:
            try:
                anomaly_result = self._perform_anomaly_detection(scaled_data, method)
                results['anomaly_results'][method] = anomaly_result
                all_anomalies[method] = anomaly_result['anomaly_indices']
                
            except Exception as e:
                self.logger.error(f"异常检测方法 {method} 执行失败: {e}")
                results['anomaly_results'][method] = {'error': str(e)}
        
        # 生成汇总统计
        if all_anomalies:
            results['summary'] = self._generate_anomaly_summary(all_anomalies, len(df))
        
        self.analysis_results['anomaly_detection'] = results
        return results
    
    def _perform_anomaly_detection(self, data: np.ndarray, method: str) -> Dict[str, Any]:
        """执行具体的异常检测算法"""
        if method == 'isolation_forest':
            model = IsolationForest(contamination=0.1, random_state=42)
            predictions = model.fit_predict(data)
            anomaly_indices = np.where(predictions == -1)[0].tolist()
            
            return {
                'anomaly_indices': anomaly_indices,
                'anomaly_count': len(anomaly_indices),
                'anomaly_ratio': len(anomaly_indices) / len(data),
                'model': model,
                'scores': model.decision_function(data).tolist()
            }
        
        elif method == 'one_class_svm':
            model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
            predictions = model.fit_predict(data)
            anomaly_indices = np.where(predictions == -1)[0].tolist()
            
            return {
                'anomaly_indices': anomaly_indices,
                'anomaly_count': len(anomaly_indices),
                'anomaly_ratio': len(anomaly_indices) / len(data),
                'model': model,
                'scores': model.decision_function(data).tolist()
            }
        
        else:
            raise ValueError(f"不支持的异常检测方法: {method}")
    
    def _generate_anomaly_summary(self, all_anomalies: Dict[str, List[int]], total_samples: int) -> Dict[str, Any]:
        """生成异常检测汇总"""
        # 找到所有方法都认为是异常的点
        if len(all_anomalies) > 1:
            common_anomalies = set.intersection(*[set(anomalies) for anomalies in all_anomalies.values()])
        else:
            common_anomalies = set(list(all_anomalies.values())[0])
        
        # 找到任何方法认为是异常的点
        all_anomaly_indices = set.union(*[set(anomalies) for anomalies in all_anomalies.values()])
        
        return {
            'total_samples': total_samples,
            'common_anomalies': list(common_anomalies),
            'common_anomaly_count': len(common_anomalies),
            'any_method_anomalies': list(all_anomaly_indices),
            'any_method_anomaly_count': len(all_anomaly_indices),
            'consensus_ratio': len(common_anomalies) / len(all_anomaly_indices) if all_anomaly_indices else 0
        }
    
    def dimensionality_reduction(self, df: pd.DataFrame,
                               methods: List[str] = ['pca', 'tsne'],
                               n_components: int = 2) -> Dict[str, Any]:
        """
        降维分析
        
        Args:
            df: 输入数据
            methods: 降维方法列表
            n_components: 降维后的维度数
            
        Returns:
            降维分析结果
        """
        self.logger.info("开始降维分析")
        
        # 数据预处理
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("没有数值型数据可用于降维分析")
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df.fillna(numeric_df.mean()))
        
        results = {
            'original_dimensions': scaled_data.shape[1],
            'target_dimensions': n_components,
            'methods_used': methods,
            'reduction_results': {}
        }
        
        # 执行不同的降维方法
        for method in methods:
            try:
                reduction_result = self._perform_dimensionality_reduction(
                    scaled_data, method, n_components
                )
                results['reduction_results'][method] = reduction_result
                
            except Exception as e:
                self.logger.error(f"降维方法 {method} 执行失败: {e}")
                results['reduction_results'][method] = {'error': str(e)}
        
        self.analysis_results['dimensionality_reduction'] = results
        return results
    
    def _perform_dimensionality_reduction(self, data: np.ndarray, method: str, 
                                        n_components: int) -> Dict[str, Any]:
        """执行具体的降维算法"""
        if method == 'pca':
            model = PCA(n_components=n_components, random_state=42)
            transformed_data = model.fit_transform(data)
            
            return {
                'transformed_data': transformed_data.tolist(),
                'explained_variance_ratio': model.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(model.explained_variance_ratio_).tolist(),
                'model': model,
                'components': model.components_.tolist()
            }
        
        elif method == 'tsne':
            model = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(data)-1))
            transformed_data = model.fit_transform(data)
            
            return {
                'transformed_data': transformed_data.tolist(),
                'model': model,
                'kl_divergence': float(model.kl_divergence_)
            }
        
        else:
            raise ValueError(f"不支持的降维方法: {method}")
    
    def association_rules_mining(self, df: pd.DataFrame,
                                min_support: float = 0.01,
                                min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        关联规则挖掘
        
        Args:
            df: 输入数据（应该是交易数据格式）
            min_support: 最小支持度
            min_confidence: 最小置信度
            
        Returns:
            关联规则挖掘结果
        """
        self.logger.info("开始关联规则挖掘")
        
        try:
            # 准备交易数据
            transactions = self._prepare_transaction_data(df)
            
            if not transactions:
                return {'error': '无法准备交易数据'}
            
            # 编码交易数据
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
            
            # 找频繁项集
            frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                return {
                    'error': f'在支持度 {min_support} 下没有找到频繁项集',
                    'suggestion': '尝试降低最小支持度'
                }
            
            # 生成关联规则
            rules = association_rules(frequent_itemsets, metric="confidence", 
                                    min_threshold=min_confidence)
            
            results = {
                'transaction_count': len(transactions),
                'unique_items': len(te.columns_),
                'frequent_itemsets_count': len(frequent_itemsets),
                'rules_count': len(rules),
                'frequent_itemsets': self._format_frequent_itemsets(frequent_itemsets),
                'association_rules': self._format_association_rules(rules),
                'parameters': {
                    'min_support': min_support,
                    'min_confidence': min_confidence
                }
            }
            
            self.analysis_results['association_rules'] = results
            return results
            
        except Exception as e:
            self.logger.error(f"关联规则挖掘失败: {e}")
            return {'error': str(e)}
    
    def _prepare_transaction_data(self, df: pd.DataFrame) -> List[List[str]]:
        """准备交易数据格式"""
        transactions = []
        
        # 如果数据有明确的交易ID列
        if 'transaction_id' in df.columns or 'order_id' in df.columns:
            id_col = 'transaction_id' if 'transaction_id' in df.columns else 'order_id'
            
            for trans_id in df[id_col].unique():
                trans_data = df[df[id_col] == trans_id]
                # 假设产品信息在 'product' 或 'item' 列中
                if 'product' in df.columns:
                    items = trans_data['product'].tolist()
                elif 'item' in df.columns:
                    items = trans_data['item'].tolist()
                else:
                    # 使用所有非ID、非数值列作为项目
                    item_cols = [col for col in df.columns 
                               if col != id_col and df[col].dtype == 'object']
                    items = []
                    for col in item_cols:
                        items.extend(trans_data[col].dropna().astype(str).tolist())
                
                if items:
                    transactions.append(list(set(items)))  # 去重
        
        else:
            # 如果没有明确的交易ID，将每行作为一个交易
            for idx, row in df.iterrows():
                items = []
                for col in df.columns:
                    if df[col].dtype == 'object' and pd.notna(row[col]):
                        items.append(str(row[col]))
                    elif df[col].dtype in ['int64', 'float64'] and pd.notna(row[col]):
                        # 将数值转换为分类
                        value = row[col]
                        if value > df[col].median():
                            items.append(f"{col}_high")
                        else:
                            items.append(f"{col}_low")
                
                if items:
                    transactions.append(items)
        
        return transactions
    
    def _format_frequent_itemsets(self, frequent_itemsets: pd.DataFrame) -> List[Dict[str, Any]]:
        """格式化频繁项集结果"""
        formatted = []
        for _, row in frequent_itemsets.iterrows():
            formatted.append({
                'itemset': list(row['itemsets']),
                'support': float(row['support']),
                'length': len(row['itemsets'])
            })
        return formatted
    
    def _format_association_rules(self, rules: pd.DataFrame) -> List[Dict[str, Any]]:
        """格式化关联规则结果"""
        formatted = []
        for _, row in rules.iterrows():
            formatted.append({
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift']),
                'conviction': float(row['conviction']) if not np.isinf(row['conviction']) else None
            })
        return formatted
    
    def _generate_clustering_visualizations(self, data: np.ndarray, 
                                          clustering_results: Dict[str, Any],
                                          feature_names: List[str]) -> Dict[str, str]:
        """生成聚类可视化"""
        visualizations = {}
        
        # 如果数据维度大于2，先降维
        if data.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            data_2d = pca.fit_transform(data)
        else:
            data_2d = data
        
        # 为每种聚类方法生成可视化
        for method, result in clustering_results.items():
            if 'error' not in result and 'labels' in result:
                try:
                    # 创建散点图
                    fig = px.scatter(
                        x=data_2d[:, 0], 
                        y=data_2d[:, 1],
                        color=result['labels'],
                        title=f'{method.upper()} 聚类结果',
                        labels={'x': 'PC1' if data.shape[1] > 2 else feature_names[0],
                               'y': 'PC2' if data.shape[1] > 2 else feature_names[1]},
                        color_continuous_scale='viridis'
                    )
                    
                    # 保存为HTML字符串
                    visualizations[method] = fig.to_html(include_plotlyjs='cdn')
                    
                except Exception as e:
                    self.logger.warning(f"生成 {method} 可视化失败: {e}")
        
        return visualizations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合分析报告"""
        report = {
            'generation_time': datetime.now().isoformat(),
            'analyses_performed': list(self.analysis_results.keys()),
            'summary': {},
            'detailed_results': self.analysis_results
        }
        
        # 生成总结
        if 'clustering' in self.analysis_results:
            clustering_summary = self._summarize_clustering_results()
            report['summary']['clustering'] = clustering_summary
        
        if 'anomaly_detection' in self.analysis_results:
            anomaly_summary = self._summarize_anomaly_results()
            report['summary']['anomaly_detection'] = anomaly_summary
        
        if 'dimensionality_reduction' in self.analysis_results:
            reduction_summary = self._summarize_reduction_results()
            report['summary']['dimensionality_reduction'] = reduction_summary
        
        if 'association_rules' in self.analysis_results:
            rules_summary = self._summarize_rules_results()
            report['summary']['association_rules'] = rules_summary
        
        return report
    
    def _summarize_clustering_results(self) -> Dict[str, Any]:
        """总结聚类分析结果"""
        clustering_data = self.analysis_results['clustering']
        
        best_method = None
        best_score = -1
        
        for method, metrics in clustering_data.get('evaluation_metrics', {}).items():
            if 'silhouette_score' in metrics and metrics['silhouette_score'] > best_score:
                best_score = metrics['silhouette_score']
                best_method = method
        
        return {
            'optimal_clusters': clustering_data.get('optimal_clusters'),
            'best_method': best_method,
            'best_silhouette_score': best_score,
            'methods_tested': len(clustering_data.get('clustering_results', {}))
        }
    
    def _summarize_anomaly_results(self) -> Dict[str, Any]:
        """总结异常检测结果"""
        anomaly_data = self.analysis_results['anomaly_detection']
        summary = anomaly_data.get('summary', {})
        
        return {
            'total_samples': summary.get('total_samples', 0),
            'consensus_anomalies': summary.get('common_anomaly_count', 0),
            'any_method_anomalies': summary.get('any_method_anomaly_count', 0),
            'consensus_ratio': summary.get('consensus_ratio', 0),
            'methods_used': len(anomaly_data.get('anomaly_results', {}))
        }
    
    def _summarize_reduction_results(self) -> Dict[str, Any]:
        """总结降维分析结果"""
        reduction_data = self.analysis_results['dimensionality_reduction']
        
        # 获取PCA的解释方差比例
        pca_variance = None
        if 'pca' in reduction_data.get('reduction_results', {}):
            pca_result = reduction_data['reduction_results']['pca']
            if 'cumulative_variance_ratio' in pca_result:
                pca_variance = pca_result['cumulative_variance_ratio'][-1]
        
        return {
            'original_dimensions': reduction_data.get('original_dimensions'),
            'target_dimensions': reduction_data.get('target_dimensions'),
            'pca_variance_explained': pca_variance,
            'methods_used': len(reduction_data.get('reduction_results', {}))
        }
    
    def _summarize_rules_results(self) -> Dict[str, Any]:
        """总结关联规则结果"""
        rules_data = self.analysis_results['association_rules']
        
        return {
            'frequent_itemsets_found': rules_data.get('frequent_itemsets_count', 0),
            'association_rules_found': rules_data.get('rules_count', 0),
            'transaction_count': rules_data.get('transaction_count', 0),
            'unique_items': rules_data.get('unique_items', 0)
        }