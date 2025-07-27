"""
关系提取模块

从数据中提取和分析实体间的关系
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class RelationshipExtractor:
    """关系提取类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化关系提取器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 关系类型配置
        self.relationship_types = [
            'belongs_to',      # 属于关系
            'supplies',        # 供应关系
            'competes_with',   # 竞争关系
            'correlates_with', # 相关关系
            'depends_on'       # 依赖关系
        ]
        
        # 图形分析
        self.graph = nx.Graph()
        
        self.logger.info("关系提取器初始化完成")
    
    async def extract_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """提取数据中的关系"""
        try:
            self.logger.info(f"开始提取关系，数据形状: {data.shape}")
            
            relationships = {}
            
            # 提取不同类型的关系
            relationships['category_product'] = await self._extract_category_relationships(data)
            relationships['supplier_product'] = await self._extract_supplier_relationships(data)
            relationships['product_correlations'] = await self._extract_product_correlations(data)
            relationships['temporal_patterns'] = await self._extract_temporal_relationships(data)
            
            # 构建关系图
            relationship_graph = await self._build_relationship_graph(relationships)
            
            # 分析关系网络
            network_analysis = await self._analyze_relationship_network(relationship_graph)
            
            result = {
                'relationships': relationships,
                'graph': relationship_graph,
                'network_analysis': network_analysis,
                'total_relationships': sum(len(rels) for rels in relationships.values() if isinstance(rels, list))
            }
            
            self.logger.info(f"关系提取完成，发现 {result['total_relationships']} 个关系")
            return result
            
        except Exception as e:
            self.logger.error(f"关系提取失败: {e}")
            raise
    
    async def _extract_category_relationships(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """提取分类-商品关系"""
        relationships = []
        
        if 'category' in data.columns and 'product_name' in data.columns:
            category_groups = data.groupby('category')
            
            for category, group in category_groups:
                products = group['product_name'].unique()
                
                for product in products:
                    relationships.append({
                        'type': 'belongs_to',
                        'source': product,
                        'target': category,
                        'weight': 1.0,
                        'properties': {
                            'product_count': len(group[group['product_name'] == product])
                        }
                    })
        
        return relationships
    
    async def _extract_supplier_relationships(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """提取供应商-商品关系"""
        relationships = []
        
        if 'supplier' in data.columns and 'product_name' in data.columns:
            supplier_groups = data.groupby('supplier')
            
            for supplier, group in supplier_groups:
                products = group['product_name'].unique()
                
                for product in products:
                    product_data = group[group['product_name'] == product]
                    avg_price = product_data['price'].mean() if 'price' in data.columns else 0
                    total_quantity = product_data['quantity'].sum() if 'quantity' in data.columns else 0
                    
                    relationships.append({
                        'type': 'supplies',
                        'source': supplier,
                        'target': product,
                        'weight': total_quantity / 100.0,  # 归一化权重
                        'properties': {
                            'avg_price': avg_price,
                            'total_quantity': total_quantity,
                            'supply_frequency': len(product_data)
                        }
                    })
        
        return relationships
    
    async def _extract_product_correlations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """提取商品间的相关关系"""
        relationships = []
        
        try:
            # 选择数值列进行相关性分析
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return relationships
            
            # 按商品分组计算统计信息
            if 'product_name' in data.columns:
                product_stats = data.groupby('product_name')[numeric_cols].agg(['mean', 'sum', 'count'])
                
                # 计算商品间的相似度
                similarity_matrix = self._calculate_product_similarity(product_stats)
                
                # 提取高相似度的商品对
                threshold = 0.7
                products = product_stats.index.tolist()
                
                for i in range(len(products)):
                    for j in range(i + 1, len(products)):
                        similarity = similarity_matrix[i][j]
                        
                        if similarity > threshold:
                            relationships.append({
                                'type': 'correlates_with',
                                'source': products[i],
                                'target': products[j],
                                'weight': similarity,
                                'properties': {
                                    'similarity_score': similarity,
                                    'relationship_strength': 'strong' if similarity > 0.8 else 'moderate'
                                }
                            })
        
        except Exception as e:
            self.logger.warning(f"商品相关性提取失败: {e}")
        
        return relationships
    
    def _calculate_product_similarity(self, product_stats: pd.DataFrame) -> np.ndarray:
        """计算商品相似度矩阵"""
        try:
            # 展平多级列索引
            feature_matrix = product_stats.values.reshape(len(product_stats), -1)
            
            # 标准化特征
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(normalized_features)
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.warning(f"相似度计算失败: {e}")
            return np.eye(len(product_stats))
    
    async def _extract_temporal_relationships(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """提取时间模式关系"""
        relationships = []
        
        try:
            # 查找时间列
            time_cols = []
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    time_cols.append(col)
            
            if not time_cols:
                return relationships
            
            time_col = time_cols[0]
            
            # 确保时间列是datetime类型
            if data[time_col].dtype == 'object':
                data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
            
            # 按时间段分析模式
            data['time_period'] = data[time_col].dt.to_period('D')
            
            if 'product_name' in data.columns:
                # 分析商品的时间依赖关系
                time_patterns = data.groupby(['time_period', 'product_name']).size().unstack(fill_value=0)
                
                # 计算时间序列相关性
                correlation_matrix = time_patterns.corr()
                
                # 提取强时间相关的商品对
                threshold = 0.6
                products = correlation_matrix.columns.tolist()
                
                for i in range(len(products)):
                    for j in range(i + 1, len(products)):
                        correlation = correlation_matrix.iloc[i, j]
                        
                        if abs(correlation) > threshold:
                            relationships.append({
                                'type': 'temporal_correlation',
                                'source': products[i],
                                'target': products[j],
                                'weight': abs(correlation),
                                'properties': {
                                    'correlation': correlation,
                                    'pattern_type': 'positive' if correlation > 0 else 'negative'
                                }
                            })
        
        except Exception as e:
            self.logger.warning(f"时间关系提取失败: {e}")
        
        return relationships
    
    async def _build_relationship_graph(self, relationships: Dict[str, List]) -> Dict[str, Any]:
        """构建关系图"""
        try:
            # 清空现有图
            self.graph.clear()
            
            # 添加所有关系到图中
            all_relationships = []
            for rel_type, rels in relationships.items():
                if isinstance(rels, list):
                    all_relationships.extend(rels)
            
            # 构建节点和边
            nodes = set()
            edges = []
            
            for rel in all_relationships:
                source = rel['source']
                target = rel['target']
                weight = rel.get('weight', 1.0)
                
                nodes.add(source)
                nodes.add(target)
                
                edges.append({
                    'source': source,
                    'target': target,
                    'weight': weight,
                    'type': rel['type'],
                    'properties': rel.get('properties', {})
                })
                
                # 添加到NetworkX图
                self.graph.add_edge(source, target, weight=weight, type=rel['type'])
            
            graph_data = {
                'nodes': [{'id': node, 'label': str(node)} for node in nodes],
                'edges': edges,
                'statistics': {
                    'num_nodes': len(nodes),
                    'num_edges': len(edges),
                    'density': nx.density(self.graph) if len(nodes) > 1 else 0
                }
            }
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"关系图构建失败: {e}")
            return {'nodes': [], 'edges': [], 'statistics': {}}
    
    async def _analyze_relationship_network(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析关系网络"""
        try:
            if len(self.graph.nodes()) == 0:
                return {'error': '图为空'}
            
            analysis = {}
            
            # 基本网络统计
            analysis['basic_stats'] = {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_connected(self.graph)
            }
            
            # 中心性分析
            if len(self.graph.nodes()) > 1:
                try:
                    degree_centrality = nx.degree_centrality(self.graph)
                    betweenness_centrality = nx.betweenness_centrality(self.graph)
                    closeness_centrality = nx.closeness_centrality(self.graph)
                    
                    # 找到最重要的节点
                    most_central_nodes = {
                        'degree': max(degree_centrality.items(), key=lambda x: x[1]),
                        'betweenness': max(betweenness_centrality.items(), key=lambda x: x[1]),
                        'closeness': max(closeness_centrality.items(), key=lambda x: x[1])
                    }
                    
                    analysis['centrality'] = {
                        'most_central_nodes': most_central_nodes,
                        'avg_degree_centrality': np.mean(list(degree_centrality.values())),
                        'avg_betweenness_centrality': np.mean(list(betweenness_centrality.values())),
                        'avg_closeness_centrality': np.mean(list(closeness_centrality.values()))
                    }
                    
                except Exception as e:
                    self.logger.warning(f"中心性分析失败: {e}")
                    analysis['centrality'] = {'error': str(e)}
            
            # 社区检测
            try:
                communities = nx.community.greedy_modularity_communities(self.graph)
                analysis['communities'] = {
                    'num_communities': len(communities),
                    'modularity': nx.community.modularity(self.graph, communities),
                    'communities': [list(community) for community in communities]
                }
            except Exception as e:
                self.logger.warning(f"社区检测失败: {e}")
                analysis['communities'] = {'error': str(e)}
            
            # 路径分析
            try:
                if nx.is_connected(self.graph):
                    avg_path_length = nx.average_shortest_path_length(self.graph)
                    diameter = nx.diameter(self.graph)
                    
                    analysis['path_analysis'] = {
                        'average_path_length': avg_path_length,
                        'diameter': diameter,
                        'radius': nx.radius(self.graph)
                    }
                else:
                    # 对于非连通图，分析最大连通分量
                    largest_cc = max(nx.connected_components(self.graph), key=len)
                    subgraph = self.graph.subgraph(largest_cc)
                    
                    analysis['path_analysis'] = {
                        'is_connected': False,
                        'largest_component_size': len(largest_cc),
                        'num_components': nx.number_connected_components(self.graph)
                    }
                    
            except Exception as e:
                self.logger.warning(f"路径分析失败: {e}")
                analysis['path_analysis'] = {'error': str(e)}
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"网络分析失败: {e}")
            return {'error': str(e)}
    
    async def find_relationship_paths(self, source_entity: str, 
                                    target_entity: str,
                                    max_length: int = 3) -> List[List[str]]:
        """查找两个实体间的关系路径"""
        try:
            if source_entity not in self.graph.nodes() or target_entity not in self.graph.nodes():
                return []
            
            # 查找所有简单路径
            paths = list(nx.all_simple_paths(
                self.graph, 
                source_entity, 
                target_entity, 
                cutoff=max_length
            ))
            
            # 按路径长度排序
            paths.sort(key=len)
            
            return paths[:10]  # 返回前10条最短路径
            
        except Exception as e:
            self.logger.error(f"路径查找失败: {e}")
            return []
    
    async def get_entity_relationships(self, entity: str) -> Dict[str, Any]:
        """获取指定实体的所有关系"""
        try:
            if entity not in self.graph.nodes():
                return {'error': f'实体 {entity} 不存在'}
            
            # 获取邻居节点
            neighbors = list(self.graph.neighbors(entity))
            
            # 获取相关边的详细信息
            relationships = []
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(entity, neighbor)
                relationships.append({
                    'target': neighbor,
                    'type': edge_data.get('type', 'unknown'),
                    'weight': edge_data.get('weight', 1.0)
                })
            
            # 计算实体的重要性指标
            degree = self.graph.degree(entity)
            
            result = {
                'entity': entity,
                'degree': degree,
                'num_relationships': len(relationships),
                'relationships': relationships,
                'neighbors': neighbors
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"实体关系获取失败: {e}")
            return {'error': str(e)}
    
    def export_graph(self, format: str = 'gexf') -> str:
        """导出关系图"""
        try:
            if format == 'gexf':
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.gexf', delete=False) as f:
                    nx.write_gexf(self.graph, f.name)
                    return f.name
            elif format == 'graphml':
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
                    nx.write_graphml(self.graph, f.name)
                    return f.name
            else:
                return ''
                
        except Exception as e:
            self.logger.error(f"图导出失败: {e}")
            return ''