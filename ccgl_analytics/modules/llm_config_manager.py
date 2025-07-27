"""
LLM配置管理模块

管理多个大模型的配置、查询意图解析、分析策略生成和报告生成
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
import logging
import re


class LLMConfigManager:
    """LLM配置管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化LLM配置管理器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 支持的模型
        self.models = {
            'openai': self.config.get('openai', {}),
            'anthropic': self.config.get('anthropic', {}),
            'zhipu': self.config.get('zhipu', {}),
            'dashscope': self.config.get('dashscope', {})
        }
        
        # 当前活跃模型
        self.active_model = 'openai'
        
        # 模拟的客户端（实际使用时需要安装对应的SDK）
        self.clients = {}
        self._initialize_clients()
        
        self.logger.info("LLM配置管理器初始化完成")
    
    def _initialize_clients(self):
        """初始化LLM客户端"""
        for model_name, model_config in self.models.items():
            if model_config.get('api_key'):
                self.clients[model_name] = MockLLMClient(model_name, model_config)
                self.logger.info(f"初始化 {model_name} 客户端")
    
    def set_active_model(self, model_name: str):
        """设置活跃模型"""
        if model_name in self.models:
            self.active_model = model_name
            self.logger.info(f"切换到模型: {model_name}")
        else:
            self.logger.warning(f"未知模型: {model_name}")
    
    async def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """解析查询意图"""
        try:
            # 使用规则匹配解析意图
            intent = self._rule_based_intent_parsing(query)
            
            # 如果有可用的LLM客户端，进行增强解析
            if self.active_model in self.clients:
                llm_intent = await self._llm_intent_parsing(query)
                intent.update(llm_intent)
            
            return intent
            
        except Exception as e:
            self.logger.error(f"查询意图解析失败: {e}")
            return {'error': str(e)}
    
    def _rule_based_intent_parsing(self, query: str) -> Dict[str, Any]:
        """基于规则的意图解析"""
        query_lower = query.lower()
        
        intent = {
            'original_query': query,
            'analysis_type': 'summary',
            'entities': [],
            'time_range': None,
            'filters': {},
            'confidence': 0.8
        }
        
        # 分析类型识别
        if any(keyword in query_lower for keyword in ['聚类', 'cluster', '分组', '分类']):
            intent['analysis_type'] = 'clustering'
        elif any(keyword in query_lower for keyword in ['异常', 'anomaly', '离群', '异常值']):
            intent['analysis_type'] = 'anomaly'
        elif any(keyword in query_lower for keyword in ['关联', 'association', '规则', '相关性']):
            intent['analysis_type'] = 'association'
        elif any(keyword in query_lower for keyword in ['关系', 'relationship', '联系']):
            intent['analysis_type'] = 'relationship'
        
        # 时间范围提取
        time_patterns = {
            r'最近(\d+)天': lambda m: {'days': int(m.group(1))},
            r'最近(\d+)周': lambda m: {'weeks': int(m.group(1))},
            r'最近(\d+)月': lambda m: {'months': int(m.group(1))},
            r'今天|当天': lambda m: {'days': 1},
            r'本周': lambda m: {'weeks': 1},
            r'本月': lambda m: {'months': 1}
        }
        
        for pattern, handler in time_patterns.items():
            match = re.search(pattern, query)
            if match:
                intent['time_range'] = handler(match)
                break
        
        # 实体提取（简单的关键词匹配）
        entities = []
        entity_keywords = {
            '库存': 'inventory',
            '商品': 'product',
            '销量': 'sales',
            '价格': 'price',
            '供应商': 'supplier',
            '分类': 'category'
        }
        
        for keyword, entity_type in entity_keywords.items():
            if keyword in query:
                entities.append({'text': keyword, 'type': entity_type})
        
        intent['entities'] = entities
        
        return intent
    
    async def _llm_intent_parsing(self, query: str) -> Dict[str, Any]:
        """使用LLM进行意图解析"""
        client = self.clients.get(self.active_model)
        if not client:
            return {}
        
        prompt = f"""
        请分析以下查询的意图，并以JSON格式返回结果：
        
        查询: {query}
        
        请识别：
        1. 分析类型 (summary, clustering, anomaly, association, relationship)
        2. 涉及的业务实体
        3. 时间范围
        4. 筛选条件
        
        返回JSON格式：
        {{
            "analysis_type": "...",
            "entities": [...],
            "time_range": {{...}},
            "filters": {{...}},
            "confidence": 0.9
        }}
        """
        
        try:
            response = await client.generate(prompt)
            return json.loads(response.get('content', '{}'))
        except Exception as e:
            self.logger.warning(f"LLM意图解析失败: {e}")
            return {}
    
    async def generate_analysis_strategy(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析策略"""
        try:
            # 基础策略
            strategy = {
                'analysis_type': intent.get('analysis_type', 'summary'),
                'data_sources': ['warehouse_inventory'],
                'filters': intent.get('filters', {}),
                'parameters': {},
                'output_format': 'detailed'
            }
            
            # 根据时间范围添加过滤条件
            if intent.get('time_range'):
                strategy['time_filter'] = intent['time_range']
            
            # 根据实体添加相关字段
            entities = intent.get('entities', [])
            relevant_fields = []
            for entity in entities:
                if entity.get('type') == 'product':
                    relevant_fields.extend(['product_name', 'category'])
                elif entity.get('type') == 'sales':
                    relevant_fields.extend(['quantity', 'price'])
                elif entity.get('type') == 'supplier':
                    relevant_fields.append('supplier')
            
            if relevant_fields:
                strategy['focus_fields'] = list(set(relevant_fields))
            
            # 根据分析类型设置参数
            if strategy['analysis_type'] == 'clustering':
                strategy['parameters'] = {
                    'algorithm': 'kmeans',
                    'max_clusters': 8
                }
            elif strategy['analysis_type'] == 'anomaly':
                strategy['parameters'] = {
                    'contamination': 0.1,
                    'methods': ['isolation_forest']
                }
            elif strategy['analysis_type'] == 'association':
                strategy['parameters'] = {
                    'min_support': 0.01,
                    'min_confidence': 0.5
                }
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"分析策略生成失败: {e}")
            return {'error': str(e)}
    
    async def generate_report(self, results: Dict[str, Any], 
                            original_query: str) -> str:
        """生成自然语言报告"""
        try:
            # 基础报告生成
            report = self._generate_basic_report(results, original_query)
            
            # 如果有可用的LLM客户端，生成增强报告
            if self.active_model in self.clients:
                enhanced_report = await self._generate_llm_report(results, original_query)
                if enhanced_report:
                    report = enhanced_report
            
            return report
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            return f"报告生成出现错误: {e}"
    
    def _generate_basic_report(self, results: Dict[str, Any], 
                             original_query: str) -> str:
        """生成基础报告"""
        if 'error' in results:
            return f"抱歉，处理您的查询时出现错误: {results['error']}"
        
        if 'summary' in results:
            summary = results['summary']
            report = f"根据您的查询 '{original_query}'，数据分析结果如下：\n\n"
            report += f"• 总记录数: {summary.get('total_records', 0)}\n"
            report += f"• 数据列数: {len(summary.get('columns', []))}\n"
            
            if 'missing_values' in summary:
                missing_total = sum(summary['missing_values'].values())
                if missing_total > 0:
                    report += f"• 发现缺失值: {missing_total} 个\n"
            
            return report
        
        if 'clustering' in results:
            clustering = results['clustering']
            n_clusters = clustering.get('n_clusters', 0)
            report = f"聚类分析结果：发现了 {n_clusters} 个不同的数据群组。"
            
            if 'silhouette_score' in clustering:
                score = clustering['silhouette_score']
                if score > 0.7:
                    quality = "优秀"
                elif score > 0.5:
                    quality = "良好"
                else:
                    quality = "一般"
                report += f" 聚类质量: {quality} (轮廓系数: {score:.3f})"
            
            return report
        
        if 'anomaly' in results:
            anomaly = results['anomaly']
            n_anomalies = anomaly.get('n_anomalies', 0)
            percentage = anomaly.get('anomaly_percentage', 0)
            report = f"异常检测结果：发现了 {n_anomalies} 个异常数据点，"
            report += f"占总数据的 {percentage:.2f}%。"
            
            if n_anomalies > 0:
                report += " 建议进一步检查这些异常数据的原因。"
            
            return report
        
        return "数据分析已完成，但未生成详细报告。"
    
    async def _generate_llm_report(self, results: Dict[str, Any], 
                                 original_query: str) -> Optional[str]:
        """使用LLM生成报告"""
        client = self.clients.get(self.active_model)
        if not client:
            return None
        
        # 准备结果摘要
        results_summary = self._prepare_results_summary(results)
        
        prompt = f"""
        基于以下数据分析结果，为用户查询生成一份详细的中文分析报告：
        
        用户查询: {original_query}
        
        分析结果:
        {results_summary}
        
        要求：
        1. 使用通俗易懂的语言
        2. 突出关键发现和洞察
        3. 提供可行的建议
        4. 结构清晰，条理分明
        5. 字数控制在300字以内
        """
        
        try:
            response = await client.generate(prompt)
            return response.get('content', '')
        except Exception as e:
            self.logger.warning(f"LLM报告生成失败: {e}")
            return None
    
    def _prepare_results_summary(self, results: Dict[str, Any]) -> str:
        """准备结果摘要"""
        summary_parts = []
        
        for key, value in results.items():
            if key == 'summary':
                summary_parts.append(f"数据概况: {value}")
            elif key == 'clustering':
                summary_parts.append(f"聚类分析: 发现{value.get('n_clusters', 0)}个群组")
            elif key == 'anomaly':
                summary_parts.append(f"异常检测: 发现{value.get('n_anomalies', 0)}个异常点")
            elif key == 'association':
                summary_parts.append(f"关联规则: 发现{value.get('n_rules', 0)}条规则")
        
        return "\n".join(summary_parts)
    
    async def analyze_data_patterns(self, data) -> Dict[str, Any]:
        """分析数据模式"""
        try:
            patterns = {
                'trends': self._identify_trends(data),
                'seasonality': self._identify_seasonality(data),
                'outliers': self._identify_outlier_patterns(data),
                'correlations': self._identify_correlations(data)
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"数据模式分析失败: {e}")
            return {'error': str(e)}
    
    def _identify_trends(self, data) -> List[str]:
        """识别趋势"""
        trends = []
        
        # 简单的趋势识别（实际应用中会更复杂）
        if hasattr(data, 'columns'):
            for col in data.select_dtypes(include=['number']).columns:
                if len(data) > 10:
                    correlation_with_index = data[col].corr(pd.Series(range(len(data))))
                    if abs(correlation_with_index) > 0.3:
                        trend_direction = "上升" if correlation_with_index > 0 else "下降"
                        trends.append(f"{col}显示{trend_direction}趋势")
        
        return trends
    
    def _identify_seasonality(self, data) -> List[str]:
        """识别季节性"""
        # 简化的季节性检测
        return ["数据中可能存在周期性模式"]
    
    def _identify_outlier_patterns(self, data) -> List[str]:
        """识别异常值模式"""
        return ["检测到一些数据异常，建议进一步分析"]
    
    def _identify_correlations(self, data) -> List[str]:
        """识别相关性"""
        correlations = []
        
        if hasattr(data, 'corr'):
            corr_matrix = data.select_dtypes(include=['number']).corr()
            # 找到强相关关系
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        direction = "正相关" if corr_value > 0 else "负相关"
                        correlations.append(f"{col1}与{col2}存在强{direction}")
        
        return correlations
    
    async def generate_recommendations(self, patterns: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """生成智能推荐"""
        try:
            recommendations = {
                'items': [],
                'confidence': 0.8,
                'reasoning': []
            }
            
            # 基于模式生成推荐
            if patterns.get('trends'):
                for trend in patterns['trends']:
                    if "上升" in trend:
                        recommendations['items'].append(f"关注{trend}，考虑增加相关资源投入")
                    elif "下降" in trend:
                        recommendations['items'].append(f"关注{trend}，需要分析下降原因并采取措施")
            
            if patterns.get('outliers'):
                recommendations['items'].append("建议定期检查异常数据，建立异常监控机制")
            
            if patterns.get('correlations'):
                recommendations['items'].append("利用发现的相关性优化业务流程")
            
            # 通用推荐
            if not recommendations['items']:
                recommendations['items'] = [
                    "定期进行数据质量检查",
                    "建立数据监控仪表板",
                    "实施预测性分析",
                    "优化数据收集流程"
                ]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"智能推荐生成失败: {e}")
            return {'error': str(e)}


class MockLLMClient:
    """模拟LLM客户端（用于演示，实际使用时替换为真实客户端）"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
    
    async def generate(self, prompt: str) -> Dict[str, Any]:
        """模拟生成响应"""
        # 这里返回模拟响应，实际使用时调用真实API
        return {
            'content': f"这是来自{self.model_name}的模拟响应",
            'model': self.model_name
        }