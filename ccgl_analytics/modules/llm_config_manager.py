"""
大语言模型配置管理模块

支持多种大语言模型的配置管理和自然语言查询接口。
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

# Optional imports for different AI providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class LLMConfigManager:
    """大语言模型配置管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化LLM配置管理器
        
        Args:
            config: LLM配置信息
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.clients = {}
        self.available_models = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """初始化各种LLM客户端"""
        # OpenAI GPT客户端
        if OPENAI_AVAILABLE:
            openai_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            if openai_key:
                try:
                    self.clients['openai'] = openai.OpenAI(api_key=openai_key)
                    self.available_models['openai'] = [
                        'gpt-4o', 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'
                    ]
                    self.logger.info("OpenAI客户端初始化成功")
                except Exception as e:
                    self.logger.warning(f"OpenAI客户端初始化失败: {e}")
            else:
                self.logger.warning("未配置OpenAI API密钥")
        
        # Anthropic Claude客户端
        if ANTHROPIC_AVAILABLE:
            anthropic_key = self.config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key:
                try:
                    self.clients['anthropic'] = anthropic.Anthropic(api_key=anthropic_key)
                    self.available_models['anthropic'] = [
                        'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'
                    ]
                    self.logger.info("Anthropic客户端初始化成功")
                except Exception as e:
                    self.logger.warning(f"Anthropic客户端初始化失败: {e}")
            else:
                self.logger.warning("未配置Anthropic API密钥")
        
        # 通义千问和智谱GLM等其他模型的配置
        self._setup_other_models()
        
        if not self.clients:
            self.logger.warning("没有可用的LLM客户端，将使用模拟模式")
    
    def _setup_other_models(self):
        """设置其他模型配置"""
        # 智谱GLM配置
        zhipu_key = self.config.get('zhipu_api_key') or os.getenv('ZHIPU_API_KEY')
        if zhipu_key:
            self.available_models['zhipu'] = ['glm-4', 'glm-3-turbo']
            self.logger.info("智谱GLM配置已设置")
        
        # 通义千问配置
        qwen_key = self.config.get('qwen_api_key') or os.getenv('QWEN_API_KEY')
        if qwen_key:
            self.available_models['qwen'] = ['qwen-turbo', 'qwen-max', 'qwen-max-1201']
            self.logger.info("通义千问配置已设置")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用的模型列表"""
        return self.available_models.copy()
    
    def natural_language_query(self, query: str, data_context: str = "", 
                             model: str = None, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        自然语言数据查询接口
        
        Args:
            query: 用户查询
            data_context: 数据上下文信息
            model: 指定使用的模型
            max_tokens: 最大token数
            
        Returns:
            查询结果
        """
        try:
            # 选择模型
            if model is None:
                model = self._select_best_model()
            
            # 构建提示词
            prompt = self._build_analysis_prompt(query, data_context)
            
            # 执行查询
            if model.startswith('gpt') and 'openai' in self.clients:
                response = self._query_openai(prompt, model, max_tokens)
            elif model.startswith('claude') and 'anthropic' in self.clients:
                response = self._query_anthropic(prompt, model, max_tokens)
            else:
                # 模拟响应
                response = self._simulate_response(query, data_context)
            
            return {
                'success': True,
                'query': query,
                'model_used': model,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"自然语言查询失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    def _select_best_model(self) -> str:
        """选择最佳可用模型"""
        # 优先级排序
        priority_models = [
            'gpt-4o', 'gpt-4', 'claude-3-opus-20240229', 
            'claude-3-sonnet-20240229', 'gpt-3.5-turbo'
        ]
        
        for model in priority_models:
            for provider, models in self.available_models.items():
                if model in models and provider in self.clients:
                    return model
        
        # 默认返回第一个可用模型
        for provider, models in self.available_models.items():
            if provider in self.clients and models:
                return models[0]
        
        return 'simulated'
    
    def _build_analysis_prompt(self, query: str, data_context: str) -> str:
        """构建分析提示词"""
        prompt = f"""你是一个专业的仓储管理数据分析专家。请基于以下数据上下文回答用户的问题：

数据上下文：
{data_context}

用户问题：
{query}

请提供：
1. 对问题的理解和分析
2. 基于数据的见解和发现
3. 具体的数据支持
4. 实用的建议和结论

请用中文回答，保持专业性和准确性。"""
        
        return prompt
    
    def _query_openai(self, prompt: str, model: str, max_tokens: int) -> str:
        """使用OpenAI API查询"""
        try:
            client = self.clients['openai']
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI查询失败: {e}")
    
    def _query_anthropic(self, prompt: str, model: str, max_tokens: int) -> str:
        """使用Anthropic API查询"""
        try:
            client = self.clients['anthropic']
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic查询失败: {e}")
    
    def _simulate_response(self, query: str, data_context: str) -> str:
        """模拟AI响应（当没有可用的API时）"""
        return f"""基于您的查询"{query}"，我提供以下分析：

🔍 查询理解：
您询问了关于仓储管理数据的问题。基于当前的数据上下文，我可以看到这是一个涉及仓库运营的复杂数据集。

📊 数据分析见解：
1. 数据完整性：当前数据集显示出良好的数据质量
2. 业务模式：可以识别出明显的运营模式和趋势
3. 异常情况：检测到一些需要关注的异常数据点

💡 建议：
1. 建议进一步关注数据质量管理
2. 可以考虑实施预测性分析
3. 优化库存管理策略

⚠️ 注意：这是模拟响应，建议配置真实的AI模型API以获得更准确的分析。

如需更详细的分析，请配置OpenAI或Anthropic API密钥。"""
    
    def generate_business_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成智能业务洞察
        
        Args:
            analysis_results: 分析结果数据
            
        Returns:
            业务洞察报告
        """
        try:
            # 提取关键指标
            key_metrics = self._extract_key_metrics(analysis_results)
            
            # 构建洞察提示
            insights_prompt = self._build_insights_prompt(key_metrics)
            
            # 使用最佳模型生成洞察
            model = self._select_best_model()
            
            if model != 'simulated':
                insights = self.natural_language_query(
                    "请基于这些数据生成业务洞察和建议", 
                    insights_prompt
                )
            else:
                insights = self._generate_simulated_insights(key_metrics)
            
            return {
                'success': True,
                'insights': insights,
                'key_metrics': key_metrics,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"生成业务洞察失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation_time': datetime.now().isoformat()
            }
    
    def _extract_key_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """提取关键指标"""
        metrics = {}
        
        # 数据质量指标
        if 'data_quality_assessment' in analysis_results:
            quality = analysis_results['data_quality_assessment']
            metrics['data_quality'] = {
                'completeness': quality.get('data_quality_metrics', {}).get('completeness'),
                'total_rows': quality.get('basic_stats', {}).get('total_rows'),
                'missing_values': quality.get('missing_values', {}).get('total_missing')
            }
        
        # 聚类分析结果
        if 'comprehensive_analysis' in analysis_results:
            comp_analysis = analysis_results['comprehensive_analysis']
            if 'clustering_analysis' in comp_analysis:
                clustering = comp_analysis['clustering_analysis']
                metrics['clustering'] = {
                    'optimal_clusters': clustering.get('optimal_clusters'),
                    'evaluation_metrics': clustering.get('evaluation_metrics')
                }
            
            # 异常检测结果
            if 'anomaly_detection' in comp_analysis:
                anomaly = comp_analysis['anomaly_detection']
                metrics['anomalies'] = anomaly.get('summary', {})
        
        return metrics
    
    def _build_insights_prompt(self, key_metrics: Dict[str, Any]) -> str:
        """构建洞察生成提示"""
        return f"""基于以下仓储管理数据分析的关键指标：

{json.dumps(key_metrics, ensure_ascii=False, indent=2)}

请生成专业的业务洞察，包括：
1. 数据质量评估
2. 运营效率分析  
3. 风险识别
4. 优化建议
5. 趋势预测"""
    
    def _generate_simulated_insights(self, key_metrics: Dict[str, Any]) -> Dict[str, str]:
        """生成模拟的业务洞察"""
        return {
            'response': f"""📈 CCGL仓储管理业务洞察报告

🎯 数据质量评估：
• 数据完整性达到 {key_metrics.get('data_quality', {}).get('completeness', 0.95)*100:.1f}%，表现良好
• 总计 {key_metrics.get('data_quality', {}).get('total_rows', 0)} 条记录，数据规模适中
• 缺失值 {key_metrics.get('data_quality', {}).get('missing_values', 0)} 个，需要关注

🔍 聚类分析洞察：
• 发现 {key_metrics.get('clustering', {}).get('optimal_clusters', 3)} 个主要客户/产品群体
• 建议针对不同群体制定差异化策略
• 可优化库存配置和服务策略

⚠️ 风险识别：
• 检测到 {key_metrics.get('anomalies', {}).get('consensus_anomalies', 0)} 个异常数据点
• 建议建立实时监控机制
• 需要加强数据质量控制

💡 优化建议：
1. 实施动态库存管理
2. 建立预测性维护
3. 优化供应链协调
4. 加强数据治理

⚠️ 注意：这是基于模拟AI的分析，建议配置真实AI模型以获得更深入的洞察。"""
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        # 这里可以实现对话历史的存储和检索
        return []
    
    def clear_conversation_history(self):
        """清除对话历史"""
        pass
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'available_providers': list(self.clients.keys()),
            'available_models': self.available_models,
            'default_model': self._select_best_model(),
            'openai_available': OPENAI_AVAILABLE and 'openai' in self.clients,
            'anthropic_available': ANTHROPIC_AVAILABLE and 'anthropic' in self.clients
        }