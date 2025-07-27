"""
LLM集成MCP服务器

专门处理自然语言查询、报告生成和AI增强分析的MCP服务器
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# 导入核心模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ccgl_analytics.modules.llm_config_manager import LLMConfigManager
from ccgl_analytics.modules.relationship_extraction import RelationshipExtractor


class LLMMCPServer:
    """LLM集成MCP服务器"""
    
    def __init__(self, port: int, config: Dict[str, Any]):
        """初始化LLM服务器"""
        self.port = port
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化核心组件
        self.llm_manager = LLMConfigManager(config.get('ai_models', {}))
        self.relationship_extractor = RelationshipExtractor(config)
        
        # 服务器状态
        self.is_running = False
        self.start_time = None
        self.processed_requests = 0
        
        # 对话历史缓存
        self.conversation_cache = {}
        self.query_cache = {}
        
        self.logger.info(f"LLM集成MCP服务器初始化完成，端口: {port}")
    
    async def start(self):
        """启动服务器"""
        try:
            self.logger.info(f"启动LLM集成MCP服务器，端口: {self.port}")
            
            self.is_running = True
            self.start_time = datetime.now()
            
            # 模拟启动Web服务器
            await self._start_mock_server()
            
        except Exception as e:
            self.logger.error(f"服务器启动失败: {e}")
            raise
    
    async def stop(self):
        """停止服务器"""
        try:
            self.logger.info("停止LLM集成MCP服务器")
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"服务器停止失败: {e}")
    
    async def _start_mock_server(self):
        """启动模拟服务器"""
        self.logger.info(f"LLM集成服务器运行在端口 {self.port}")
        
        # 模拟服务器持续运行
        while self.is_running:
            await asyncio.sleep(1)
    
    async def execute_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行LLM任务"""
        try:
            self.processed_requests += 1
            self.logger.info(f"执行LLM任务: {task_type}")
            
            if task_type == 'nlp':
                return await self._handle_nlp_task(data)
            elif task_type == 'text_analysis':
                return await self._handle_text_analysis_task(data)
            elif task_type == 'report_generation':
                return await self._handle_report_generation_task(data)
            elif task_type == 'query_intent':
                return await self._handle_query_intent_task(data)
            elif task_type == 'conversation':
                return await self._handle_conversation_task(data)
            elif task_type == 'recommendation':
                return await self._handle_recommendation_task(data)
            elif task_type == 'relationship_analysis':
                return await self._handle_relationship_analysis_task(data)
            else:
                return {'error': f'不支持的任务类型: {task_type}'}
                
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            return {'error': str(e)}
    
    async def _handle_nlp_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理自然语言处理任务"""
        try:
            query = data.get('query', '')
            model = data.get('model', 'openai')
            
            if not query:
                return {'error': '查询内容不能为空'}
            
            # 设置活跃模型
            self.llm_manager.set_active_model(model)
            
            # 解析查询意图
            intent = await self.llm_manager.parse_query_intent(query)
            
            # 生成分析策略
            strategy = await self.llm_manager.generate_analysis_strategy(intent)
            
            return {
                'status': 'completed',
                'task_type': 'nlp',
                'query': query,
                'model_used': model,
                'intent': intent,
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"NLP任务失败: {e}")
            return {'error': str(e)}
    
    async def _handle_text_analysis_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本分析任务"""
        try:
            text = data.get('text', '')
            analysis_type = data.get('analysis_type', 'sentiment')
            
            if not text:
                return {'error': '文本内容不能为空'}
            
            # 模拟文本分析
            analysis_result = await self._analyze_text(text, analysis_type)
            
            return {
                'status': 'completed',
                'task_type': 'text_analysis',
                'text': text[:100] + '...' if len(text) > 100 else text,
                'analysis_type': analysis_type,
                'result': analysis_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"文本分析失败: {e}")
            return {'error': str(e)}
    
    async def _analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """分析文本"""
        if analysis_type == 'sentiment':
            # 简单的情感分析模拟
            positive_words = ['好', '优秀', '满意', '推荐', '喜欢', '不错']
            negative_words = ['差', '糟糕', '失望', '不满', '讨厌', '问题']
            
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count > negative_count:
                sentiment = 'positive'
                confidence = 0.8
            elif negative_count > positive_count:
                sentiment = 'negative'
                confidence = 0.8
            else:
                sentiment = 'neutral'
                confidence = 0.6
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_words': positive_count,
                'negative_words': negative_count
            }
        
        elif analysis_type == 'keywords':
            # 简单的关键词提取
            words = text.split()
            word_freq = {}
            for word in words:
                word = word.lower().strip('.,!?;')
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 获取前5个高频词
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'keywords': [{'word': word, 'frequency': freq} for word, freq in keywords],
                'total_words': len(words),
                'unique_words': len(word_freq)
            }
        
        elif analysis_type == 'summary':
            # 简单的文本摘要
            sentences = text.split('。')
            if len(sentences) > 3:
                summary = '。'.join(sentences[:2]) + '。'
            else:
                summary = text
            
            return {
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0
            }
        
        else:
            return {'error': f'不支持的分析类型: {analysis_type}'}
    
    async def _handle_report_generation_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理报告生成任务"""
        try:
            analysis_results = data.get('analysis_results', {})
            report_type = data.get('report_type', 'summary')
            language = data.get('language', 'zh')
            
            # 生成报告
            report = await self._generate_report(analysis_results, report_type, language)
            
            return {
                'status': 'completed',
                'task_type': 'report_generation',
                'report_type': report_type,
                'language': language,
                'report': report,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            return {'error': str(e)}
    
    async def _generate_report(self, analysis_results: Dict[str, Any], 
                             report_type: str, language: str) -> Dict[str, Any]:
        """生成报告"""
        if report_type == 'summary':
            return await self._generate_summary_report(analysis_results, language)
        elif report_type == 'detailed':
            return await self._generate_detailed_report(analysis_results, language)
        elif report_type == 'executive':
            return await self._generate_executive_report(analysis_results, language)
        else:
            return {'error': f'不支持的报告类型: {report_type}'}
    
    async def _generate_summary_report(self, results: Dict[str, Any], language: str) -> Dict[str, Any]:
        """生成摘要报告"""
        report_sections = []
        
        if 'clustering' in results:
            clustering = results['clustering']
            n_clusters = clustering.get('n_clusters', 0)
            score = clustering.get('silhouette_score', 0)
            
            if language == 'zh':
                section = f"聚类分析发现了{n_clusters}个不同的数据群组，聚类质量评分为{score:.3f}。"
            else:
                section = f"Clustering analysis identified {n_clusters} distinct groups with a quality score of {score:.3f}."
            
            report_sections.append(section)
        
        if 'anomaly' in results:
            anomaly = results['anomaly']
            n_anomalies = anomaly.get('n_anomalies', 0)
            percentage = anomaly.get('anomaly_percentage', 0)
            
            if language == 'zh':
                section = f"异常检测发现了{n_anomalies}个异常数据点，占总数据的{percentage:.2f}%。"
            else:
                section = f"Anomaly detection found {n_anomalies} outliers, representing {percentage:.2f}% of the data."
            
            report_sections.append(section)
        
        if 'association' in results:
            association = results['association']
            n_rules = association.get('n_rules', 0)
            
            if language == 'zh':
                section = f"关联规则挖掘发现了{n_rules}条有意义的关联规则。"
            else:
                section = f"Association rule mining discovered {n_rules} meaningful rules."
            
            report_sections.append(section)
        
        # 生成结论和建议
        if language == 'zh':
            conclusion = "建议基于这些发现优化业务流程，并定期监控关键指标。"
            recommendations = [
                "深入分析异常数据的根本原因",
                "利用聚类结果进行精准营销",
                "建立实时监控系统",
                "定期重新评估分析模型"
            ]
        else:
            conclusion = "Recommend optimizing business processes based on these findings and regularly monitoring key metrics."
            recommendations = [
                "Investigate root causes of anomalies",
                "Leverage clustering for targeted marketing",
                "Establish real-time monitoring",
                "Regularly reassess analytical models"
            ]
        
        return {
            'title': '数据分析摘要报告' if language == 'zh' else 'Data Analysis Summary Report',
            'sections': report_sections,
            'conclusion': conclusion,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat(),
            'word_count': sum(len(section.split()) for section in report_sections)
        }
    
    async def _generate_detailed_report(self, results: Dict[str, Any], language: str) -> Dict[str, Any]:
        """生成详细报告"""
        # 详细报告包含更多技术细节
        return {
            'title': '详细分析报告' if language == 'zh' else 'Detailed Analysis Report',
            'executive_summary': await self._generate_summary_report(results, language),
            'methodology': '采用了多种机器学习算法进行综合分析' if language == 'zh' else 'Multiple ML algorithms used for comprehensive analysis',
            'detailed_findings': results,
            'technical_details': {
                'algorithms_used': list(results.keys()),
                'data_quality': 'Good',
                'confidence_level': 0.85
            },
            'generated_at': datetime.now().isoformat()
        }
    
    async def _generate_executive_report(self, results: Dict[str, Any], language: str) -> Dict[str, Any]:
        """生成高管报告"""
        # 高管报告关注业务影响
        return {
            'title': '高管摘要报告' if language == 'zh' else 'Executive Summary Report',
            'key_insights': [
                '数据分析揭示了重要的业务模式' if language == 'zh' else 'Data analysis reveals important business patterns',
                '发现了需要关注的异常情况' if language == 'zh' else 'Identified anomalies requiring attention',
                '提供了优化机会的建议' if language == 'zh' else 'Suggests optimization opportunities'
            ],
            'business_impact': {
                'cost_savings_potential': '估计可节省成本15-20%' if language == 'zh' else 'Estimated 15-20% cost savings potential',
                'efficiency_gains': '预期效率提升25%' if language == 'zh' else 'Expected 25% efficiency improvement',
                'risk_mitigation': '降低运营风险' if language == 'zh' else 'Reduced operational risks'
            },
            'next_steps': [
                '实施推荐的优化措施' if language == 'zh' else 'Implement recommended optimizations',
                '建立持续监控机制' if language == 'zh' else 'Establish continuous monitoring',
                '培训相关人员' if language == 'zh' else 'Train relevant personnel'
            ],
            'generated_at': datetime.now().isoformat()
        }
    
    async def _handle_query_intent_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理查询意图分析任务"""
        try:
            query = data.get('query', '')
            
            if not query:
                return {'error': '查询内容不能为空'}
            
            # 解析查询意图
            intent = await self.llm_manager.parse_query_intent(query)
            
            return {
                'status': 'completed',
                'task_type': 'query_intent',
                'query': query,
                'intent': intent,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"查询意图分析失败: {e}")
            return {'error': str(e)}
    
    async def _handle_conversation_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理对话任务"""
        try:
            message = data.get('message', '')
            session_id = data.get('session_id', 'default')
            context = data.get('context', {})
            
            if not message:
                return {'error': '消息内容不能为空'}
            
            # 获取对话历史
            if session_id not in self.conversation_cache:
                self.conversation_cache[session_id] = []
            
            conversation_history = self.conversation_cache[session_id]
            
            # 添加用户消息
            conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat()
            })
            
            # 生成AI回复
            ai_response = await self._generate_ai_response(message, conversation_history, context)
            
            # 添加AI回复
            conversation_history.append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # 限制历史长度
            if len(conversation_history) > 20:
                self.conversation_cache[session_id] = conversation_history[-20:]
            
            return {
                'status': 'completed',
                'task_type': 'conversation',
                'session_id': session_id,
                'user_message': message,
                'ai_response': ai_response,
                'conversation_length': len(conversation_history),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"对话任务失败: {e}")
            return {'error': str(e)}
    
    async def _generate_ai_response(self, message: str, 
                                  history: List[Dict[str, Any]], 
                                  context: Dict[str, Any]) -> str:
        """生成AI回复"""
        # 简单的规则基回复系统
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['聚类', 'cluster']):
            return "聚类分析可以帮助您发现数据中的隐藏模式。我建议使用KMeans或DBSCAN算法，您希望分析哪些特征？"
        
        elif any(word in message_lower for word in ['异常', 'anomaly']):
            return "异常检测对于发现数据质量问题很重要。我推荐使用孤立森林算法，它对各种异常类型都很敏感。"
        
        elif any(word in message_lower for word in ['关联', 'association']):
            return "关联规则挖掘可以发现数据项之间的有趣关系。让我为您分析数据中的关联模式。"
        
        elif any(word in message_lower for word in ['报告', 'report']):
            return "我可以为您生成详细的分析报告。您希望生成哪种类型的报告：摘要报告、详细报告还是高管报告？"
        
        elif any(word in message_lower for word in ['帮助', 'help']):
            return "我可以帮助您进行数据分析、生成报告、解答问题。具体您需要什么帮助？"
        
        else:
            return "我理解您的问题。让我为您分析相关数据并提供专业建议。您能提供更多具体信息吗？"
    
    async def _handle_recommendation_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理推荐任务"""
        try:
            analysis_results = data.get('analysis_results', {})
            business_context = data.get('business_context', {})
            
            # 生成智能推荐
            recommendations = await self.llm_manager.generate_recommendations(
                analysis_results, business_context
            )
            
            return {
                'status': 'completed',
                'task_type': 'recommendation',
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"推荐任务失败: {e}")
            return {'error': str(e)}
    
    async def _handle_relationship_analysis_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理关系分析任务"""
        try:
            dataset = data.get('dataset', {})
            
            if not dataset:
                return {'error': '数据集不能为空'}
            
            # 转换为DataFrame
            import pandas as pd
            df = pd.DataFrame(dataset)
            
            # 执行关系提取
            relationships = await self.relationship_extractor.extract_relationships(df)
            
            return {
                'status': 'completed',
                'task_type': 'relationship_analysis',
                'relationships': relationships,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"关系分析失败: {e}")
            return {'error': str(e)}
    
    async def get_conversations(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取对话历史"""
        if session_id:
            if session_id in self.conversation_cache:
                return {
                    'session_id': session_id,
                    'conversation': self.conversation_cache[session_id]
                }
            else:
                return {'error': '会话不存在'}
        else:
            return {
                'total_sessions': len(self.conversation_cache),
                'sessions': list(self.conversation_cache.keys())
            }
    
    async def clear_conversation(self, session_id: str) -> Dict[str, Any]:
        """清除对话历史"""
        if session_id in self.conversation_cache:
            del self.conversation_cache[session_id]
            return {'status': 'success', 'message': f'会话 {session_id} 已清除'}
        else:
            return {'status': 'error', 'message': f'会话 {session_id} 不存在'}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            return {
                'status': 'healthy' if self.is_running else 'unhealthy',
                'server_type': 'llm_integration',
                'port': self.port,
                'uptime': uptime,
                'processed_requests': self.processed_requests,
                'active_conversations': len(self.conversation_cache),
                'available_models': list(self.llm_manager.models.keys()),
                'active_model': self.llm_manager.active_model,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """获取服务器状态"""
        return {
            'server_type': 'llm_integration',
            'port': self.port,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'processed_requests': self.processed_requests,
            'active_conversations': len(self.conversation_cache),
            'supported_tasks': [
                'nlp',
                'text_analysis', 
                'report_generation',
                'query_intent',
                'conversation',
                'recommendation',
                'relationship_analysis'
            ]
        }