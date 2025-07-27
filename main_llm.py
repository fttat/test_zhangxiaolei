#!/usr/bin/env python3
"""
CCGL 仓储管理系统 - AI增强主程序

集成大模型的智能分析和对话式交互系统
"""

import argparse
import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 确保模块可以被导入
sys.path.insert(0, str(Path(__file__).parent))

from ccgl_analytics.modules.llm_config_manager import LLMConfigManager
from ccgl_analytics.modules.data_connection import DataConnection
from ccgl_analytics.modules.data_preprocessing import DataPreprocessor
from ccgl_analytics.modules.analysis_core import AnalysisCore
from ccgl_analytics.modules.relationship_extraction import RelationshipExtractor
from ccgl_analytics.utils.logger import setup_logger


class CCGLLLMSystem:
    """CCGL AI增强系统主类"""
    
    def __init__(self, config_path: str):
        """初始化LLM系统"""
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config.get('logging', {}))
        
        # 初始化LLM配置管理器
        self.llm_manager = LLMConfigManager(self.config.get('ai_models', {}))
        
        # 初始化核心分析组件
        self.data_connection = DataConnection(self.config['database'])
        self.data_preprocessor = DataPreprocessor(self.config['data_processing'])
        self.analysis_core = AnalysisCore(self.config['machine_learning'])
        self.relationship_extractor = RelationshipExtractor(self.config)
        
        self.logger.info("CCGL LLM 系统初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    async def natural_language_query(self, query: str) -> Dict[str, Any]:
        """自然语言查询处理"""
        try:
            self.logger.info(f"处理自然语言查询: {query}")
            
            # 1. 解析查询意图
            intent = await self.llm_manager.parse_query_intent(query)
            self.logger.info(f"查询意图: {intent}")
            
            # 2. 生成分析策略
            strategy = await self.llm_manager.generate_analysis_strategy(intent)
            
            # 3. 执行分析
            results = await self._execute_analysis_strategy(strategy)
            
            # 4. 生成自然语言报告
            report = await self.llm_manager.generate_report(results, query)
            
            return {
                'query': query,
                'intent': intent,
                'strategy': strategy,
                'results': results,
                'report': report
            }
            
        except Exception as e:
            self.logger.error(f"自然语言查询处理失败: {e}")
            return {'error': str(e)}
    
    async def _execute_analysis_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """执行分析策略"""
        results = {}
        
        try:
            # 获取数据
            if not await self.data_connection.test_connection():
                raise Exception("数据库连接失败")
            
            data = await self.data_connection.get_data(
                filters=strategy.get('filters', {}),
                limit=strategy.get('limit', 1000)
            )
            
            if data is None or data.empty:
                return {'message': '没有找到匹配的数据'}
            
            # 数据预处理
            processed_data = await self.data_preprocessor.process(data)
            
            # 根据策略执行分析
            analysis_type = strategy.get('analysis_type', 'summary')
            
            if analysis_type == 'clustering':
                results['clustering'] = await self.analysis_core.cluster_analysis(processed_data)
            elif analysis_type == 'anomaly':
                results['anomaly'] = await self.analysis_core.anomaly_detection(processed_data)
            elif analysis_type == 'association':
                results['association'] = await self.analysis_core.association_rules(processed_data)
            elif analysis_type == 'relationship':
                results['relationships'] = await self.relationship_extractor.extract_relationships(processed_data)
            else:
                # 默认执行数据摘要
                results['summary'] = {
                    'total_records': len(data),
                    'columns': list(data.columns),
                    'data_types': data.dtypes.to_dict(),
                    'missing_values': data.isnull().sum().to_dict(),
                    'basic_stats': data.describe().to_dict()
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"分析策略执行失败: {e}")
            return {'error': str(e)}
    
    async def intelligent_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """智能推荐系统"""
        try:
            self.logger.info("生成智能推荐")
            
            # 获取历史数据和当前状态
            current_data = await self.data_connection.get_recent_data()
            
            # 使用LLM分析数据模式
            patterns = await self.llm_manager.analyze_data_patterns(current_data)
            
            # 生成推荐
            recommendations = await self.llm_manager.generate_recommendations(patterns, context)
            
            return {
                'patterns': patterns,
                'recommendations': recommendations,
                'confidence': recommendations.get('confidence', 0.8)
            }
            
        except Exception as e:
            self.logger.error(f"智能推荐生成失败: {e}")
            return {'error': str(e)}
    
    async def interactive_mode(self):
        """交互式对话模式"""
        print("=== CCGL AI增强交互式模式 ===")
        print("您可以用自然语言提问，例如:")
        print("- '显示最近一周的异常数据'")
        print("- '分析库存商品的聚类模式'")
        print("- '找出销量和季节的关联规则'")
        print("- '推荐优化库存管理的策略'")
        print("输入 'quit' 退出")
        print()
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                print("AI正在思考...")
                
                # 处理用户查询
                if user_input.startswith('推荐') or 'recommend' in user_input.lower():
                    # 智能推荐模式
                    context = {'conversation_history': conversation_history}
                    response = await self.intelligent_recommendation(context)
                else:
                    # 自然语言查询模式
                    response = await self.natural_language_query(user_input)
                
                # 显示AI回复
                if 'error' in response:
                    print(f"AI: 抱歉，处理您的请求时出现错误: {response['error']}")
                elif 'report' in response:
                    print(f"AI: {response['report']}")
                elif 'recommendations' in response:
                    print("AI: 基于数据分析，我为您提供以下推荐:")
                    for i, rec in enumerate(response['recommendations'].get('items', []), 1):
                        print(f"  {i}. {rec}")
                else:
                    print(f"AI: {response}")
                
                # 记录对话历史
                conversation_history.append({
                    'user': user_input,
                    'ai': response
                })
                
                # 限制历史记录长度
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
                print()
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"处理错误: {e}")
    
    async def batch_analysis(self, queries_file: str):
        """批量分析模式"""
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = []
            for i, query in enumerate(queries, 1):
                print(f"处理查询 {i}/{len(queries)}: {query}")
                result = await self.natural_language_query(query)
                results.append(result)
            
            # 保存结果
            import json
            output_file = f"batch_analysis_results_{asyncio.get_event_loop().time()}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"批量分析完成，结果保存到: {output_file}")
            return results
            
        except Exception as e:
            self.logger.error(f"批量分析失败: {e}")
            return []
    
    def close(self):
        """关闭系统"""
        self.data_connection.close()
        self.logger.info("CCGL LLM 系统已关闭")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CCGL 仓储管理系统 - AI增强")
    parser.add_argument('-c', '--config', default='config.yml', help='配置文件路径')
    parser.add_argument('-i', '--interactive', action='store_true', help='交互式对话模式')
    parser.add_argument('-q', '--query', help='单次自然语言查询')
    parser.add_argument('-b', '--batch', help='批量分析文件路径')
    parser.add_argument('-r', '--recommend', action='store_true', help='智能推荐模式')
    parser.add_argument('--model', help='指定使用的AI模型 (openai, anthropic, zhipu, dashscope)')
    
    args = parser.parse_args()
    
    try:
        # 初始化LLM系统
        llm_system = CCGLLLMSystem(args.config)
        
        # 设置模型
        if args.model:
            llm_system.llm_manager.set_active_model(args.model)
        
        if args.interactive:
            # 交互式模式
            await llm_system.interactive_mode()
            
        elif args.query:
            # 单次查询模式
            print(f"处理查询: {args.query}")
            result = await llm_system.natural_language_query(args.query)
            if 'report' in result:
                print(f"分析结果: {result['report']}")
            else:
                print(f"查询结果: {result}")
                
        elif args.batch:
            # 批量分析模式
            print(f"开始批量分析: {args.batch}")
            await llm_system.batch_analysis(args.batch)
            
        elif args.recommend:
            # 智能推荐模式
            print("生成智能推荐...")
            recommendations = await llm_system.intelligent_recommendation({})
            print(f"推荐结果: {recommendations}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"系统错误: {e}")
        sys.exit(1)
    finally:
        if 'llm_system' in locals():
            llm_system.close()


if __name__ == "__main__":
    asyncio.run(main())