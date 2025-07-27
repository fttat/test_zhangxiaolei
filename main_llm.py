#!/usr/bin/env python3
"""
CCGL仓储管理系统数据分析工程 - AI增强模式

集成大语言模型，提供自然语言查询和智能业务洞察功能。
"""

import os
import sys
import pandas as pd
import webbrowser
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ccgl_analytics.modules.data_connection import DataConnectionManager
from ccgl_analytics.modules.data_preprocessing import DataPreprocessor  
from ccgl_analytics.modules.analysis_core import AnalysisCore
from ccgl_analytics.modules.llm_config_manager import LLMConfigManager
from ccgl_analytics.modules.web_dashboard import WebDashboard
from ccgl_analytics.utils.config_loader import ConfigLoader
from ccgl_analytics.utils.logger_setup import setup_logger

class CCGLAnalyticsLLM:
    """CCGL分析系统AI增强版"""
    
    def __init__(self):
        """初始化AI增强系统"""
        self.logger = setup_logger("ccgl_llm")
        self.config_loader = ConfigLoader()
        self.db_manager = None
        self.preprocessor = None
        self.analyzer = None
        self.llm_manager = None
        self.dashboard = None
        
    def initialize_system(self):
        """初始化系统组件"""
        try:
            self.logger.info("=== CCGL仓储管理系统数据分析工程启动 ===")
            self.logger.info("模式: AI增强分析模式 🤖")
            
            # 加载配置
            config = self.config_loader.load_config()
            self.logger.info("配置加载完成")
            
            # 初始化基础组件
            db_config = config.get('database', {})
            self.db_manager = DataConnectionManager(db_config)
            self.logger.info("数据库连接管理器初始化完成")
            
            preprocessing_config = config.get('preprocessing', {})
            self.preprocessor = DataPreprocessor(preprocessing_config)
            self.logger.info("数据预处理器初始化完成")
            
            analysis_config = config.get('analysis', {})
            self.analyzer = AnalysisCore(analysis_config)
            self.logger.info("分析核心初始化完成")
            
            # 初始化AI组件
            llm_config = config.get('llm', {})
            self.llm_manager = LLMConfigManager(llm_config)
            self.logger.info("LLM配置管理器初始化完成")
            
            # 初始化Web仪表板
            web_config = config.get('web', {})
            self.dashboard = WebDashboard(web_config)
            self.logger.info("Web仪表板初始化完成")
            
            # 显示AI能力状态
            self._show_ai_capabilities()
            
            self.logger.info("AI增强系统初始化完成 ✨")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    def _show_ai_capabilities(self):
        """显示AI能力状态"""
        config_summary = self.llm_manager.get_config_summary()
        
        self.logger.info("🤖 AI能力状态:")
        self.logger.info(f"  - 可用AI提供商: {', '.join(config_summary['available_providers']) if config_summary['available_providers'] else '无 (将使用模拟模式)'}")
        self.logger.info(f"  - 默认模型: {config_summary['default_model']}")
        self.logger.info(f"  - OpenAI可用: {'✅' if config_summary['openai_available'] else '❌'}")
        self.logger.info(f"  - Anthropic可用: {'✅' if config_summary['anthropic_available'] else '❌'}")
        
        if not config_summary['available_providers']:
            self.logger.warning("💡 提示: 配置AI API密钥以获得真正的AI增强功能")
    
    def generate_sample_data(self) -> pd.DataFrame:
        """生成示例仓储数据"""
        try:
            import numpy as np
            
            # 生成更真实的仓储管理数据
            n_samples = 1200
            np.random.seed(42)
            
            # 仓库和产品数据
            warehouses = ['北京智能仓储中心', '上海物流枢纽', '广州配送中心', '深圳电商仓库']
            products = ['智能手机', '笔记本电脑', '平板电脑', '智能手表', '无线耳机', '充电宝']
            suppliers = ['华为技术', '小米科技', '苹果公司', '三星电子', '联想集团']
            
            # 生成相关性更强的数据
            base_prices = {'智能手机': 3000, '笔记本电脑': 6000, '平板电脑': 2500, 
                          '智能手表': 1500, '无线耳机': 800, '充电宝': 200}
            
            data = {
                'warehouse_id': np.random.choice(warehouses, n_samples),
                'product_id': np.random.choice(products, n_samples),
                'supplier_id': np.random.choice(suppliers, n_samples),
                'quantity': np.random.randint(50, 2000, n_samples),
                'unit_price': [],
                'storage_cost': np.random.uniform(5, 100, n_samples),
                'order_date': pd.date_range(start='2024-01-01', periods=n_samples, freq='6H'),
                'delivery_days': np.random.randint(1, 20, n_samples),
                'quality_score': np.random.uniform(0.7, 1.0, n_samples),
                'temperature': np.random.uniform(18, 26, n_samples),
                'humidity': np.random.uniform(35, 75, n_samples),
                'customer_satisfaction': np.random.uniform(3.0, 5.0, n_samples)
            }
            
            # 基于产品类型生成价格
            for product in data['product_id']:
                base_price = base_prices.get(product, 1000)
                variation = np.random.uniform(0.8, 1.3)
                data['unit_price'].append(round(base_price * variation, 2))
            
            df = pd.DataFrame(data)
            
            # 添加业务逻辑关联
            # 高价值产品需要更好的存储条件
            high_value_mask = df['unit_price'] > 3000
            df.loc[high_value_mask, 'storage_cost'] *= 1.5
            df.loc[high_value_mask, 'quality_score'] += 0.1
            df.loc[high_value_mask, 'quality_score'] = df.loc[high_value_mask, 'quality_score'].clip(0, 1)
            
            # 添加季节性因素
            df['month'] = df['order_date'].dt.month
            holiday_months = [11, 12, 1, 2]  # 假日购物季
            holiday_mask = df['month'].isin(holiday_months)
            df.loc[holiday_mask, 'quantity'] *= 1.3
            df.loc[holiday_mask, 'unit_price'] *= 1.1
            
            # 引入一些现实的缺失值和异常
            missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
            df.loc[missing_indices[:len(missing_indices)//2], 'quality_score'] = np.nan
            df.loc[missing_indices[len(missing_indices)//2:], 'customer_satisfaction'] = np.nan
            
            # 引入合理的异常值（例如促销活动导致的大订单）
            promotion_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
            df.loc[promotion_indices, 'quantity'] *= 5
            df.loc[promotion_indices, 'unit_price'] *= 0.7  # 促销降价
            
            self.logger.info(f"生成AI增强示例数据: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"生成示例数据失败: {e}")
            raise
    
    def run_enhanced_analysis(self, df: pd.DataFrame) -> dict:
        """运行AI增强的综合分析"""
        try:
            self.logger.info("🚀 开始AI增强数据分析")
            
            # 1. 基础数据分析（与main.py相同）
            self.logger.info("步骤1: 基础数据预处理")
            cleaned_data, preprocessing_report = self.preprocessor.clean_data(
                df, 
                missing_strategy='auto',
                outlier_method='iqr',
                scaling_method='standard'
            )
            
            self.logger.info("步骤2: 机器学习分析")
            clustering_results = self.analyzer.clustering_analysis(
                cleaned_data,
                methods=['kmeans', 'dbscan', 'hierarchical']
            )
            
            anomaly_results = self.analyzer.anomaly_detection(
                cleaned_data,
                methods=['isolation_forest', 'one_class_svm']
            )
            
            reduction_results = self.analyzer.dimensionality_reduction(
                cleaned_data,
                methods=['pca', 'tsne'],
                n_components=2
            )
            
            enhanced_data, feature_report = self.preprocessor.create_features(cleaned_data)
            
            # 2. AI增强业务洞察生成
            self.logger.info("步骤3: 🤖 AI业务洞察生成")
            
            # 构建分析结果
            base_analysis = {
                'data_preprocessing': preprocessing_report,
                'feature_engineering': feature_report,
                'clustering_analysis': clustering_results,
                'anomaly_detection': anomaly_results,
                'dimensionality_reduction': reduction_results
            }
            
            # 使用AI生成业务洞察
            ai_insights = self.llm_manager.generate_business_insights(base_analysis)
            
            # 3. 智能数据查询演示
            self.logger.info("步骤4: 🔍 智能数据查询演示")
            data_context = self._prepare_data_context(df, base_analysis)
            
            # 演示几个自然语言查询
            demo_queries = [
                "分析我们的库存数据中最需要关注的问题是什么？",
                "哪个仓库的运营效率最高？为什么？",
                "异常数据可能表明什么业务问题？",
                "如何优化我们的库存管理策略？"
            ]
            
            query_results = []
            for query in demo_queries:
                result = self.llm_manager.natural_language_query(query, data_context)
                query_results.append(result)
                self.logger.info(f"  ✓ 完成查询: {query[:30]}...")
            
            # 4. 生成AI增强仪表板
            self.logger.info("步骤5: 📊 生成AI增强仪表板")
            enhanced_analysis = {
                **base_analysis,
                'ai_insights': ai_insights,
                'natural_language_queries': query_results
            }
            
            dashboard_html = self.dashboard.create_fastapi_dashboard({
                'data_quality_assessment': self._create_quality_assessment(df),
                'comprehensive_analysis': enhanced_analysis
            })
            
            dashboard_file = self.dashboard.save_dashboard(
                dashboard_html, 
                f"ccgl_ai_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            
            # 构建最终结果
            final_results = {
                'mode': 'ai_enhanced_analysis',
                'analysis_time': datetime.now().isoformat(),
                'data_preprocessing': preprocessing_report,
                'machine_learning_analysis': {
                    'clustering': clustering_results,
                    'anomaly_detection': anomaly_results,
                    'dimensionality_reduction': reduction_results
                },
                'feature_engineering': feature_report,
                'ai_insights': ai_insights,
                'natural_language_queries': query_results,
                'dashboard_file': dashboard_file,
                'summary': self._generate_ai_summary(
                    preprocessing_report, clustering_results, anomaly_results, ai_insights
                )
            }
            
            self.logger.info("AI增强数据分析完成 🎉")
            return final_results
            
        except Exception as e:
            self.logger.error(f"AI增强分析失败: {e}")
            raise
    
    def _prepare_data_context(self, df: pd.DataFrame, analysis_results: dict) -> str:
        """准备数据上下文用于AI查询"""
        context = f"""
        仓储管理数据分析上下文：
        
        📊 数据基本信息：
        - 总记录数: {len(df)} 条
        - 数据维度: {len(df.columns)} 列
        - 仓库数量: {df['warehouse_id'].nunique()}
        - 产品种类: {df['product_id'].nunique()}
        - 供应商数量: {df['supplier_id'].nunique()}
        - 时间范围: {df['order_date'].min()} 到 {df['order_date'].max()}
        
        📈 关键指标统计：
        - 平均库存数量: {df['quantity'].mean():.0f}
        - 平均单价: {df['unit_price'].mean():.2f}元
        - 平均存储成本: {df['storage_cost'].mean():.2f}元
        - 平均质量评分: {df['quality_score'].mean():.2f}
        - 平均配送天数: {df['delivery_days'].mean():.1f}天
        - 平均客户满意度: {df['customer_satisfaction'].mean():.2f}/5.0
        
        🎯 机器学习分析结果：
        - 最优聚类数量: {analysis_results.get('clustering_analysis', {}).get('optimal_clusters', 'N/A')}
        - 检测到的异常点: {analysis_results.get('anomaly_detection', {}).get('summary', {}).get('consensus_anomalies', 0)} 个
        
        🏷️ 主要业务特征：
        - 主要仓库: {', '.join(df['warehouse_id'].value_counts().head(3).index.tolist())}
        - 热门产品: {', '.join(df['product_id'].value_counts().head(3).index.tolist())}
        - 主要供应商: {', '.join(df['supplier_id'].value_counts().head(3).index.tolist())}
        
        💰 财务指标：
        - 总库存价值: {(df['quantity'] * df['unit_price']).sum():,.0f}元
        - 总存储成本: {(df['quantity'] * df['storage_cost']).sum():,.0f}元
        - 平均库存周转: 基于{df['delivery_days'].mean():.1f}天配送周期
        """
        
        return context
    
    def _create_quality_assessment(self, df: pd.DataFrame) -> dict:
        """创建数据质量评估（简化版）"""
        return {
            'table_name': 'ai_enhanced_warehouse_data',
            'assessment_time': datetime.now().isoformat(),
            'basic_stats': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': df.select_dtypes(include=['number']).shape[1],
                'categorical_columns': df.select_dtypes(include=['object']).shape[1],
                'datetime_columns': df.select_dtypes(include=['datetime64']).shape[1]
            },
            'data_quality_metrics': {
                'completeness': (df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]),
                'consistency': 0.96,
                'validity': 0.97,
                'uniqueness': len(df.drop_duplicates()) / len(df)
            },
            'missing_values': {
                'total_missing': int(df.isnull().sum().sum()),
                'missing_by_column': df.isnull().sum().to_dict()
            }
        }
    
    def _generate_ai_summary(self, preprocessing_report, clustering_results, 
                           anomaly_results, ai_insights) -> dict:
        """生成AI增强的分析摘要"""
        summary = {
            'data_processing': {
                'original_shape': preprocessing_report.get('original_shape'),
                'final_shape': preprocessing_report.get('final_shape'),
                'rows_removed': preprocessing_report.get('removed_rows', 0)
            },
            'machine_learning': {
                'optimal_clusters': clustering_results.get('optimal_clusters'),
                'anomalies_detected': anomaly_results.get('summary', {}).get('consensus_anomalies', 0)
            },
            'ai_enhancement': {
                'insights_generated': ai_insights.get('success', False),
                'ai_model_used': ai_insights.get('insights', {}).get('response', '').startswith('📈') if ai_insights.get('success') else False,
                'business_recommendations': True if ai_insights.get('success') else False
            }
        }
        
        return summary
    
    def save_ai_results(self, results: dict, output_dir: str = "output"):
        """保存AI增强分析结果"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存完整结果
            import json
            results_file = output_path / f"ccgl_ai_analysis_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存AI洞察报告
            if results.get('ai_insights', {}).get('success'):
                insights_file = output_path / f"ccgl_ai_insights_{timestamp}.txt"
                with open(insights_file, 'w', encoding='utf-8') as f:
                    f.write("CCGL仓储管理系统 - AI增强分析洞察报告\\n")
                    f.write("=" * 60 + "\\n")
                    f.write(f"生成时间: {results.get('analysis_time')}\\n\\n")
                    
                    ai_response = results['ai_insights'].get('insights', {}).get('response', '')
                    f.write("🤖 AI业务洞察:\\n")
                    f.write(ai_response)
                    f.write("\\n\\n")
                    
                    f.write("🔍 自然语言查询结果:\\n")
                    for i, query_result in enumerate(results.get('natural_language_queries', []), 1):
                        if query_result.get('success'):
                            f.write(f"{i}. 查询: {query_result.get('query')}\\n")
                            f.write(f"   回答: {query_result.get('response', '')[:200]}...\\n\\n")
            
            # 保存简化摘要
            summary_file = output_path / f"ccgl_ai_summary_{timestamp}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("CCGL仓储管理系统 - AI增强分析摘要\\n")
                f.write("=" * 50 + "\\n")
                f.write(f"分析时间: {results.get('analysis_time')}\\n\\n")
                
                summary = results.get('summary', {})
                f.write("数据处理摘要:\\n")
                f.write(f"  - 原始数据: {summary.get('data_processing', {}).get('original_shape')}\\n")
                f.write(f"  - 处理后: {summary.get('data_processing', {}).get('final_shape')}\\n")
                f.write(f"  - 聚类数量: {summary.get('machine_learning', {}).get('optimal_clusters')}\\n")
                f.write(f"  - 异常点: {summary.get('machine_learning', {}).get('anomalies_detected')}\\n")
                f.write(f"  - AI洞察: {'✅' if summary.get('ai_enhancement', {}).get('insights_generated') else '❌'}\\n")
                f.write(f"  - 仪表板: {results.get('dashboard_file', 'N/A')}\\n")
            
            self.logger.info(f"AI分析结果已保存到: {output_path}")
            return str(results_file), str(summary_file)
            
        except Exception as e:
            self.logger.error(f"保存AI结果失败: {e}")
            raise
    
    def interactive_ai_query(self):
        """交互式AI查询界面"""
        self.logger.info("\\n🤖 进入交互式AI查询模式")
        self.logger.info("输入'quit'或'exit'退出")
        
        # 准备一些示例数据上下文
        sample_context = """
        当前仓储系统包含4个主要仓库，6种产品类别，5个供应商。
        平均库存量约1000件，单价范围500-8000元。
        检测到4个主要的产品群体和约15个异常数据点。
        """
        
        while True:
            try:
                user_query = input("\\n🔍 请输入您的查询 (或输入'quit'退出): ").strip()
                
                if user_query.lower() in ['quit', 'exit', '退出']:
                    self.logger.info("退出交互式查询模式")
                    break
                
                if not user_query:
                    continue
                
                self.logger.info(f"🤖 AI正在处理您的查询...")
                result = self.llm_manager.natural_language_query(user_query, sample_context)
                
                if result.get('success'):
                    print("\\n" + "="*60)
                    print("🤖 AI回答:")
                    print("="*60)
                    print(result.get('response', ''))
                    print("="*60)
                    print(f"使用模型: {result.get('model_used', 'unknown')}")
                else:
                    print(f"\\n❌ 查询失败: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                self.logger.info("\\n用户中断，退出交互模式")
                break
            except Exception as e:
                self.logger.error(f"查询过程出错: {e}")
    
    def run(self):
        """运行AI增强主程序"""
        try:
            # 1. 初始化系统
            if not self.initialize_system():
                return False
            
            # 2. 生成示例数据
            self.logger.info("准备AI增强数据...")
            sample_data = self.generate_sample_data()
            
            # 3. 运行AI增强分析
            self.logger.info("运行AI增强综合分析...")
            analysis_results = self.run_enhanced_analysis(sample_data)
            
            # 4. 保存结果
            self.logger.info("保存AI分析结果...")
            results_file, summary_file = self.save_ai_results(analysis_results)
            
            # 5. 显示完成信息
            self.logger.info("=" * 70)
            self.logger.info("🎉 CCGL仓储管理系统AI增强分析完成！")
            self.logger.info("=" * 70)
            self.logger.info(f"🤖 分析模式: AI增强分析模式")
            self.logger.info(f"📁 详细结果: {results_file}")
            self.logger.info(f"📄 摘要报告: {summary_file}")
            self.logger.info(f"📊 AI仪表板: {analysis_results.get('dashboard_file', 'N/A')}")
            self.logger.info(f"🔍 数据样本: {sample_data.shape[0]} 行, {sample_data.shape[1]} 列")
            
            # 显示AI洞察预览
            if analysis_results.get('ai_insights', {}).get('success'):
                ai_response = analysis_results['ai_insights']['insights'].get('response', '')
                preview = ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
                self.logger.info(f"🧠 AI洞察预览: {preview}")
            
            # 询问是否打开仪表板
            dashboard_file = analysis_results.get('dashboard_file')
            if dashboard_file and Path(dashboard_file).exists():
                try:
                    response = input("\\n🌐 是否在浏览器中打开AI仪表板? (y/N): ").strip().lower()
                    if response in ['y', 'yes', '是']:
                        webbrowser.open(f"file://{Path(dashboard_file).absolute()}")
                        self.logger.info("📊 仪表板已在浏览器中打开")
                except:
                    pass
            
            # 询问是否进入交互式AI查询
            try:
                response = input("\\n💬 是否进入交互式AI查询模式? (y/N): ").strip().lower()
                if response in ['y', 'yes', '是']:
                    self.interactive_ai_query()
            except:
                pass
            
            self.logger.info("\\n💡 提示:")
            self.logger.info("  - 使用 python main.py 运行基础分析模式")
            self.logger.info("  - 使用 python main_mcp.py 运行MCP架构模式")
            self.logger.info("  - 配置AI API密钥以获得更强大的分析能力")
            
            return True
            
        except Exception as e:
            self.logger.error(f"AI增强程序运行失败: {e}")
            return False
        
        finally:
            # 清理资源
            if self.db_manager:
                self.db_manager.close()

def main():
    """主入口函数"""
    app = CCGLAnalyticsLLM()
    success = app.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()