#!/usr/bin/env python3
"""
CCGL仓储管理系统数据分析工程 - 主程序（基础分析模式）

提供标准的数据连接、质量评估和机器学习分析功能。
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ccgl_analytics.modules.data_connection import DataConnectionManager
from ccgl_analytics.modules.data_preprocessing import DataPreprocessor  
from ccgl_analytics.modules.analysis_core import AnalysisCore
from ccgl_analytics.utils.config_loader import ConfigLoader
from ccgl_analytics.utils.logger_setup import setup_logger

class CCGLAnalyticsMain:
    """CCGL分析系统主程序"""
    
    def __init__(self):
        """初始化主程序"""
        self.logger = setup_logger("ccgl_main")
        self.config_loader = ConfigLoader()
        self.db_manager = None
        self.preprocessor = None
        self.analyzer = None
        
    def initialize_system(self):
        """初始化系统组件"""
        try:
            self.logger.info("=== CCGL仓储管理系统数据分析工程启动 ===")
            self.logger.info("模式: 基础分析模式")
            
            # 加载配置
            config = self.config_loader.load_config()
            self.logger.info("配置加载完成")
            
            # 初始化数据库连接管理器
            db_config = config.get('database', {})
            if not db_config:
                # 使用示例配置
                db_config = {
                    'host': 'localhost',
                    'port': 3306,
                    'database': 'ccgl_warehouse',
                    'user': 'root',
                    'password': 'password'
                }
                self.logger.warning("使用默认数据库配置，请检查配置文件")
            
            # 注意：实际部署时需要真实的数据库连接
            # 这里我们使用模拟数据进行演示
            self.db_manager = DataConnectionManager(db_config)
            self.logger.info("数据库连接管理器初始化完成")
            
            # 初始化数据预处理器
            preprocessing_config = config.get('preprocessing', {})
            self.preprocessor = DataPreprocessor(preprocessing_config)
            self.logger.info("数据预处理器初始化完成")
            
            # 初始化分析核心
            analysis_config = config.get('analysis', {})
            self.analyzer = AnalysisCore(analysis_config)
            self.logger.info("分析核心初始化完成")
            
            self.logger.info("系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    def generate_sample_data(self) -> pd.DataFrame:
        """生成示例仓储数据用于演示"""
        try:
            import numpy as np
            
            # 生成示例仓储管理数据
            n_samples = 1000
            np.random.seed(42)
            
            # 仓库信息
            warehouses = ['仓库A', '仓库B', '仓库C', '仓库D']
            products = ['产品001', '产品002', '产品003', '产品004', '产品005']
            suppliers = ['供应商甲', '供应商乙', '供应商丙']
            
            data = {
                'warehouse_id': np.random.choice(warehouses, n_samples),
                'product_id': np.random.choice(products, n_samples),
                'supplier_id': np.random.choice(suppliers, n_samples),
                'quantity': np.random.randint(1, 1000, n_samples),
                'unit_price': np.random.uniform(10, 500, n_samples),
                'storage_cost': np.random.uniform(1, 50, n_samples),
                'order_date': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
                'delivery_days': np.random.randint(1, 30, n_samples),
                'quality_score': np.random.uniform(0.6, 1.0, n_samples),
                'temperature': np.random.uniform(15, 25, n_samples),
                'humidity': np.random.uniform(40, 80, n_samples)
            }
            
            # 添加一些缺失值和异常值来模拟真实数据
            df = pd.DataFrame(data)
            
            # 引入缺失值
            missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
            df.loc[missing_indices[:len(missing_indices)//2], 'quality_score'] = np.nan
            df.loc[missing_indices[len(missing_indices)//2:], 'storage_cost'] = np.nan
            
            # 引入异常值
            outlier_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
            df.loc[outlier_indices, 'unit_price'] = df.loc[outlier_indices, 'unit_price'] * 10
            
            self.logger.info(f"生成示例数据: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"生成示例数据失败: {e}")
            raise
    
    def run_data_quality_assessment(self, df: pd.DataFrame) -> dict:
        """运行数据质量评估"""
        try:
            self.logger.info("开始数据质量评估")
            
            # 模拟表数据质量评估
            quality_report = {
                'table_name': 'sample_warehouse_data',
                'assessment_time': datetime.now().isoformat(),
                'basic_stats': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'numeric_columns': df.select_dtypes(include=['number']).shape[1],
                    'categorical_columns': df.select_dtypes(include=['object']).shape[1],
                    'datetime_columns': df.select_dtypes(include=['datetime64']).shape[1]
                },
                'data_quality_metrics': {
                    'completeness': (df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]),
                    'consistency': 0.95,  # 模拟值
                    'validity': 0.98,     # 模拟值
                    'uniqueness': len(df.drop_duplicates()) / len(df)
                },
                'missing_values': {
                    'total_missing': int(df.isnull().sum().sum()),
                    'missing_by_column': df.isnull().sum().to_dict(),
                    'missing_percentage_by_column': (df.isnull().sum() / len(df) * 100).to_dict()
                }
            }
            
            self.logger.info("数据质量评估完成")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"数据质量评估失败: {e}")
            raise
    
    def run_comprehensive_analysis(self, df: pd.DataFrame) -> dict:
        """运行综合数据分析"""
        try:
            self.logger.info("开始综合数据分析")
            
            # 1. 数据预处理
            self.logger.info("步骤1: 数据预处理")
            cleaned_data, preprocessing_report = self.preprocessor.clean_data(
                df, 
                missing_strategy='auto',
                outlier_method='iqr',
                scaling_method='standard'
            )
            
            # 2. 聚类分析
            self.logger.info("步骤2: 聚类分析")
            clustering_results = self.analyzer.clustering_analysis(
                cleaned_data,
                methods=['kmeans', 'dbscan', 'hierarchical'],
                n_clusters=None  # 自动确定
            )
            
            # 3. 异常检测
            self.logger.info("步骤3: 异常检测分析")
            anomaly_results = self.analyzer.anomaly_detection(
                cleaned_data,
                methods=['isolation_forest', 'one_class_svm']
            )
            
            # 4. 降维分析
            self.logger.info("步骤4: 降维分析")
            reduction_results = self.analyzer.dimensionality_reduction(
                cleaned_data,
                methods=['pca', 'tsne'],
                n_components=2
            )
            
            # 5. 特征工程
            self.logger.info("步骤5: 特征工程")
            enhanced_data, feature_report = self.preprocessor.create_features(cleaned_data)
            
            # 6. 关联规则挖掘（如果数据适合）
            self.logger.info("步骤6: 关联规则挖掘")
            try:
                rules_results = self.analyzer.association_rules_mining(
                    df[['warehouse_id', 'product_id', 'supplier_id']],
                    min_support=0.01,
                    min_confidence=0.5
                )
            except Exception as e:
                rules_results = {'error': str(e)}
                self.logger.warning(f"关联规则挖掘跳过: {e}")
            
            # 生成综合报告
            comprehensive_report = {
                'analysis_time': datetime.now().isoformat(),
                'data_preprocessing': preprocessing_report,
                'feature_engineering': feature_report,
                'clustering_analysis': clustering_results,
                'anomaly_detection': anomaly_results,
                'dimensionality_reduction': reduction_results,
                'association_rules': rules_results,
                'summary': self._generate_analysis_summary(
                    preprocessing_report, clustering_results, anomaly_results, reduction_results
                )
            }
            
            self.logger.info("综合数据分析完成")
            return comprehensive_report
            
        except Exception as e:
            self.logger.error(f"综合数据分析失败: {e}")
            raise
    
    def _generate_analysis_summary(self, preprocessing_report, clustering_results, 
                                 anomaly_results, reduction_results) -> dict:
        """生成分析摘要"""
        try:
            summary = {
                'data_processing': {
                    'original_shape': preprocessing_report.get('original_shape'),
                    'final_shape': preprocessing_report.get('final_shape'),
                    'rows_removed': preprocessing_report.get('removed_rows', 0),
                    'processing_steps': len(preprocessing_report.get('processing_steps', []))
                },
                'clustering': {
                    'optimal_clusters': clustering_results.get('optimal_clusters'),
                    'best_method': None,
                    'best_silhouette_score': None
                },
                'anomalies': {
                    'consensus_anomalies': anomaly_results.get('summary', {}).get('common_anomaly_count', 0),
                    'total_samples': anomaly_results.get('summary', {}).get('total_samples', 0)
                },
                'dimensionality_reduction': {
                    'original_dimensions': reduction_results.get('original_dimensions'),
                    'variance_explained': None
                }
            }
            
            # 找最佳聚类方法
            if 'evaluation_metrics' in clustering_results:
                best_score = -1
                best_method = None
                for method, metrics in clustering_results['evaluation_metrics'].items():
                    if 'silhouette_score' in metrics and metrics['silhouette_score'] > best_score:
                        best_score = metrics['silhouette_score']
                        best_method = method
                
                summary['clustering']['best_method'] = best_method
                summary['clustering']['best_silhouette_score'] = best_score
            
            # PCA方差解释比例
            if 'pca' in reduction_results.get('reduction_results', {}):
                pca_result = reduction_results['reduction_results']['pca']
                if 'cumulative_variance_ratio' in pca_result:
                    summary['dimensionality_reduction']['variance_explained'] = \
                        pca_result['cumulative_variance_ratio'][-1]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成分析摘要失败: {e}")
            return {'error': str(e)}
    
    def save_results(self, results: dict, output_dir: str = "output"):
        """保存分析结果"""
        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 保存主要结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存JSON格式的完整结果
            import json
            results_file = output_path / f"ccgl_analysis_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存简化的汇总报告
            summary_file = output_path / f"ccgl_analysis_summary_{timestamp}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("CCGL仓储管理系统数据分析报告\\n")
                f.write("=" * 50 + "\\n")
                f.write(f"分析时间: {results.get('analysis_time', 'N/A')}\\n\\n")
                
                if 'summary' in results:
                    summary = results['summary']
                    f.write("数据处理摘要:\\n")
                    f.write(f"  - 原始数据形状: {summary.get('data_processing', {}).get('original_shape')}\\n")
                    f.write(f"  - 处理后形状: {summary.get('data_processing', {}).get('final_shape')}\\n")
                    f.write(f"  - 移除行数: {summary.get('data_processing', {}).get('rows_removed')}\\n\\n")
                    
                    f.write("聚类分析摘要:\\n")
                    f.write(f"  - 最优聚类数: {summary.get('clustering', {}).get('optimal_clusters')}\\n")
                    f.write(f"  - 最佳方法: {summary.get('clustering', {}).get('best_method')}\\n")
                    f.write(f"  - 最佳轮廓系数: {summary.get('clustering', {}).get('best_silhouette_score')}\\n\\n")
                    
                    f.write("异常检测摘要:\\n")
                    f.write(f"  - 共识异常点: {summary.get('anomalies', {}).get('consensus_anomalies')}\\n")
                    f.write(f"  - 总样本数: {summary.get('anomalies', {}).get('total_samples')}\\n\\n")
            
            self.logger.info(f"分析结果已保存到: {output_path}")
            return str(results_file), str(summary_file)
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            raise
    
    def run(self):
        """运行主程序"""
        try:
            # 1. 初始化系统
            if not self.initialize_system():
                return False
            
            # 2. 生成示例数据（在实际部署中，这里会从数据库加载数据）
            self.logger.info("准备数据...")
            sample_data = self.generate_sample_data()
            
            # 3. 运行数据质量评估
            self.logger.info("运行数据质量评估...")
            quality_report = self.run_data_quality_assessment(sample_data)
            
            # 4. 运行综合分析
            self.logger.info("运行综合数据分析...")
            analysis_results = self.run_comprehensive_analysis(sample_data)
            
            # 5. 合并结果
            final_results = {
                'mode': 'basic_analysis',
                'data_quality_assessment': quality_report,
                'comprehensive_analysis': analysis_results
            }
            
            # 6. 保存结果
            self.logger.info("保存分析结果...")
            results_file, summary_file = self.save_results(final_results)
            
            # 7. 显示完成信息
            self.logger.info("=" * 60)
            self.logger.info("🎉 CCGL仓储管理系统数据分析完成！")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 分析模式: 基础分析模式")
            self.logger.info(f"📁 详细结果: {results_file}")
            self.logger.info(f"📄 摘要报告: {summary_file}")
            self.logger.info(f"🔍 数据样本: {sample_data.shape[0]} 行, {sample_data.shape[1]} 列")
            
            if 'summary' in analysis_results:
                summary = analysis_results['summary']
                self.logger.info(f"🎯 最优聚类数: {summary.get('clustering', {}).get('optimal_clusters')}")
                self.logger.info(f"🚨 异常点数量: {summary.get('anomalies', {}).get('consensus_anomalies')}")
            
            self.logger.info("\\n💡 提示:")
            self.logger.info("  - 使用 python main_mcp.py 启动MCP架构模式")
            self.logger.info("  - 使用 python main_llm.py 启动AI增强模式")
            self.logger.info("  - 使用 python quick_start.py 快速演示")
            
            return True
            
        except Exception as e:
            self.logger.error(f"程序运行失败: {e}")
            return False
        
        finally:
            # 清理资源
            if self.db_manager:
                self.db_manager.close()

def main():
    """主入口函数"""
    app = CCGLAnalyticsMain()
    success = app.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()