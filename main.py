#!/usr/bin/env python3
"""
CCGL 仓储管理系统 - 基础分析主程序

提供基础的数据分析和机器学习功能
"""

import argparse
import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# 确保ccgl_analytics模块可以被导入
sys.path.insert(0, str(Path(__file__).parent))

from ccgl_analytics.modules.data_connection import DataConnection
from ccgl_analytics.modules.data_preprocessing import DataPreprocessor
from ccgl_analytics.modules.analysis_core import AnalysisCore
from ccgl_analytics.modules.web_dashboard import WebDashboard
from ccgl_analytics.modules.result_output import ResultOutput
from ccgl_analytics.utils.logger import setup_logger


class CCGLAnalytics:
    """CCGL 仓储分析系统主类"""
    
    def __init__(self, config_path: str):
        """初始化系统"""
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config.get('logging', {}))
        
        # 初始化核心组件
        self.data_connection = DataConnection(self.config['database'])
        self.data_preprocessor = DataPreprocessor(self.config['data_processing'])
        self.analysis_core = AnalysisCore(self.config['machine_learning'])
        self.result_output = ResultOutput(self.config['storage'])
        
        self.logger.info("CCGL Analytics 系统初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    async def run_analysis(self, analysis_type: str = "full"):
        """运行数据分析"""
        try:
            self.logger.info(f"开始执行 {analysis_type} 分析")
            
            # 1. 数据连接测试
            if not await self.data_connection.test_connection():
                raise Exception("数据库连接失败")
            
            # 2. 获取数据
            data = await self.data_connection.get_data()
            if data is None or data.empty:
                self.logger.warning("未获取到数据")
                return
            
            self.logger.info(f"成功获取 {len(data)} 条记录")
            
            # 3. 数据预处理
            processed_data = await self.data_preprocessor.process(data)
            self.logger.info("数据预处理完成")
            
            # 4. 执行分析
            results = {}
            if analysis_type in ["full", "clustering"]:
                clustering_result = await self.analysis_core.cluster_analysis(processed_data)
                results['clustering'] = clustering_result
            
            if analysis_type in ["full", "anomaly"]:
                anomaly_result = await self.analysis_core.anomaly_detection(processed_data)
                results['anomaly'] = anomaly_result
            
            if analysis_type in ["full", "association"]:
                association_result = await self.analysis_core.association_rules(processed_data)
                results['association'] = association_result
            
            # 5. 输出结果
            await self.result_output.save_results(results)
            self.logger.info("分析结果已保存")
            
            return results
            
        except Exception as e:
            self.logger.error(f"分析执行失败: {e}")
            raise
    
    async def start_web_dashboard(self):
        """启动Web仪表板"""
        try:
            dashboard = WebDashboard(self.config['web'])
            await dashboard.start()
        except Exception as e:
            self.logger.error(f"Web仪表板启动失败: {e}")
            raise
    
    def close(self):
        """关闭系统"""
        self.data_connection.close()
        self.logger.info("CCGL Analytics 系统已关闭")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CCGL 仓储管理系统 - 基础分析")
    parser.add_argument('-c', '--config', default='config.yml', help='配置文件路径')
    parser.add_argument('-a', '--analysis', default='full', 
                       choices=['full', 'clustering', 'anomaly', 'association'],
                       help='分析类型')
    parser.add_argument('-w', '--web', action='store_true', help='启动Web仪表板')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    try:
        # 初始化系统
        ccgl = CCGLAnalytics(args.config)
        
        if args.web:
            # 启动Web仪表板
            print("启动Web仪表板...")
            await ccgl.start_web_dashboard()
        else:
            # 执行数据分析
            print(f"执行 {args.analysis} 分析...")
            results = await ccgl.run_analysis(args.analysis)
            
            if args.verbose and results:
                print("\n=== 分析结果摘要 ===")
                for analysis_type, result in results.items():
                    print(f"{analysis_type}: {len(result) if isinstance(result, (list, dict)) else 'completed'}")
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"系统错误: {e}")
        sys.exit(1)
    finally:
        if 'ccgl' in locals():
            ccgl.close()


if __name__ == "__main__":
    asyncio.run(main())