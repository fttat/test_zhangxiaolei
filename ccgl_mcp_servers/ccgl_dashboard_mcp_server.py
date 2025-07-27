"""
仪表板MCP服务器

专门处理数据可视化和报告生成的MCP服务器
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import pandas as pd

# 导入核心模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ccgl_analytics.modules.web_dashboard import WebDashboard
from ccgl_analytics.modules.result_output import ResultOutput


class DashboardMCPServer:
    """仪表板MCP服务器"""
    
    def __init__(self, port: int, config: Dict[str, Any]):
        """初始化仪表板服务器"""
        self.port = port
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化核心组件
        self.web_dashboard = WebDashboard(config.get('web', {}))
        self.result_output = ResultOutput(config.get('storage', {}))
        
        # 服务器状态
        self.is_running = False
        self.start_time = None
        self.processed_requests = 0
        
        # 仪表板缓存
        self.dashboard_cache = {}
        self.chart_templates = {}
        
        self._initialize_chart_templates()
        
        self.logger.info(f"仪表板MCP服务器初始化完成，端口: {port}")
    
    def _initialize_chart_templates(self):
        """初始化图表模板"""
        self.chart_templates = {
            'line_chart': {
                'type': 'line',
                'config': {
                    'x_axis': 'date',
                    'y_axis': 'value',
                    'title': '趋势图'
                }
            },
            'bar_chart': {
                'type': 'bar',
                'config': {
                    'x_axis': 'category',
                    'y_axis': 'count',
                    'title': '柱状图'
                }
            },
            'pie_chart': {
                'type': 'pie',
                'config': {
                    'value_field': 'value',
                    'label_field': 'label',
                    'title': '饼图'
                }
            },
            'scatter_plot': {
                'type': 'scatter',
                'config': {
                    'x_axis': 'x',
                    'y_axis': 'y',
                    'title': '散点图'
                }
            },
            'heatmap': {
                'type': 'heatmap',
                'config': {
                    'x_axis': 'x',
                    'y_axis': 'y',
                    'value_field': 'value',
                    'title': '热力图'
                }
            }
        }
    
    async def start(self):
        """启动服务器"""
        try:
            self.logger.info(f"启动仪表板MCP服务器，端口: {self.port}")
            
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
            self.logger.info("停止仪表板MCP服务器")
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"服务器停止失败: {e}")
    
    async def _start_mock_server(self):
        """启动模拟服务器"""
        self.logger.info(f"仪表板服务器运行在端口 {self.port}")
        
        # 模拟服务器持续运行
        while self.is_running:
            await asyncio.sleep(1)
    
    async def execute_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行仪表板任务"""
        try:
            self.processed_requests += 1
            self.logger.info(f"执行仪表板任务: {task_type}")
            
            if task_type == 'generate':
                return await self._handle_generate_dashboard_task(data)
            elif task_type == 'create_chart':
                return await self._handle_create_chart_task(data)
            elif task_type == 'export_dashboard':
                return await self._handle_export_dashboard_task(data)
            elif task_type == 'update_data':
                return await self._handle_update_data_task(data)
            elif task_type == 'get_dashboard':
                return await self._handle_get_dashboard_task(data)
            else:
                return {'error': f'不支持的任务类型: {task_type}'}
                
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            return {'error': str(e)}
    
    async def _handle_generate_dashboard_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成仪表板任务"""
        try:
            analysis_results = data.get('analysis_results', {})
            dashboard_config = data.get('dashboard_config', {})
            
            # 生成仪表板
            dashboard = await self._generate_dashboard(analysis_results, dashboard_config)
            
            # 缓存仪表板
            dashboard_id = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.dashboard_cache[dashboard_id] = dashboard
            
            return {
                'status': 'completed',
                'task_type': 'generate_dashboard',
                'dashboard_id': dashboard_id,
                'dashboard_url': f"http://localhost:{self.port}/dashboard/{dashboard_id}",
                'dashboard': dashboard,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"仪表板生成失败: {e}")
            return {'error': str(e)}
    
    async def _generate_dashboard(self, analysis_results: Dict[str, Any], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """生成仪表板"""
        dashboard = {
            'title': config.get('title', 'CCGL 数据分析仪表板'),
            'layout': config.get('layout', {'rows': 2, 'columns': 2}),
            'charts': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'analysis_results_count': len(analysis_results)
            }
        }
        
        # 根据分析结果自动生成图表
        chart_position = 0
        
        for analysis_type, result in analysis_results.items():
            charts = await self._generate_charts_for_analysis(analysis_type, result, chart_position)
            dashboard['charts'].extend(charts)
            chart_position += len(charts)
        
        return dashboard
    
    async def _generate_charts_for_analysis(self, analysis_type: str, 
                                          result: Dict[str, Any], 
                                          start_position: int) -> List[Dict[str, Any]]:
        """为特定分析类型生成图表"""
        charts = []
        
        if analysis_type == 'clustering':
            # 聚类结果图表
            charts.append({
                'id': f'clustering_chart_{start_position}',
                'type': 'pie',
                'title': '聚类分布',
                'data': self._prepare_clustering_chart_data(result),
                'position': {'row': start_position // 2, 'col': start_position % 2}
            })
            
        elif analysis_type == 'anomaly':
            # 异常检测图表
            charts.append({
                'id': f'anomaly_chart_{start_position}',
                'type': 'bar',
                'title': '异常检测结果',
                'data': self._prepare_anomaly_chart_data(result),
                'position': {'row': start_position // 2, 'col': start_position % 2}
            })
            
        elif analysis_type == 'association':
            # 关联规则图表
            charts.append({
                'id': f'association_chart_{start_position}',
                'type': 'network',
                'title': '关联规则网络',
                'data': self._prepare_association_chart_data(result),
                'position': {'row': start_position // 2, 'col': start_position % 2}
            })
            
        elif analysis_type == 'preprocessing':
            # 数据质量图表
            charts.append({
                'id': f'quality_chart_{start_position}',
                'type': 'bar',
                'title': '数据质量指标',
                'data': self._prepare_preprocessing_chart_data(result),
                'position': {'row': start_position // 2, 'col': start_position % 2}
            })
        
        return charts
    
    def _prepare_clustering_chart_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """准备聚类图表数据"""
        if 'cluster_statistics' in result:
            labels = []
            values = []
            
            for cluster_id, stats in result['cluster_statistics'].items():
                labels.append(f"簇 {cluster_id.split('_')[1]}")
                values.append(stats['size'])
            
            return {
                'labels': labels,
                'values': values,
                'type': 'pie'
            }
        
        return {'error': '聚类数据格式不正确'}
    
    def _prepare_anomaly_chart_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """准备异常检测图表数据"""
        if 'individual_results' in result:
            categories = []
            anomaly_counts = []
            
            for algorithm, algo_result in result['individual_results'].items():
                categories.append(algorithm)
                anomaly_counts.append(algo_result.get('n_anomalies', 0))
            
            return {
                'categories': categories,
                'values': anomaly_counts,
                'type': 'bar'
            }
        
        return {'error': '异常检测数据格式不正确'}
    
    def _prepare_association_chart_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """准备关联规则图表数据"""
        if 'rules' in result and isinstance(result['rules'], list):
            nodes = set()
            edges = []
            
            for rule in result['rules'][:10]:  # 限制显示规则数量
                antecedent = str(rule.get('antecedents', 'A'))
                consequent = str(rule.get('consequents', 'B'))
                confidence = rule.get('confidence', 0.5)
                
                nodes.add(antecedent)
                nodes.add(consequent)
                edges.append({
                    'source': antecedent,
                    'target': consequent,
                    'weight': confidence
                })
            
            return {
                'nodes': [{'id': node, 'label': node} for node in nodes],
                'edges': edges,
                'type': 'network'
            }
        
        return {'error': '关联规则数据格式不正确'}
    
    def _prepare_preprocessing_chart_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """准备预处理图表数据"""
        if 'summary' in result:
            summary = result['summary']
            
            metrics = ['原始记录数', '处理后记录数', '缺失值数量']
            values = [
                summary.get('original_shape', [0])[0],
                summary.get('processed_shape', [0])[0],
                summary.get('missing_values_before', 0)
            ]
            
            return {
                'categories': metrics,
                'values': values,
                'type': 'bar'
            }
        
        return {'error': '预处理数据格式不正确'}
    
    async def _handle_create_chart_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理创建图表任务"""
        try:
            chart_type = data.get('chart_type', 'line')
            chart_data = data.get('chart_data', {})
            chart_config = data.get('chart_config', {})
            
            # 验证图表类型
            if chart_type not in self.chart_templates:
                return {'error': f'不支持的图表类型: {chart_type}'}
            
            # 创建图表
            chart = {
                'id': f'chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'type': chart_type,
                'data': chart_data,
                'config': {**self.chart_templates[chart_type]['config'], **chart_config},
                'created_at': datetime.now().isoformat()
            }
            
            return {
                'status': 'completed',
                'task_type': 'create_chart',
                'chart': chart,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"图表创建失败: {e}")
            return {'error': str(e)}
    
    async def _handle_export_dashboard_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理导出仪表板任务"""
        try:
            dashboard_id = data.get('dashboard_id')
            export_format = data.get('format', 'html')
            
            if dashboard_id not in self.dashboard_cache:
                return {'error': '仪表板不存在'}
            
            dashboard = self.dashboard_cache[dashboard_id]
            
            # 导出仪表板
            if export_format == 'html':
                export_result = await self._export_dashboard_html(dashboard)
            elif export_format == 'pdf':
                export_result = await self._export_dashboard_pdf(dashboard)
            elif export_format == 'json':
                export_result = await self._export_dashboard_json(dashboard)
            else:
                return {'error': f'不支持的导出格式: {export_format}'}
            
            return {
                'status': 'completed',
                'task_type': 'export_dashboard',
                'dashboard_id': dashboard_id,
                'format': export_format,
                'export_result': export_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"仪表板导出失败: {e}")
            return {'error': str(e)}
    
    async def _export_dashboard_html(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """导出HTML格式的仪表板"""
        # 使用result_output模块生成HTML
        html_content = await self.result_output._generate_html_report(
            {'dashboard': dashboard}, 
            dashboard['title']
        )
        
        # 保存文件
        file_path = await self.result_output._save_as_html(
            {'dashboard': dashboard}, 
            f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return {
            'file_path': str(file_path) if file_path else None,
            'content_preview': html_content[:500] + '...' if len(html_content) > 500 else html_content
        }
    
    async def _export_dashboard_pdf(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """导出PDF格式的仪表板"""
        # PDF导出需要额外的依赖
        return {
            'error': 'PDF导出功能需要安装额外的依赖包',
            'suggestion': '请安装 weasyprint 或 reportlab'
        }
    
    async def _export_dashboard_json(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """导出JSON格式的仪表板"""
        file_path = await self.result_output._save_as_json(
            {'dashboard': dashboard},
            f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return {
            'file_path': str(file_path) if file_path else None,
            'data': dashboard
        }
    
    async def _handle_update_data_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据更新任务"""
        try:
            dashboard_id = data.get('dashboard_id')
            new_data = data.get('data', {})
            
            if dashboard_id not in self.dashboard_cache:
                return {'error': '仪表板不存在'}
            
            # 更新仪表板数据
            dashboard = self.dashboard_cache[dashboard_id]
            
            # 更新图表数据
            updated_charts = 0
            for chart in dashboard.get('charts', []):
                chart_id = chart['id']
                if chart_id in new_data:
                    chart['data'] = new_data[chart_id]
                    updated_charts += 1
            
            dashboard['metadata']['last_updated'] = datetime.now().isoformat()
            
            return {
                'status': 'completed',
                'task_type': 'update_data',
                'dashboard_id': dashboard_id,
                'updated_charts': updated_charts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"数据更新失败: {e}")
            return {'error': str(e)}
    
    async def _handle_get_dashboard_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取仪表板任务"""
        try:
            dashboard_id = data.get('dashboard_id')
            
            if dashboard_id not in self.dashboard_cache:
                return {'error': '仪表板不存在'}
            
            dashboard = self.dashboard_cache[dashboard_id]
            
            return {
                'status': 'completed',
                'task_type': 'get_dashboard',
                'dashboard_id': dashboard_id,
                'dashboard': dashboard,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取仪表板失败: {e}")
            return {'error': str(e)}
    
    async def get_dashboards(self) -> Dict[str, Any]:
        """获取所有仪表板列表"""
        dashboards_info = {}
        
        for dashboard_id, dashboard in self.dashboard_cache.items():
            dashboards_info[dashboard_id] = {
                'title': dashboard.get('title', '未命名仪表板'),
                'charts_count': len(dashboard.get('charts', [])),
                'created_at': dashboard.get('metadata', {}).get('created_at'),
                'last_updated': dashboard.get('metadata', {}).get('last_updated')
            }
        
        return {
            'total_dashboards': len(self.dashboard_cache),
            'dashboards': dashboards_info
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            return {
                'status': 'healthy' if self.is_running else 'unhealthy',
                'server_type': 'dashboard',
                'port': self.port,
                'uptime': uptime,
                'processed_requests': self.processed_requests,
                'cached_dashboards': len(self.dashboard_cache),
                'chart_templates': len(self.chart_templates),
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
            'server_type': 'dashboard',
            'port': self.port,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'processed_requests': self.processed_requests,
            'cached_dashboards': len(self.dashboard_cache),
            'chart_templates': list(self.chart_templates.keys()),
            'supported_tasks': [
                'generate',
                'create_chart',
                'export_dashboard',
                'update_data',
                'get_dashboard'
            ]
        }