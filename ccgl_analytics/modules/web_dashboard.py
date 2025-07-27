"""
Web仪表板模块

提供响应式Web界面生成器、实时数据展示组件。
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pathlib import Path

# Optional web framework imports
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

class WebDashboard:
    """Web仪表板生成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Web仪表板
        
        Args:
            config: Web配置信息
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.app = None
        self.data_store = {}
        
    def create_fastapi_dashboard(self, analysis_results: Dict[str, Any]) -> str:
        """
        创建FastAPI仪表板
        
        Args:
            analysis_results: 分析结果数据
            
        Returns:
            仪表板HTML内容
        """
        try:
            # 生成可视化组件
            visualizations = self._generate_visualizations(analysis_results)
            
            # 创建HTML仪表板
            html_content = self._create_dashboard_html(visualizations, analysis_results)
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"创建FastAPI仪表板失败: {e}")
            return self._create_error_html(str(e))
    
    def _generate_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """生成可视化图表"""
        visualizations = {}
        
        try:
            # 数据质量可视化
            quality_chart = self._create_data_quality_chart(analysis_results)
            if quality_chart:
                visualizations['data_quality'] = quality_chart
            
            # 聚类分析可视化
            clustering_chart = self._create_clustering_chart(analysis_results)
            if clustering_chart:
                visualizations['clustering'] = clustering_chart
            
            # 异常检测可视化
            anomaly_chart = self._create_anomaly_chart(analysis_results)
            if anomaly_chart:
                visualizations['anomaly'] = anomaly_chart
            
            # 关键指标仪表板
            metrics_chart = self._create_metrics_dashboard(analysis_results)
            if metrics_chart:
                visualizations['metrics'] = metrics_chart
            
        except Exception as e:
            self.logger.error(f"生成可视化失败: {e}")
        
        return visualizations
    
    def _create_data_quality_chart(self, analysis_results: Dict[str, Any]) -> Optional[str]:
        """创建数据质量图表"""
        try:
            quality_data = analysis_results.get('data_quality_assessment', {})
            quality_metrics = quality_data.get('data_quality_metrics', {})
            
            if not quality_metrics:
                return None
            
            # 数据质量指标图表
            metrics = ['完整性', '一致性', '有效性', '唯一性']
            values = [
                quality_metrics.get('completeness', 0) * 100,
                quality_metrics.get('consistency', 0) * 100,
                quality_metrics.get('validity', 0) * 100,
                quality_metrics.get('uniqueness', 0) * 100
            ]
            
            fig = go.Figure()
            
            # 添加雷达图
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='数据质量指标',
                line=dict(color='rgb(0, 123, 255)'),
                fillcolor='rgba(0, 123, 255, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="数据质量评估",
                font=dict(family="SimHei, Arial", size=12)
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id="data_quality_chart")
            
        except Exception as e:
            self.logger.error(f"创建数据质量图表失败: {e}")
            return None
    
    def _create_clustering_chart(self, analysis_results: Dict[str, Any]) -> Optional[str]:
        """创建聚类分析图表"""
        try:
            comp_analysis = analysis_results.get('comprehensive_analysis', {})
            clustering_data = comp_analysis.get('clustering_analysis', {})
            clustering_results = clustering_data.get('clustering_results', {})
            
            if not clustering_results:
                return None
            
            # 创建聚类结果对比图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('K-Means', 'DBSCAN', '层次聚类', '评估指标'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # 模拟聚类散点图数据
            import numpy as np
            np.random.seed(42)
            x = np.random.randn(100)
            y = np.random.randn(100)
            
            # K-means结果
            if 'kmeans' in clustering_results:
                kmeans_labels = clustering_results['kmeans'].get('labels', [])
                if len(kmeans_labels) >= 100:
                    fig.add_trace(
                        go.Scatter(x=x, y=y, mode='markers', 
                                 marker=dict(color=kmeans_labels[:100], colorscale='viridis'),
                                 name='K-Means'),
                        row=1, col=1
                    )
            
            # DBSCAN结果  
            if 'dbscan' in clustering_results:
                dbscan_labels = clustering_results['dbscan'].get('labels', [])
                if len(dbscan_labels) >= 100:
                    fig.add_trace(
                        go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(color=dbscan_labels[:100], colorscale='plasma'),
                                 name='DBSCAN'),
                        row=1, col=2
                    )
            
            # 层次聚类结果
            if 'hierarchical' in clustering_results:
                hier_labels = clustering_results['hierarchical'].get('labels', [])
                if len(hier_labels) >= 100:
                    fig.add_trace(
                        go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(color=hier_labels[:100], colorscale='inferno'),
                                 name='层次聚类'),
                        row=2, col=1
                    )
            
            # 评估指标
            evaluation_metrics = clustering_data.get('evaluation_metrics', {})
            if evaluation_metrics:
                methods = list(evaluation_metrics.keys())
                silhouette_scores = [evaluation_metrics[m].get('silhouette_score', 0) 
                                   for m in methods]
                
                fig.add_trace(
                    go.Bar(x=methods, y=silhouette_scores, name='轮廓系数',
                           marker_color='rgb(55, 83, 109)'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="聚类分析结果",
                showlegend=False,
                font=dict(family="SimHei, Arial", size=10)
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id="clustering_chart")
            
        except Exception as e:
            self.logger.error(f"创建聚类图表失败: {e}")
            return None
    
    def _create_anomaly_chart(self, analysis_results: Dict[str, Any]) -> Optional[str]:
        """创建异常检测图表"""
        try:
            comp_analysis = analysis_results.get('comprehensive_analysis', {})
            anomaly_data = comp_analysis.get('anomaly_detection', {})
            anomaly_results = anomaly_data.get('anomaly_results', {})
            
            if not anomaly_results:
                return None
            
            # 异常检测结果对比
            methods = list(anomaly_results.keys())
            anomaly_counts = []
            anomaly_ratios = []
            
            for method in methods:
                result = anomaly_results[method]
                if 'error' not in result:
                    anomaly_counts.append(result.get('anomaly_count', 0))
                    anomaly_ratios.append(result.get('anomaly_ratio', 0) * 100)
                else:
                    anomaly_counts.append(0)
                    anomaly_ratios.append(0)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('异常点数量', '异常点比例 (%)'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=methods, y=anomaly_counts, name='异常点数量',
                       marker_color='rgb(255, 0, 0)'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=methods, y=anomaly_ratios, name='异常点比例',
                       marker_color='rgb(255, 165, 0)'),
                row=1, col=2
            )
            
            fig.update_layout(
                title="异常检测分析",
                showlegend=False,
                font=dict(family="SimHei, Arial", size=12)
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id="anomaly_chart")
            
        except Exception as e:
            self.logger.error(f"创建异常检测图表失败: {e}")
            return None
    
    def _create_metrics_dashboard(self, analysis_results: Dict[str, Any]) -> Optional[str]:
        """创建关键指标仪表板"""
        try:
            # 提取关键指标
            quality_data = analysis_results.get('data_quality_assessment', {})
            basic_stats = quality_data.get('basic_stats', {})
            
            comp_analysis = analysis_results.get('comprehensive_analysis', {})
            summary = comp_analysis.get('summary', {})
            
            # 创建指标卡片
            metrics = [
                {
                    'title': '数据行数',
                    'value': basic_stats.get('total_rows', 0),
                    'color': 'blue'
                },
                {
                    'title': '数据列数', 
                    'value': basic_stats.get('total_columns', 0),
                    'color': 'green'
                },
                {
                    'title': '最优聚类数',
                    'value': summary.get('clustering', {}).get('optimal_clusters', 0),
                    'color': 'orange'
                },
                {
                    'title': '异常点数',
                    'value': summary.get('anomalies', {}).get('consensus_anomalies', 0),
                    'color': 'red'
                }
            ]
            
            # 创建指标展示图表
            fig = go.Figure()
            
            titles = [m['title'] for m in metrics]
            values = [m['value'] for m in metrics]
            colors = [m['color'] for m in metrics]
            
            fig.add_trace(go.Bar(
                x=titles,
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto'
            ))
            
            fig.update_layout(
                title="关键指标概览",
                xaxis_title="指标类型",
                yaxis_title="数值",
                font=dict(family="SimHei, Arial", size=12)
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id="metrics_chart")
            
        except Exception as e:
            self.logger.error(f"创建指标仪表板失败: {e}")
            return None
    
    def _create_dashboard_html(self, visualizations: Dict[str, str], 
                             analysis_results: Dict[str, Any]) -> str:
        """创建完整的仪表板HTML"""
        
        # 提取摘要信息
        quality_data = analysis_results.get('data_quality_assessment', {})
        comp_analysis = analysis_results.get('comprehensive_analysis', {})
        
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCGL仓储管理系统数据分析仪表板</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'SimHei', 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .card h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .chart-container {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .chart-title {{
            background: #f8f9fa;
            padding: 15px 20px;
            font-size: 1.2em;
            font-weight: bold;
            border-bottom: 1px solid #e9ecef;
        }}
        .chart-content {{
            padding: 20px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #e9ecef;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏭 CCGL仓储管理系统数据分析仪表板</h1>
        <p>企业级智能数据分析平台 - 实时洞察您的业务数据</p>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary-cards">
        <div class="card">
            <h3>📊 数据总量</h3>
            <div class="value">{quality_data.get('basic_stats', {}).get('total_rows', 0):,}</div>
            <p>条记录</p>
        </div>
        <div class="card">
            <h3>📈 数据完整性</h3>
            <div class="value">{quality_data.get('data_quality_metrics', {}).get('completeness', 0)*100:.1f}%</div>
            <p>完整度</p>
        </div>
        <div class="card">
            <h3>🎯 聚类群体</h3>
            <div class="value">{comp_analysis.get('clustering_analysis', {}).get('optimal_clusters', 0)}</div>
            <p>个群体</p>
        </div>
        <div class="card">
            <h3>⚠️ 异常点</h3>
            <div class="value">{comp_analysis.get('anomaly_detection', {}).get('summary', {}).get('consensus_anomalies', 0)}</div>
            <p>个异常</p>
        </div>
    </div>
    
    {self._insert_visualizations(visualizations)}
    
    <div class="footer">
        <p>CCGL仓储管理系统数据分析工程 &copy; 2024 | 基于机器学习的智能分析平台</p>
        <p>💡 提示: 本仪表板提供实时数据分析和可视化展示</p>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _insert_visualizations(self, visualizations: Dict[str, str]) -> str:
        """插入可视化组件到HTML中"""
        html_sections = []
        
        chart_titles = {
            'data_quality': '📋 数据质量评估',
            'clustering': '🎯 聚类分析结果',
            'anomaly': '🚨 异常检测分析',
            'metrics': '📊 关键指标概览'
        }
        
        for chart_id, chart_html in visualizations.items():
            title = chart_titles.get(chart_id, chart_id.title())
            section_html = f"""
    <div class="chart-container">
        <div class="chart-title">{title}</div>
        <div class="chart-content">
            {chart_html}
        </div>
    </div>
            """
            html_sections.append(section_html)
        
        return '\\n'.join(html_sections)
    
    def _create_error_html(self, error_message: str) -> str:
        """创建错误页面HTML"""
        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCGL仪表板 - 错误</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 50px;
            text-align: center;
        }}
        .error {{
            color: #d32f2f;
            background: #ffebee;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>🚨 仪表板生成错误</h1>
    <div class="error">
        <p><strong>错误信息:</strong> {error_message}</p>
    </div>
    <p>请检查数据格式或联系系统管理员。</p>
</body>
</html>
        """
    
    def save_dashboard(self, html_content: str, filename: str = None) -> str:
        """
        保存仪表板到文件
        
        Args:
            html_content: HTML内容
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ccgl_dashboard_{timestamp}.html"
            
            # 确保输出目录存在
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            file_path = output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"仪表板已保存到: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"保存仪表板失败: {e}")
            raise
    
    def create_streamlit_dashboard(self, analysis_results: Dict[str, Any]):
        """创建Streamlit仪表板（如果可用）"""
        if not STREAMLIT_AVAILABLE:
            self.logger.warning("Streamlit不可用")
            return
        
        # Streamlit仪表板实现
        st.title("🏭 CCGL仓储管理系统数据分析仪表板")
        st.markdown("企业级智能数据分析平台")
        
        # 显示关键指标
        quality_data = analysis_results.get('data_quality_assessment', {})
        basic_stats = quality_data.get('basic_stats', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("数据总量", f"{basic_stats.get('total_rows', 0):,}")
        
        with col2:
            completeness = quality_data.get('data_quality_metrics', {}).get('completeness', 0)
            st.metric("数据完整性", f"{completeness*100:.1f}%")
        
        with col3:
            optimal_clusters = analysis_results.get('comprehensive_analysis', {}).get('clustering_analysis', {}).get('optimal_clusters', 0)
            st.metric("聚类群体", optimal_clusters)
        
        with col4:
            anomalies = analysis_results.get('comprehensive_analysis', {}).get('anomaly_detection', {}).get('summary', {}).get('consensus_anomalies', 0)
            st.metric("异常点", anomalies)
    
    def start_dashboard_server(self, analysis_results: Dict[str, Any], 
                             host: str = "0.0.0.0", port: int = 8000):
        """启动仪表板服务器"""
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI不可用，无法启动服务器")
            return
        
        # 创建仪表板HTML
        dashboard_html = self.create_fastapi_dashboard(analysis_results)
        
        # 保存到文件
        dashboard_file = self.save_dashboard(dashboard_html)
        
        self.logger.info(f"仪表板已生成: {dashboard_file}")
        self.logger.info(f"可以直接打开文件查看，或运行Web服务器")
        
        return dashboard_file

def main():
    """主函数 - 用于测试"""
    # 模拟分析结果
    sample_results = {
        'data_quality_assessment': {
            'basic_stats': {'total_rows': 1000, 'total_columns': 10},
            'data_quality_metrics': {'completeness': 0.95}
        },
        'comprehensive_analysis': {
            'clustering_analysis': {'optimal_clusters': 4},
            'anomaly_detection': {'summary': {'consensus_anomalies': 15}}
        }
    }
    
    dashboard = WebDashboard()
    html_content = dashboard.create_fastapi_dashboard(sample_results)
    dashboard_file = dashboard.save_dashboard(html_content)
    
    print(f"测试仪表板已生成: {dashboard_file}")

if __name__ == "__main__":
    main()