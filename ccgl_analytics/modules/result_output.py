"""
结果输出模块

处理分析结果的存储、格式化和导出
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import pickle


class ResultOutput:
    """结果输出类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化结果输出器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 输出路径配置
        self.data_path = Path(config.get('data_path', 'data/'))
        self.model_path = Path(config.get('model_path', 'models/'))
        self.cache_path = Path(config.get('cache_path', 'cache/'))
        
        # 创建输出目录
        self._create_directories()
        
        # 输出格式配置
        self.supported_formats = ['json', 'csv', 'excel', 'html', 'pdf']
        self.default_format = 'json'
        
        self.logger.info("结果输出器初始化完成")
    
    def _create_directories(self):
        """创建输出目录"""
        directories = [self.data_path, self.model_path, self.cache_path]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"创建目录: {directory}")
    
    async def save_results(self, results: Dict[str, Any], 
                          output_name: Optional[str] = None,
                          formats: Optional[List[str]] = None) -> Dict[str, str]:
        """保存分析结果"""
        try:
            if output_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"analysis_results_{timestamp}"
            
            formats = formats or [self.default_format]
            saved_files = {}
            
            for format_type in formats:
                if format_type in self.supported_formats:
                    file_path = await self._save_in_format(results, output_name, format_type)
                    if file_path:
                        saved_files[format_type] = str(file_path)
                else:
                    self.logger.warning(f"不支持的输出格式: {format_type}")
            
            # 保存元数据
            metadata_path = await self._save_metadata(results, output_name)
            if metadata_path:
                saved_files['metadata'] = str(metadata_path)
            
            self.logger.info(f"结果保存完成: {len(saved_files)} 个文件")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"结果保存失败: {e}")
            raise
    
    async def _save_in_format(self, results: Dict[str, Any], 
                            output_name: str, format_type: str) -> Optional[Path]:
        """以指定格式保存结果"""
        try:
            if format_type == 'json':
                return await self._save_as_json(results, output_name)
            elif format_type == 'csv':
                return await self._save_as_csv(results, output_name)
            elif format_type == 'excel':
                return await self._save_as_excel(results, output_name)
            elif format_type == 'html':
                return await self._save_as_html(results, output_name)
            elif format_type == 'pdf':
                return await self._save_as_pdf(results, output_name)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"保存 {format_type} 格式失败: {e}")
            return None
    
    async def _save_as_json(self, results: Dict[str, Any], output_name: str) -> Path:
        """保存为JSON格式"""
        file_path = self.data_path / f"{output_name}.json"
        
        # 序列化处理
        serializable_results = self._make_serializable(results)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        self.logger.debug(f"JSON文件已保存: {file_path}")
        return file_path
    
    async def _save_as_csv(self, results: Dict[str, Any], output_name: str) -> Path:
        """保存为CSV格式"""
        # 对于CSV，我们需要提取表格数据
        csv_dir = self.data_path / f"{output_name}_csv"
        csv_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                csv_path = csv_dir / f"{key}.csv"
                value.to_csv(csv_path, index=False, encoding='utf-8')
                saved_files.append(csv_path)
            elif isinstance(value, dict) and 'data' in value:
                if isinstance(value['data'], list) and value['data']:
                    df = pd.DataFrame(value['data'])
                    csv_path = csv_dir / f"{key}.csv"
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    saved_files.append(csv_path)
        
        # 创建汇总文件
        summary_path = csv_dir / "summary.csv"
        summary_data = []
        
        for key, value in results.items():
            if isinstance(value, dict):
                row = {'component': key}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str)):
                        row[sub_key] = sub_value
                summary_data.append(row)
        
        if summary_data:
            pd.DataFrame(summary_data).to_csv(summary_path, index=False)
            saved_files.append(summary_path)
        
        self.logger.debug(f"CSV文件已保存: {len(saved_files)} 个文件")
        return csv_dir
    
    async def _save_as_excel(self, results: Dict[str, Any], output_name: str) -> Path:
        """保存为Excel格式"""
        file_path = self.data_path / f"{output_name}.xlsx"
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 写入汇总信息
                summary_data = self._create_summary_data(results)
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # 写入各个分析结果
                for key, value in results.items():
                    sheet_name = key[:31]  # Excel工作表名称限制
                    
                    if isinstance(value, pd.DataFrame):
                        value.to_excel(writer, sheet_name=sheet_name, index=False)
                    elif isinstance(value, dict):
                        if 'data' in value and isinstance(value['data'], list):
                            df = pd.DataFrame(value['data'])
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                        else:
                            # 将字典转换为DataFrame
                            df = self._dict_to_dataframe(value)
                            if not df.empty:
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.debug(f"Excel文件已保存: {file_path}")
            return file_path
            
        except ImportError:
            self.logger.warning("openpyxl未安装，无法保存Excel格式")
            return None
        except Exception as e:
            self.logger.error(f"Excel保存失败: {e}")
            return None
    
    async def _save_as_html(self, results: Dict[str, Any], output_name: str) -> Path:
        """保存为HTML格式"""
        file_path = self.data_path / f"{output_name}.html"
        
        html_content = self._generate_html_report(results, output_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.debug(f"HTML文件已保存: {file_path}")
        return file_path
    
    async def _save_as_pdf(self, results: Dict[str, Any], output_name: str) -> Optional[Path]:
        """保存为PDF格式"""
        try:
            # 首先生成HTML
            html_path = await self._save_as_html(results, output_name)
            
            # 然后转换为PDF（需要安装wkhtmltopdf或使用其他PDF库）
            pdf_path = self.data_path / f"{output_name}.pdf"
            
            # 这里使用模拟的PDF生成
            # 实际使用时可以使用 weasyprint 或 reportlab
            self.logger.warning("PDF生成功能需要额外的依赖库")
            
            return None
            
        except Exception as e:
            self.logger.error(f"PDF保存失败: {e}")
            return None
    
    def _make_serializable(self, obj: Any) -> Any:
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return {
                'type': 'DataFrame',
                'data': obj.to_dict('records'),
                'columns': obj.columns.tolist(),
                'shape': obj.shape
            }
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # 自定义对象
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def _create_summary_data(self, results: Dict[str, Any]) -> pd.DataFrame:
        """创建汇总数据"""
        summary_rows = []
        
        for key, value in results.items():
            row = {
                'Analysis_Type': key,
                'Status': 'Completed',
                'Timestamp': datetime.now().isoformat()
            }
            
            if isinstance(value, dict):
                # 提取关键指标
                if 'n_clusters' in value:
                    row['Clusters'] = value['n_clusters']
                if 'n_anomalies' in value:
                    row['Anomalies'] = value['n_anomalies']
                if 'n_rules' in value:
                    row['Rules'] = value['n_rules']
                if 'total_records' in value:
                    row['Records'] = value['total_records']
                
                # 添加评估指标
                if 'silhouette_score' in value:
                    row['Quality_Score'] = value['silhouette_score']
            
            summary_rows.append(row)
        
        return pd.DataFrame(summary_rows)
    
    def _dict_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """将字典转换为DataFrame"""
        try:
            flat_data = []
            
            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            
            flattened = flatten_dict(data)
            
            # 创建单行DataFrame
            return pd.DataFrame([flattened])
            
        except Exception:
            return pd.DataFrame()
    
    def _generate_html_report(self, results: Dict[str, Any], title: str) -> str:
        """生成HTML报告"""
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCGL 分析报告 - {title}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metric {{
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .metric-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #27ae60;
            margin-left: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏪 CCGL 仓储管理分析报告</h1>
        <h2>📊 报告名称: {title}</h2>
        
        {self._generate_html_sections(results)}
        
        <div class="timestamp">
            报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_html_sections(self, results: Dict[str, Any]) -> str:
        """生成HTML报告的各个部分"""
        sections = []
        
        for key, value in results.items():
            section_title = self._get_section_title(key)
            section_html = f"<h2>{section_title}</h2>\n"
            
            if isinstance(value, dict):
                section_html += self._dict_to_html(value)
            elif isinstance(value, pd.DataFrame):
                section_html += self._dataframe_to_html(value)
            else:
                section_html += f"<p>{str(value)}</p>\n"
            
            sections.append(section_html)
        
        return "\n".join(sections)
    
    def _get_section_title(self, key: str) -> str:
        """获取章节标题"""
        title_mapping = {
            'summary': '📈 数据概览',
            'clustering': '🎯 聚类分析',
            'anomaly': '⚠️ 异常检测',
            'association': '🔗 关联规则',
            'relationships': '🌐 关系分析',
            'preprocessing': '🔧 数据预处理'
        }
        
        return title_mapping.get(key, f"📊 {key.upper()}")
    
    def _dict_to_html(self, data: Dict[str, Any]) -> str:
        """将字典转换为HTML"""
        html_parts = []
        
        for key, value in data.items():
            if isinstance(value, (int, float, str)):
                html_parts.append(f"""
                <div class="metric">
                    <span class="metric-name">{key}:</span>
                    <span class="metric-value">{value}</span>
                </div>
                """)
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    df = pd.DataFrame(value)
                    html_parts.append(self._dataframe_to_html(df))
                else:
                    html_parts.append(f"<p><strong>{key}:</strong> {', '.join(map(str, value))}</p>")
        
        return "\n".join(html_parts)
    
    def _dataframe_to_html(self, df: pd.DataFrame) -> str:
        """将DataFrame转换为HTML表格"""
        if df.empty:
            return "<p>暂无数据</p>"
        
        # 限制显示的行数
        display_df = df.head(20) if len(df) > 20 else df
        
        html_table = display_df.to_html(
            classes='table table-striped',
            table_id='data-table',
            escape=False,
            index=False
        )
        
        if len(df) > 20:
            html_table += f"<p><em>显示前20行，共{len(df)}行数据</em></p>"
        
        return html_table
    
    async def _save_metadata(self, results: Dict[str, Any], output_name: str) -> Path:
        """保存元数据"""
        metadata = {
            'output_name': output_name,
            'timestamp': datetime.now().isoformat(),
            'analysis_types': list(results.keys()),
            'total_components': len(results),
            'config': self.config
        }
        
        metadata_path = self.data_path / f"{output_name}_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return metadata_path
    
    async def load_results(self, output_name: str) -> Optional[Dict[str, Any]]:
        """加载保存的结果"""
        try:
            json_path = self.data_path / f"{output_name}.json"
            
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                self.logger.info(f"结果加载成功: {output_name}")
                return results
            else:
                self.logger.warning(f"结果文件不存在: {json_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"结果加载失败: {e}")
            return None
    
    async def list_saved_results(self) -> List[Dict[str, Any]]:
        """列出所有保存的结果"""
        try:
            results_list = []
            
            for json_file in self.data_path.glob("*.json"):
                if not json_file.name.endswith("_metadata.json"):
                    metadata_file = json_file.with_suffix("").with_suffix("") / "_metadata.json"
                    
                    result_info = {
                        'name': json_file.stem,
                        'file_path': str(json_file),
                        'size': json_file.stat().st_size,
                        'created_time': datetime.fromtimestamp(json_file.stat().st_ctime).isoformat()
                    }
                    
                    # 如果有元数据文件，加载其中的信息
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            result_info.update(metadata)
                        except:
                            pass
                    
                    results_list.append(result_info)
            
            # 按创建时间排序
            results_list.sort(key=lambda x: x['created_time'], reverse=True)
            
            return results_list
            
        except Exception as e:
            self.logger.error(f"结果列表获取失败: {e}")
            return []