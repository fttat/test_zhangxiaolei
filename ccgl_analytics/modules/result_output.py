"""
CCGL Analytics - Result Output Module
Format and output analysis results in various formats
"""

import json
import csv
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import pandas as pd

from ..utils.logger import get_logger, LoggerMixin

class ResultOutput(LoggerMixin):
    """Handle output of analysis results in various formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize result output handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.output_config = self.config.get('storage', {})
        self.default_output_path = self.output_config.get('report_output_path', 'reports/')
        self.supported_formats = ['json', 'csv', 'html', 'txt', 'xlsx']
        
        # Create output directory
        Path(self.default_output_path).mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: Dict[str, Any], 
                    output_format: str = 'json',
                    filename: Optional[str] = None,
                    output_path: Optional[str] = None) -> Dict[str, Any]:
        """Save analysis results to file.
        
        Args:
            results: Analysis results to save
            output_format: Output format ('json', 'csv', 'html', 'txt', 'xlsx')
            filename: Custom filename (without extension)
            output_path: Custom output path
            
        Returns:
            Save operation result
        """
        try:
            if output_format not in self.supported_formats:
                return {
                    'status': 'error',
                    'error': f'Unsupported format: {output_format}. Supported: {self.supported_formats}'
                }
            
            # Determine output path and filename
            if output_path is None:
                output_path = self.default_output_path
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'ccgl_analysis_{timestamp}'
            
            # Ensure output directory exists
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Generate full file path
            file_extension = output_format if output_format != 'xlsx' else 'xlsx'
            full_path = os.path.join(output_path, f'{filename}.{file_extension}')
            
            # Save based on format
            if output_format == 'json':
                save_result = self._save_json(results, full_path)
            elif output_format == 'csv':
                save_result = self._save_csv(results, full_path)
            elif output_format == 'html':
                save_result = self._save_html(results, full_path)
            elif output_format == 'txt':
                save_result = self._save_txt(results, full_path)
            elif output_format == 'xlsx':
                save_result = self._save_xlsx(results, full_path)
            else:
                save_result = {
                    'status': 'error',
                    'error': f'Format handler not implemented: {output_format}'
                }
            
            if save_result['status'] == 'success':
                self.logger.info(f"Results saved to: {full_path}")
            
            return save_result
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_json(self, results: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Save results as JSON.
        
        Args:
            results: Results to save
            file_path: Output file path
            
        Returns:
            Save operation result
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'format': 'json',
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _save_csv(self, results: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Save results as CSV.
        
        Args:
            results: Results to save
            file_path: Output file path
            
        Returns:
            Save operation result
        """
        try:
            # Convert results to tabular format
            tabular_data = self._convert_to_tabular(results)
            
            if not tabular_data:
                return {
                    'status': 'error',
                    'error': 'No tabular data found in results'
                }
            
            # Create DataFrame and save
            df = pd.DataFrame(tabular_data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            return {
                'status': 'success',
                'file_path': file_path,
                'format': 'csv',
                'rows': len(df),
                'columns': len(df.columns),
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _save_html(self, results: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Save results as HTML report.
        
        Args:
            results: Results to save
            file_path: Output file path
            
        Returns:
            Save operation result
        """
        try:
            html_content = self._generate_html_report(results)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'format': 'html',
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _save_txt(self, results: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Save results as plain text.
        
        Args:
            results: Results to save
            file_path: Output file path
            
        Returns:
            Save operation result
        """
        try:
            text_content = self._generate_text_report(results)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'format': 'txt',
                'lines': len(text_content.split('\n')),
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _save_xlsx(self, results: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Save results as Excel file.
        
        Args:
            results: Results to save
            file_path: Output file path
            
        Returns:
            Save operation result
        """
        try:
            # Convert results to multiple sheets
            sheets_data = self._convert_to_excel_sheets(results)
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, sheet_data in sheets_data.items():
                    if isinstance(sheet_data, list) and sheet_data:
                        df = pd.DataFrame(sheet_data)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    elif isinstance(sheet_data, dict):
                        # Convert dict to DataFrame
                        df = pd.DataFrame([sheet_data])
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'format': 'xlsx',
                'sheets': len(sheets_data),
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _convert_to_tabular(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert results to tabular format.
        
        Args:
            results: Results to convert
            
        Returns:
            List of records for tabular output
        """
        tabular_data = []
        
        def flatten_dict(d, parent_key='', sep='_'):
            """Flatten nested dictionary."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Convert list to string representation
                    items.append((new_key, str(v)))
                else:
                    items.append((new_key, v))
            return dict(items)
        
        # Handle different result structures
        if 'results' in results and isinstance(results['results'], dict):
            # Multiple analysis results
            for analysis_type, analysis_result in results['results'].items():
                if isinstance(analysis_result, dict):
                    flattened = flatten_dict(analysis_result)
                    flattened['analysis_type'] = analysis_type
                    tabular_data.append(flattened)
        else:
            # Single result
            flattened = flatten_dict(results)
            tabular_data.append(flattened)
        
        return tabular_data
    
    def _convert_to_excel_sheets(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to Excel sheets format.
        
        Args:
            results: Results to convert
            
        Returns:
            Dictionary with sheet names and data
        """
        sheets = {}
        
        # Summary sheet
        summary_data = {
            'Analysis_Type': [],
            'Status': [],
            'Key_Metric': [],
            'Value': []
        }
        
        if 'results' in results:
            for analysis_type, analysis_result in results['results'].items():
                if isinstance(analysis_result, dict):
                    summary_data['Analysis_Type'].append(analysis_type)
                    summary_data['Status'].append(analysis_result.get('status', 'unknown'))
                    
                    # Extract key metric
                    if 'n_clusters' in analysis_result:
                        summary_data['Key_Metric'].append('clusters')
                        summary_data['Value'].append(analysis_result['n_clusters'])
                    elif 'anomaly_count' in analysis_result:
                        summary_data['Key_Metric'].append('anomalies')
                        summary_data['Value'].append(analysis_result['anomaly_count'])
                    else:
                        summary_data['Key_Metric'].append('processed')
                        summary_data['Value'].append('yes')
        
        sheets['Summary'] = summary_data
        
        # Individual analysis sheets
        if 'results' in results:
            for analysis_type, analysis_result in results['results'].items():
                if isinstance(analysis_result, dict) and 'error' not in analysis_result:
                    # Create sheet for this analysis
                    sheet_data = self._extract_sheet_data(analysis_result)
                    if sheet_data:
                        sheets[analysis_type.title()] = sheet_data
        
        return sheets
    
    def _extract_sheet_data(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data suitable for Excel sheet.
        
        Args:
            analysis_result: Analysis result data
            
        Returns:
            List of records for sheet
        """
        sheet_data = []
        
        # Handle clustering results
        if 'cluster_labels' in analysis_result:
            for i, label in enumerate(analysis_result['cluster_labels']):
                sheet_data.append({
                    'Data_Point': i + 1,
                    'Cluster': label
                })
        
        # Handle anomaly detection results
        elif 'anomaly_labels' in analysis_result:
            for i, is_anomaly in enumerate(analysis_result['anomaly_labels']):
                sheet_data.append({
                    'Data_Point': i + 1,
                    'Is_Anomaly': is_anomaly,
                    'Anomaly_Score': analysis_result.get('anomaly_scores', [0])[i] if i < len(analysis_result.get('anomaly_scores', [])) else 0
                })
        
        # Handle general metrics
        else:
            metrics = {}
            for key, value in analysis_result.items():
                if not isinstance(value, (dict, list)):
                    metrics[key] = value
            
            if metrics:
                sheet_data.append(metrics)
        
        return sheet_data
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report.
        
        Args:
            results: Results to format
            
        Returns:
            HTML content
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCGL Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .analysis-section {{ margin: 20px 0; padding: 15px; border: 1px solid #bdc3c7; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #3498db; color: white; border-radius: 5px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 CCGL Analytics Report</h1>
        <div class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="summary">
            <h2>📊 Analysis Summary</h2>
            <div class="metric">Input Shape: {results.get('input_shape', 'N/A')}</div>
            <div class="metric">Total Processing Time: {results.get('total_processing_time', 'N/A')} seconds</div>
        </div>
        
        {self._generate_html_analysis_sections(results)}
        
        <div class="summary">
            <h2>💡 Key Insights</h2>
            {self._generate_html_insights(results)}
        </div>
    </div>
</body>
</html>
"""
        return html_content
    
    def _generate_html_analysis_sections(self, results: Dict[str, Any]) -> str:
        """Generate HTML sections for each analysis.
        
        Args:
            results: Results data
            
        Returns:
            HTML content for analysis sections
        """
        sections_html = ""
        
        if 'results' in results:
            for analysis_type, analysis_result in results['results'].items():
                if 'error' not in analysis_result:
                    sections_html += f"""
        <div class="analysis-section">
            <h2>📈 {analysis_type.replace('_', ' ').title()}</h2>
            {self._format_analysis_html(analysis_result)}
        </div>
"""
        
        return sections_html
    
    def _format_analysis_html(self, analysis_result: Dict[str, Any]) -> str:
        """Format individual analysis result as HTML.
        
        Args:
            analysis_result: Analysis result data
            
        Returns:
            HTML content for the analysis
        """
        html = "<table><tr><th>Metric</th><th>Value</th></tr>"
        
        for key, value in analysis_result.items():
            if not isinstance(value, (dict, list)):
                html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_html_insights(self, results: Dict[str, Any]) -> str:
        """Generate HTML insights section.
        
        Args:
            results: Results data
            
        Returns:
            HTML content for insights
        """
        insights = []
        
        if 'summary' in results and 'key_findings' in results['summary']:
            for finding in results['summary']['key_findings']:
                insights.append(f"<li>{finding}</li>")
        
        if not insights:
            insights.append("<li>Analysis completed successfully</li>")
        
        return f"<ul>{''.join(insights)}</ul>"
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate plain text report.
        
        Args:
            results: Results to format
            
        Returns:
            Text content
        """
        lines = [
            "CCGL Analytics Report",
            "=" * 50,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY:",
            f"Input Shape: {results.get('input_shape', 'N/A')}",
            f"Processing Time: {results.get('total_processing_time', 'N/A')} seconds",
            ""
        ]
        
        if 'results' in results:
            lines.append("ANALYSIS RESULTS:")
            lines.append("-" * 30)
            
            for analysis_type, analysis_result in results['results'].items():
                lines.append(f"\n{analysis_type.replace('_', ' ').title()}:")
                
                if 'error' in analysis_result:
                    lines.append(f"  Error: {analysis_result['error']}")
                else:
                    for key, value in analysis_result.items():
                        if not isinstance(value, (dict, list)):
                            lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        if 'summary' in results and 'key_findings' in results['summary']:
            lines.append("\nKEY FINDINGS:")
            lines.append("-" * 20)
            for finding in results['summary']['key_findings']:
                lines.append(f"• {finding}")
        
        return "\n".join(lines)
    
    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary report of analysis results.
        
        Args:
            results: Analysis results
            
        Returns:
            Summary report
        """
        try:
            summary = {
                'report_id': f"report_{int(datetime.now().timestamp())}",
                'generated_at': datetime.now().isoformat(),
                'analysis_overview': {},
                'key_metrics': {},
                'recommendations': []
            }
            
            # Analyze results structure
            if 'results' in results:
                analysis_count = 0
                successful_analyses = 0
                
                for analysis_type, analysis_result in results['results'].items():
                    analysis_count += 1
                    
                    if analysis_result.get('status') != 'error':
                        successful_analyses += 1
                        
                        # Extract key metrics
                        if 'n_clusters' in analysis_result:
                            summary['key_metrics'][f'{analysis_type}_clusters'] = analysis_result['n_clusters']
                        
                        if 'anomaly_count' in analysis_result:
                            summary['key_metrics'][f'{analysis_type}_anomalies'] = analysis_result['anomaly_count']
                
                summary['analysis_overview'] = {
                    'total_analyses': analysis_count,
                    'successful_analyses': successful_analyses,
                    'success_rate': (successful_analyses / analysis_count * 100) if analysis_count > 0 else 0
                }
            
            # Generate recommendations
            if summary['analysis_overview'].get('success_rate', 0) > 80:
                summary['recommendations'].append("Analysis pipeline is performing well")
            
            if 'clustering_clusters' in summary['key_metrics']:
                clusters = summary['key_metrics']['clustering_clusters']
                if clusters > 1:
                    summary['recommendations'].append(f"Consider {clusters} distinct customer segments for targeted strategies")
            
            return {
                'status': 'success',
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }