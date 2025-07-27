"""
数据预处理模块

提供智能数据清洗、缺失值处理、异常检测和数据标准化功能。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """智能数据预处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据预处理器
        
        Args:
            config: 预处理配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.imputers = {}
        self.outlier_detectors = {}
        
    def clean_data(self, df: pd.DataFrame, 
                  missing_strategy: str = 'auto',
                  outlier_method: str = 'iqr',
                  scaling_method: str = 'standard') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        综合数据清洗流程
        
        Args:
            df: 输入数据
            missing_strategy: 缺失值处理策略
            outlier_method: 异常值检测方法
            scaling_method: 数据标准化方法
            
        Returns:
            清洗后的数据和处理报告
        """
        self.logger.info("开始数据清洗流程")
        
        report = {
            'original_shape': df.shape,
            'processing_steps': [],
            'removed_rows': 0,
            'processed_columns': []
        }
        
        # 1. 处理缺失值
        df_clean, missing_report = self.handle_missing_values(df, strategy=missing_strategy)
        report['processing_steps'].append({
            'step': 'missing_values',
            'details': missing_report
        })
        
        # 2. 检测和处理异常值
        df_clean, outlier_report = self.detect_and_handle_outliers(
            df_clean, method=outlier_method
        )
        report['processing_steps'].append({
            'step': 'outliers',
            'details': outlier_report
        })
        report['removed_rows'] = len(df) - len(df_clean)
        
        # 3. 数据标准化
        df_clean, scaling_report = self.scale_features(df_clean, method=scaling_method)
        report['processing_steps'].append({
            'step': 'scaling',
            'details': scaling_report
        })
        
        report['final_shape'] = df_clean.shape
        report['processing_time'] = datetime.now().isoformat()
        
        self.logger.info(f"数据清洗完成: {df.shape} -> {df_clean.shape}")
        return df_clean, report
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'auto') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        智能缺失值处理
        
        Args:
            df: 输入数据
            strategy: 处理策略 ('auto', 'mean', 'median', 'mode', 'knn', 'drop')
            
        Returns:
            处理后的数据和处理报告
        """
        df_processed = df.copy()
        missing_info = df.isnull().sum()
        
        report = {
            'original_missing_count': int(missing_info.sum()),
            'columns_with_missing': missing_info[missing_info > 0].to_dict(),
            'strategies_used': {}
        }
        
        # 按列处理缺失值
        for column in df.columns:
            if df[column].isnull().any():
                missing_ratio = df[column].isnull().sum() / len(df)
                
                # 自动选择策略
                if strategy == 'auto':
                    chosen_strategy = self._choose_missing_strategy(df[column], missing_ratio)
                else:
                    chosen_strategy = strategy
                
                # 应用处理策略
                df_processed[column] = self._apply_missing_strategy(
                    df[column], chosen_strategy
                )
                
                report['strategies_used'][column] = {
                    'strategy': chosen_strategy,
                    'missing_count': int(missing_info[column]),
                    'missing_ratio': float(missing_ratio)
                }
        
        report['final_missing_count'] = int(df_processed.isnull().sum().sum())
        return df_processed, report
    
    def _choose_missing_strategy(self, series: pd.Series, missing_ratio: float) -> str:
        """智能选择缺失值处理策略"""
        if missing_ratio > 0.5:
            return 'drop'
        elif series.dtype == 'object':
            return 'mode'
        elif missing_ratio < 0.1:
            return 'mean'
        elif missing_ratio < 0.3:
            return 'median'
        else:
            return 'knn'
    
    def _apply_missing_strategy(self, series: pd.Series, strategy: str) -> pd.Series:
        """应用具体的缺失值处理策略"""
        if strategy == 'drop':
            return series.dropna()
        elif strategy == 'mean' and series.dtype != 'object':
            return series.fillna(series.mean())
        elif strategy == 'median' and series.dtype != 'object':
            return series.fillna(series.median())
        elif strategy == 'mode':
            mode_value = series.mode()
            if len(mode_value) > 0:
                return series.fillna(mode_value[0])
            else:
                return series.fillna('Unknown')
        elif strategy == 'knn' and series.dtype != 'object':
            # 简化的KNN填充（这里使用前后值的平均）
            return series.interpolate(method='linear')
        else:
            # 默认策略
            if series.dtype == 'object':
                return series.fillna('Unknown')
            else:
                return series.fillna(series.median())
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, 
                                  method: str = 'iqr',
                                  action: str = 'remove') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        检测和处理异常值
        
        Args:
            df: 输入数据
            method: 检测方法 ('iqr', 'zscore', 'isolation_forest', 'lof')
            action: 处理动作 ('remove', 'cap', 'transform')
            
        Returns:
            处理后的数据和处理报告
        """
        df_processed = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        report = {
            'method': method,
            'action': action,
            'outliers_detected': {},
            'total_outliers': 0
        }
        
        outlier_indices = set()
        
        for column in numeric_columns:
            if method == 'iqr':
                outliers = self._detect_outliers_iqr(df[column])
            elif method == 'zscore':
                outliers = self._detect_outliers_zscore(df[column])
            elif method == 'isolation_forest':
                outliers = self._detect_outliers_isolation_forest(df[[column]])
            elif method == 'lof':
                outliers = self._detect_outliers_lof(df[[column]])
            else:
                outliers = []
            
            if len(outliers) > 0:
                outlier_indices.update(outliers)
                report['outliers_detected'][column] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100
                }
                
                # 应用处理动作
                if action == 'remove':
                    continue  # 在最后统一移除
                elif action == 'cap':
                    df_processed = self._cap_outliers(df_processed, column, outliers)
                elif action == 'transform':
                    df_processed[column] = self._transform_outliers(df_processed[column])
        
        # 移除异常行
        if action == 'remove' and outlier_indices:
            df_processed = df_processed.drop(index=list(outlier_indices)).reset_index(drop=True)
        
        report['total_outliers'] = len(outlier_indices)
        report['rows_removed'] = len(outlier_indices) if action == 'remove' else 0
        
        return df_processed, report
    
    def _detect_outliers_iqr(self, series: pd.Series, factor: float = 1.5) -> List[int]:
        """使用IQR方法检测异常值"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outliers
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """使用Z-Score方法检测异常值"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = series[z_scores > threshold].index.tolist()
        return outliers
    
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame, 
                                         contamination: float = 0.1) -> List[int]:
        """使用孤立森林检测异常值"""
        try:
            clf = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = clf.fit_predict(df.dropna())
            outliers = df.dropna()[outlier_labels == -1].index.tolist()
            return outliers
        except Exception as e:
            self.logger.warning(f"孤立森林检测失败: {e}")
            return []
    
    def _detect_outliers_lof(self, df: pd.DataFrame, 
                           n_neighbors: int = 20) -> List[int]:
        """使用局部异常因子检测异常值"""
        try:
            clf = LocalOutlierFactor(n_neighbors=n_neighbors)
            outlier_labels = clf.fit_predict(df.dropna())
            outliers = df.dropna()[outlier_labels == -1].index.tolist()
            return outliers
        except Exception as e:
            self.logger.warning(f"LOF检测失败: {e}")
            return []
    
    def _cap_outliers(self, df: pd.DataFrame, column: str, outlier_indices: List[int]) -> pd.DataFrame:
        """限制异常值到合理范围"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df.loc[df[column] < lower_bound, column] = lower_bound
        df.loc[df[column] > upper_bound, column] = upper_bound
        
        return df
    
    def _transform_outliers(self, series: pd.Series) -> pd.Series:
        """对异常值进行变换（对数变换）"""
        if (series > 0).all():
            return np.log1p(series)
        else:
            # 使用Box-Cox变换
            try:
                transformed, _ = stats.boxcox(series - series.min() + 1)
                return pd.Series(transformed, index=series.index)
            except:
                return series
    
    def scale_features(self, df: pd.DataFrame, 
                      method: str = 'standard',
                      columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        特征标准化
        
        Args:
            df: 输入数据
            method: 标准化方法 ('standard', 'minmax', 'robust')
            columns: 要标准化的列名，None表示所有数值列
            
        Returns:
            标准化后的数据和处理报告
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 选择标准化器
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        # 应用标准化
        if columns:
            df_scaled[columns] = scaler.fit_transform(df[columns])
            self.scalers[method] = scaler
        
        report = {
            'method': method,
            'columns_scaled': columns,
            'scaler_params': scaler.get_params() if hasattr(scaler, 'get_params') else {}
        }
        
        return df_scaled, report
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        智能特征工程
        
        Args:
            df: 输入数据
            
        Returns:
            增强后的数据和特征报告
        """
        df_enhanced = df.copy()
        new_features = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # 1. 创建多项式特征（二次特征）
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                new_feature_name = f"{col1}_x_{col2}"
                df_enhanced[new_feature_name] = df[col1] * df[col2]
                new_features.append(new_feature_name)
        
        # 2. 创建比率特征
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                if not (df[col2] == 0).any():
                    new_feature_name = f"{col1}_div_{col2}"
                    df_enhanced[new_feature_name] = df[col1] / df[col2]
                    new_features.append(new_feature_name)
        
        # 3. 创建聚合特征
        if len(numeric_columns) > 1:
            df_enhanced['numeric_mean'] = df[numeric_columns].mean(axis=1)
            df_enhanced['numeric_std'] = df[numeric_columns].std(axis=1)
            df_enhanced['numeric_sum'] = df[numeric_columns].sum(axis=1)
            new_features.extend(['numeric_mean', 'numeric_std', 'numeric_sum'])
        
        # 4. 创建分箱特征
        for col in numeric_columns[:3]:  # 限制数量
            try:
                df_enhanced[f"{col}_binned"] = pd.cut(df[col], bins=5, labels=False)
                new_features.append(f"{col}_binned")
            except:
                continue
        
        report = {
            'original_features': len(df.columns),
            'new_features': len(new_features),
            'total_features': len(df_enhanced.columns),
            'new_feature_names': new_features
        }
        
        return df_enhanced, report
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """获取预处理器状态摘要"""
        return {
            'scalers_fitted': list(self.scalers.keys()),
            'imputers_fitted': list(self.imputers.keys()),
            'outlier_detectors_fitted': list(self.outlier_detectors.keys()),
            'config': self.config
        }