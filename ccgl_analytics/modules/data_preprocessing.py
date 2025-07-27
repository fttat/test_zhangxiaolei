"""
数据预处理模块

提供缺失值处理、异常检测、数据标准化等功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
import logging


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化数据预处理器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.missing_strategy = config.get('missing_values', {}).get('strategy', 'auto')
        self.outlier_methods = config.get('outlier_detection', {}).get('methods', ['iqr'])
        self.outlier_threshold = config.get('outlier_detection', {}).get('threshold', 3.0)
        self.normalization_method = config.get('normalization', {}).get('default_method', 'standard')
        
        # 预处理状态
        self.scalers = {}
        self.imputers = {}
        self.outlier_detectors = {}
        
        self.logger.info("数据预处理器初始化完成")
    
    async def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """完整的数据预处理流程"""
        try:
            self.logger.info(f"开始预处理数据，原始数据: {data.shape}")
            
            # 1. 数据清洗
            cleaned_data = self._clean_data(data)
            
            # 2. 处理缺失值
            imputed_data = self._handle_missing_values(cleaned_data)
            
            # 3. 异常检测和处理
            outlier_handled_data = self._handle_outliers(imputed_data)
            
            # 4. 数据标准化
            normalized_data = self._normalize_data(outlier_handled_data)
            
            self.logger.info(f"数据预处理完成，处理后数据: {normalized_data.shape}")
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        cleaned_data = data.copy()
        
        # 删除完全重复的行
        before_dedup = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        after_dedup = len(cleaned_data)
        
        if before_dedup != after_dedup:
            self.logger.info(f"删除重复行: {before_dedup - after_dedup} 行")
        
        # 删除全为空的列
        before_cols = len(cleaned_data.columns)
        cleaned_data = cleaned_data.dropna(axis=1, how='all')
        after_cols = len(cleaned_data.columns)
        
        if before_cols != after_cols:
            self.logger.info(f"删除全空列: {before_cols - after_cols} 列")
        
        # 处理数据类型
        cleaned_data = self._convert_data_types(cleaned_data)
        
        return cleaned_data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        converted_data = data.copy()
        
        for col in converted_data.columns:
            # 尝试转换为数值类型
            if converted_data[col].dtype == 'object':
                # 检查是否为日期时间
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    try:
                        converted_data[col] = pd.to_datetime(converted_data[col])
                        self.logger.debug(f"列 {col} 转换为日期时间类型")
                        continue
                    except:
                        pass
                
                # 尝试转换为数值
                try:
                    converted_data[col] = pd.to_numeric(converted_data[col], errors='coerce')
                    if not converted_data[col].isna().all():
                        self.logger.debug(f"列 {col} 转换为数值类型")
                except:
                    pass
        
        return converted_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        if data.isnull().sum().sum() == 0:
            self.logger.info("数据无缺失值")
            return data
        
        # 获取数值列和分类列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        imputed_data = data.copy()
        
        # 处理数值列缺失值
        if len(numeric_cols) > 0:
            imputed_data[numeric_cols] = self._impute_numeric_columns(imputed_data[numeric_cols])
        
        # 处理分类列缺失值
        if len(categorical_cols) > 0:
            imputed_data[categorical_cols] = self._impute_categorical_columns(imputed_data[categorical_cols])
        
        missing_after = imputed_data.isnull().sum().sum()
        self.logger.info(f"缺失值处理完成，剩余缺失值: {missing_after}")
        
        return imputed_data
    
    def _impute_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """填充数值列缺失值"""
        imputed_data = data.copy()
        
        for col in data.columns:
            if data[col].isnull().any():
                missing_count = data[col].isnull().sum()
                missing_ratio = missing_count / len(data)
                
                if missing_ratio > 0.5:
                    # 缺失值超过50%，删除列
                    imputed_data = imputed_data.drop(columns=[col])
                    self.logger.warning(f"列 {col} 缺失值过多({missing_ratio:.2%})，已删除")
                    continue
                
                # 选择填充策略
                if self.missing_strategy == 'auto':
                    if missing_ratio < 0.1:
                        strategy = 'mean'
                    elif missing_ratio < 0.3:
                        strategy = 'median'
                    else:
                        strategy = 'knn'
                else:
                    strategy = self.missing_strategy
                
                # 执行填充
                if strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=min(5, len(data) // 10))
                    imputed_data[[col]] = imputer.fit_transform(imputed_data[[col]])
                else:
                    imputer = SimpleImputer(strategy=strategy)
                    imputed_data[[col]] = imputer.fit_transform(imputed_data[[col]])
                
                self.imputers[col] = imputer
                self.logger.debug(f"列 {col} 使用 {strategy} 策略填充缺失值")
        
        return imputed_data
    
    def _impute_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """填充分类列缺失值"""
        imputed_data = data.copy()
        
        for col in data.columns:
            if data[col].isnull().any():
                # 使用众数填充
                mode_value = data[col].mode()
                if len(mode_value) > 0:
                    imputed_data[col] = imputed_data[col].fillna(mode_value[0])
                    self.logger.debug(f"列 {col} 使用众数填充缺失值")
                else:
                    # 如果没有众数，使用固定值
                    imputed_data[col] = imputed_data[col].fillna('未知')
                    self.logger.debug(f"列 {col} 使用固定值填充缺失值")
        
        return imputed_data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return data
        
        outlier_handled_data = data.copy()
        outlier_masks = {}
        
        for method in self.outlier_methods:
            if method == 'iqr':
                mask = self._detect_outliers_iqr(numeric_data)
            elif method == 'zscore':
                mask = self._detect_outliers_zscore(numeric_data)
            elif method == 'isolation_forest':
                mask = self._detect_outliers_isolation_forest(numeric_data)
            else:
                continue
            
            outlier_masks[method] = mask
        
        # 合并异常值检测结果
        if outlier_masks:
            combined_mask = pd.Series(False, index=data.index)
            for mask in outlier_masks.values():
                combined_mask = combined_mask | mask
            
            outlier_count = combined_mask.sum()
            self.logger.info(f"检测到异常值: {outlier_count} 行")
            
            # 处理异常值（使用中位数替换）
            for col in numeric_data.columns:
                col_mask = combined_mask & data[col].notna()
                if col_mask.any():
                    median_value = data[col].median()
                    outlier_handled_data.loc[col_mask, col] = median_value
        
        return outlier_handled_data
    
    def _detect_outliers_iqr(self, data: pd.DataFrame) -> pd.Series:
        """使用IQR方法检测异常值"""
        outlier_mask = pd.Series(False, index=data.index)
        
        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers
        
        return outlier_mask
    
    def _detect_outliers_zscore(self, data: pd.DataFrame) -> pd.Series:
        """使用Z-Score方法检测异常值"""
        outlier_mask = pd.Series(False, index=data.index)
        
        for col in data.columns:
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            col_outliers = z_scores > self.outlier_threshold
            
            # 将结果映射回原始索引
            outlier_indices = data[col].dropna().index[col_outliers]
            outlier_mask.loc[outlier_indices] = True
        
        return outlier_mask
    
    def _detect_outliers_isolation_forest(self, data: pd.DataFrame) -> pd.Series:
        """使用孤立森林检测异常值"""
        try:
            # 只使用完整的行进行训练
            complete_data = data.dropna()
            if len(complete_data) < 10:
                return pd.Series(False, index=data.index)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(complete_data)
            
            outlier_mask = pd.Series(False, index=data.index)
            outlier_indices = complete_data.index[outlier_labels == -1]
            outlier_mask.loc[outlier_indices] = True
            
            return outlier_mask
            
        except Exception as e:
            self.logger.warning(f"孤立森林异常检测失败: {e}")
            return pd.Series(False, index=data.index)
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return data
        
        normalized_data = data.copy()
        
        # 选择标准化方法
        if self.normalization_method == 'standard':
            scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.normalization_method == 'robust':
            scaler = RobustScaler()
        else:
            self.logger.warning(f"未知的标准化方法: {self.normalization_method}")
            return data
        
        # 执行标准化
        try:
            normalized_values = scaler.fit_transform(numeric_data)
            normalized_data[numeric_data.columns] = normalized_values
            
            self.scalers['main'] = scaler
            self.logger.info(f"使用 {self.normalization_method} 方法完成数据标准化")
            
        except Exception as e:
            self.logger.error(f"数据标准化失败: {e}")
            return data
        
        return normalized_data
    
    def get_preprocessing_summary(self, original_data: pd.DataFrame, 
                                processed_data: pd.DataFrame) -> Dict[str, Any]:
        """获取预处理摘要"""
        summary = {
            'original_shape': original_data.shape,
            'processed_shape': processed_data.shape,
            'missing_values_before': original_data.isnull().sum().sum(),
            'missing_values_after': processed_data.isnull().sum().sum(),
            'data_types': processed_data.dtypes.value_counts().to_dict(),
            'preprocessing_steps': {
                'missing_strategy': self.missing_strategy,
                'outlier_methods': self.outlier_methods,
                'normalization_method': self.normalization_method
            }
        }
        
        return summary
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """对新数据应用已训练的预处理器"""
        try:
            transformed_data = data.copy()
            
            # 应用已训练的填充器
            for col, imputer in self.imputers.items():
                if col in transformed_data.columns:
                    transformed_data[[col]] = imputer.transform(transformed_data[[col]])
            
            # 应用已训练的标准化器
            if 'main' in self.scalers:
                numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    transformed_data[numeric_cols] = self.scalers['main'].transform(transformed_data[numeric_cols])
            
            self.logger.info("新数据预处理完成")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"新数据预处理失败: {e}")
            return data