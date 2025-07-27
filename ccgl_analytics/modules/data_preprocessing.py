"""
Data Preprocessing Module for CCGL Analytics
Handles data cleaning, transformation, and quality assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import warnings

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.logger import get_logger


class DataQualityAssessment:
    """Data quality assessment and reporting"""
    
    def __init__(self):
        self.logger = get_logger("data_quality")
    
    def assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dictionary containing quality metrics
        """
        self.logger.info("Starting data quality assessment")
        
        quality_report = {
            'overview': self._get_overview(df),
            'completeness': self._assess_completeness(df),
            'consistency': self._assess_consistency(df),
            'timeliness': self._assess_timeliness(df),
            'accuracy': self._assess_accuracy(df),
            'uniqueness': self._assess_uniqueness(df)
        }
        
        # Calculate overall quality score
        quality_report['overall_score'] = self._calculate_quality_score(quality_report)
        
        self.logger.info(f"Data quality assessment completed. Overall score: {quality_report['overall_score']:.2f}")
        
        return quality_report
    
    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic overview of the dataset"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_types': df.dtypes.value_counts().to_dict()
        }
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        return {
            'missing_values_count': {str(k): int(v) for k, v in missing_data.to_dict().items()},
            'missing_percentage': {str(k): float(v) for k, v in missing_percentage.to_dict().items()},
            'columns_with_missing': missing_data[missing_data > 0].index.tolist(),
            'completeness_score': float(1 - missing_percentage.mean() / 100)
        }
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency"""
        consistency_issues = []
        
        # Check for inconsistent data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in string columns
                try:
                    pd.to_numeric(df[col], errors='raise')
                    consistency_issues.append(f"Column '{col}' contains numeric data stored as text")
                except (ValueError, TypeError):
                    pass
        
        # Check for inconsistent date formats
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if df[col].isnull().sum() < len(df):  # If not all null
                date_range = df[col].dropna()
                if len(date_range) > 0:
                    # Check for unrealistic date ranges
                    min_date = date_range.min()
                    max_date = date_range.max()
                    if min_date.year < 1900 or max_date.year > 2030:
                        consistency_issues.append(f"Column '{col}' contains unrealistic dates")
        
        return {
            'consistency_issues': consistency_issues,
            'consistency_score': max(0, 1 - len(consistency_issues) / len(df.columns))
        }
    
    def _assess_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data timeliness"""
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        if len(date_columns) == 0:
            return {
                'timeliness_score': 1.0,
                'message': 'No date columns found for timeliness assessment'
            }
        
        timeliness_scores = []
        for col in date_columns:
            if df[col].isnull().sum() < len(df):
                latest_date = df[col].dropna().max()
                days_old = (datetime.now() - latest_date.to_pydatetime()).days
                
                # Score based on recency (90 days = 1.0, older = lower score)
                score = max(0, 1 - days_old / 90)
                timeliness_scores.append(score)
        
        return {
            'timeliness_score': np.mean(timeliness_scores) if timeliness_scores else 1.0,
            'date_columns_analyzed': list(date_columns)
        }
    
    def _assess_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data accuracy"""
        accuracy_issues = []
        
        # Check for outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().sum() < len(df):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if len(outliers) > 0:
                    accuracy_issues.append(f"Column '{col}' has {len(outliers)} outliers")
        
        # Check for negative values where they shouldn't exist
        for col in ['quantity', 'price', 'amount', 'count']:
            if col in df.columns and df[col].dtype in [np.int64, np.float64]:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    accuracy_issues.append(f"Column '{col}' has {negative_count} negative values")
        
        return {
            'accuracy_issues': accuracy_issues,
            'accuracy_score': max(0, 1 - len(accuracy_issues) / len(df.columns))
        }
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness"""
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df)) * 100
        
        # Check for columns that should be unique
        potential_id_columns = [col for col in df.columns if 'id' in col.lower()]
        uniqueness_issues = []
        
        for col in potential_id_columns:
            unique_count = df[col].nunique()
            total_count = df[col].count()  # Non-null count
            if unique_count < total_count:
                uniqueness_issues.append(f"Column '{col}' has duplicate values")
        
        return {
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': duplicate_percentage,
            'uniqueness_issues': uniqueness_issues,
            'uniqueness_score': 1 - duplicate_percentage / 100
        }
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = [
            quality_report['completeness']['completeness_score'],
            quality_report['consistency']['consistency_score'],
            quality_report['timeliness']['timeliness_score'],
            quality_report['accuracy']['accuracy_score'],
            quality_report['uniqueness']['uniqueness_score']
        ]
        
        return np.mean(scores)


class DataPreprocessor:
    """
    Comprehensive data preprocessing and transformation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data preprocessor
        
        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or {}
        self.logger = get_logger("data_preprocessing")
        self.quality_assessor = DataQualityAssessment()
        
        # Preprocessing history
        self.preprocessing_steps = []
        self.scalers = {}
        self.imputers = {}
    
    def preprocess(self, df: pd.DataFrame, steps: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: DataFrame to preprocess
            steps: List of preprocessing steps to apply
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Starting data preprocessing pipeline")
        
        # Default preprocessing steps
        if steps is None:
            steps = ['quality_check', 'handle_missing', 'detect_outliers', 'normalize_data']
        
        processed_df = df.copy()
        
        for step in steps:
            self.logger.info(f"Applying preprocessing step: {step}")
            
            if step == 'quality_check':
                quality_report = self.quality_assessor.assess_quality(processed_df)
                self.preprocessing_steps.append(('quality_check', quality_report))
                
            elif step == 'handle_missing':
                processed_df = self.handle_missing_values(processed_df)
                
            elif step == 'detect_outliers':
                processed_df = self.detect_and_handle_outliers(processed_df)
                
            elif step == 'normalize_data':
                processed_df = self.normalize_data(processed_df)
                
            elif step == 'encode_categorical':
                processed_df = self.encode_categorical_variables(processed_df)
                
            elif step == 'feature_engineering':
                processed_df = self.create_features(processed_df)
                
            else:
                self.logger.warning(f"Unknown preprocessing step: {step}")
        
        self.logger.info("Data preprocessing pipeline completed")
        return processed_df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values using various strategies
        
        Args:
            df: DataFrame with missing values
            strategy: Imputation strategy ('auto', 'mean', 'median', 'mode', 'knn', 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. Using basic imputation.")
            return self._basic_missing_value_handling(df)
        
        df_processed = df.copy()
        missing_info = df.isnull().sum()
        columns_with_missing = missing_info[missing_info > 0].index.tolist()
        
        if not columns_with_missing:
            self.logger.info("No missing values found")
            return df_processed
        
        self.logger.info(f"Handling missing values in {len(columns_with_missing)} columns")
        
        for col in columns_with_missing:
            missing_ratio = missing_info[col] / len(df)
            
            # If more than 50% missing, consider dropping the column
            if missing_ratio > 0.5:
                self.logger.warning(f"Column '{col}' has {missing_ratio:.1%} missing values. Consider dropping.")
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                # Numeric columns
                if strategy == 'auto':
                    if missing_ratio < 0.1:
                        strategy_to_use = 'mean'
                    else:
                        strategy_to_use = 'median'
                else:
                    strategy_to_use = strategy
                
                if strategy_to_use == 'knn':
                    # Use KNN imputation for numeric data
                    imputer = KNNImputer(n_neighbors=5)
                    df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
                else:
                    # Use simple imputation
                    imputer = SimpleImputer(strategy=strategy_to_use)
                    df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
                
                self.imputers[col] = imputer
                
            else:
                # Categorical columns
                imputer = SimpleImputer(strategy='most_frequent')
                df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
                self.imputers[col] = imputer
        
        self.preprocessing_steps.append(('handle_missing', {
            'strategy': strategy,
            'columns_processed': columns_with_missing
        }))
        
        return df_processed
    
    def _basic_missing_value_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic missing value handling without scikit-learn"""
        df_processed = df.copy()
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df_processed[col].fillna(df[col].median(), inplace=True)
                else:
                    # Fill categorical columns with mode
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        df_processed[col].fillna(mode_value[0], inplace=True)
        
        return df_processed
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and handle outliers
        
        Args:
            df: DataFrame to process
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            DataFrame with outliers handled
        """
        df_processed = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        outliers_detected = {}
        
        for col in numeric_columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > 3
                
            elif method == 'isolation_forest' and SKLEARN_AVAILABLE:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(df[[col]].fillna(df[col].median())) == -1
                
            else:
                continue
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                outliers_detected[col] = outlier_count
                
                # Cap outliers at percentiles
                lower_cap = df[col].quantile(0.05)
                upper_cap = df[col].quantile(0.95)
                
                df_processed[col] = df_processed[col].clip(lower=lower_cap, upper=upper_cap)
        
        if outliers_detected:
            self.logger.info(f"Detected and handled outliers: {outliers_detected}")
        
        self.preprocessing_steps.append(('detect_outliers', {
            'method': method,
            'outliers_detected': outliers_detected
        }))
        
        return df_processed
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numeric data
        
        Args:
            df: DataFrame to normalize
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Normalized DataFrame
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. Skipping normalization.")
            return df
        
        df_processed = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            self.logger.info("No numeric columns found for normalization")
            return df_processed
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown normalization method: {method}. Using standard scaling.")
            scaler = StandardScaler()
        
        # Fit and transform numeric columns
        df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        self.preprocessing_steps.append(('normalize_data', {
            'method': method,
            'columns_normalized': list(numeric_columns)
        }))
        
        self.logger.info(f"Normalized {len(numeric_columns)} numeric columns using {method} scaling")
        
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: DataFrame to encode
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_processed = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) == 0:
            self.logger.info("No categorical columns found for encoding")
            return df_processed
        
        encoding_info = {}
        
        for col in categorical_columns:
            unique_values = df[col].nunique()
            
            if unique_values <= 10:
                # Use one-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
                encoding_info[col] = f"one_hot ({unique_values} categories)"
                
            else:
                # Use label encoding for high cardinality
                df_processed[col] = pd.Categorical(df[col]).codes
                encoding_info[col] = f"label_encoded ({unique_values} categories)"
        
        self.preprocessing_steps.append(('encode_categorical', encoding_info))
        
        self.logger.info(f"Encoded {len(categorical_columns)} categorical columns")
        
        return df_processed
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from existing data
        
        Args:
            df: DataFrame to enhance with features
            
        Returns:
            DataFrame with additional features
        """
        df_processed = df.copy()
        new_features = []
        
        # Date-based features
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if not df[col].isnull().all():
                df_processed[f'{col}_year'] = df[col].dt.year
                df_processed[f'{col}_month'] = df[col].dt.month
                df_processed[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df_processed[f'{col}_quarter'] = df[col].dt.quarter
                new_features.extend([f'{col}_year', f'{col}_month', f'{col}_day_of_week', f'{col}_quarter'])
        
        # Numeric feature interactions
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            # Create ratio features for related columns
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    if not (df[col2] == 0).any():  # Avoid division by zero
                        df_processed[f'{col1}_{col2}_ratio'] = df[col1] / df[col2]
                        new_features.append(f'{col1}_{col2}_ratio')
        
        if new_features:
            self.preprocessing_steps.append(('feature_engineering', {
                'new_features': new_features
            }))
            
            self.logger.info(f"Created {len(new_features)} new features")
        
        return df_processed
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of all preprocessing steps applied
        
        Returns:
            Dictionary containing preprocessing summary
        """
        return {
            'total_steps': len(self.preprocessing_steps),
            'steps_applied': [step[0] for step in self.preprocessing_steps],
            'detailed_steps': self.preprocessing_steps
        }