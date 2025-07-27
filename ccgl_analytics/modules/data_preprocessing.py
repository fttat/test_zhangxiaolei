"""
CCGL Analytics - Data Preprocessing Module
Comprehensive data preprocessing, cleaning, and transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings

from ..utils.logger import get_logger, LoggerMixin

class MissingValueHandler(LoggerMixin):
    """Handler for missing value detection and imputation."""
    
    def __init__(self, strategy: str = 'auto'):
        """Initialize missing value handler.
        
        Args:
            strategy: Imputation strategy ('mean', 'median', 'mode', 'knn', 'forward_fill', 'backward_fill', 'auto')
        """
        self.strategy = strategy
        self.imputers = {}
        self.statistics = {}
    
    def analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Missing value analysis results
        """
        self.logger.info("Analyzing missing value patterns")
        
        missing_info = {}
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            missing_percentage = (missing_count / len(data)) * 100
            
            missing_info[column] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_percentage),
                'data_type': str(data[column].dtype),
                'non_null_count': int(len(data) - missing_count)
            }
        
        # Overall statistics
        total_missing = sum(info['missing_count'] for info in missing_info.values())
        total_cells = len(data) * len(data.columns)
        
        analysis = {
            'column_analysis': missing_info,
            'total_missing_values': total_missing,
            'total_cells': total_cells,
            'overall_missing_percentage': float((total_missing / total_cells) * 100),
            'columns_with_missing': [col for col, info in missing_info.items() if info['missing_count'] > 0],
            'missing_patterns': self._analyze_missing_patterns(data)
        }
        
        return analysis
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing values.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Missing pattern analysis
        """
        # Create missing value indicator matrix
        missing_matrix = data.isnull()
        
        # Find rows with missing values
        rows_with_missing = missing_matrix.any(axis=1).sum()
        
        # Find combinations of missing columns
        if len(data.columns) <= 20:  # Only for smaller datasets to avoid combinatorial explosion
            # Get unique missing patterns
            missing_patterns = missing_matrix.apply(lambda x: tuple(x), axis=1)
            pattern_counts = missing_patterns.value_counts()
            
            # Convert to readable format
            readable_patterns = {}
            for pattern, count in pattern_counts.head(10).items():  # Top 10 patterns
                missing_cols = [data.columns[i] for i, is_missing in enumerate(pattern) if is_missing]
                pattern_key = f"Missing: {', '.join(missing_cols)}" if missing_cols else "Complete"
                readable_patterns[pattern_key] = int(count)
        else:
            readable_patterns = {"Analysis skipped": "Too many columns for pattern analysis"}
        
        return {
            'rows_with_missing': int(rows_with_missing),
            'rows_complete': int(len(data) - rows_with_missing),
            'completion_rate': float((len(data) - rows_with_missing) / len(data)),
            'common_patterns': readable_patterns
        }
    
    def impute_missing_values(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Impute missing values in the dataset.
        
        Args:
            data: DataFrame with missing values
            target_column: Target column for supervised imputation
            
        Returns:
            DataFrame with imputed values
        """
        self.logger.info(f"Imputing missing values using strategy: {self.strategy}")
        
        data_imputed = data.copy()
        
        # Separate numeric and categorical columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric columns
        if numeric_columns:
            data_imputed[numeric_columns] = self._impute_numeric_columns(
                data[numeric_columns], target_column
            )
        
        # Handle categorical columns
        if categorical_columns:
            data_imputed[categorical_columns] = self._impute_categorical_columns(
                data[categorical_columns]
            )
        
        self.logger.info("Missing value imputation completed")
        return data_imputed
    
    def _impute_numeric_columns(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Impute missing values in numeric columns.
        
        Args:
            data: Numeric DataFrame
            target_column: Target column for supervised imputation
            
        Returns:
            DataFrame with imputed numeric values
        """
        if self.strategy == 'auto':
            # Choose strategy based on missing percentage
            strategy = self._choose_optimal_strategy(data, 'numeric')
        else:
            strategy = self.strategy
        
        if strategy == 'knn':
            # Use KNN imputation for numeric data
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            self.imputers['numeric_knn'] = imputer
            
        elif strategy in ['mean', 'median']:
            # Use statistical imputation
            imputer = SimpleImputer(strategy=strategy)
            imputed_data = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            self.imputers[f'numeric_{strategy}'] = imputer
            
        elif strategy == 'forward_fill':
            imputed_data = data.fillna(method='ffill')
            
        elif strategy == 'backward_fill':
            imputed_data = data.fillna(method='bfill')
            
        else:
            # Default to median
            imputer = SimpleImputer(strategy='median')
            imputed_data = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            self.imputers['numeric_median'] = imputer
        
        return imputed_data
    
    def _impute_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in categorical columns.
        
        Args:
            data: Categorical DataFrame
            
        Returns:
            DataFrame with imputed categorical values
        """
        strategy = 'most_frequent'  # Most common strategy for categorical data
        
        imputer = SimpleImputer(strategy=strategy)
        
        # Convert to string to handle mixed types
        data_str = data.astype(str).replace('nan', np.nan)
        
        imputed_data = pd.DataFrame(
            imputer.fit_transform(data_str),
            columns=data.columns,
            index=data.index
        )
        
        self.imputers['categorical_mode'] = imputer
        
        return imputed_data
    
    def _choose_optimal_strategy(self, data: pd.DataFrame, data_type: str) -> str:
        """Choose optimal imputation strategy based on data characteristics.
        
        Args:
            data: DataFrame to analyze
            data_type: Type of data ('numeric' or 'categorical')
            
        Returns:
            Optimal strategy name
        """
        # Calculate missing percentage
        missing_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        if data_type == 'numeric':
            if missing_percentage < 0.05:  # Less than 5% missing
                return 'mean'
            elif missing_percentage < 0.15:  # Less than 15% missing
                return 'median'
            else:  # More than 15% missing
                return 'knn'
        else:
            return 'most_frequent'

class OutlierDetector(LoggerMixin):
    """Outlier detection and handling."""
    
    def __init__(self, method: str = 'iqr'):
        """Initialize outlier detector.
        
        Args:
            method: Detection method ('iqr', 'zscore', 'isolation_forest', 'local_outlier_factor')
        """
        self.method = method
        self.outlier_bounds = {}
        self.outlier_models = {}
    
    def detect_outliers(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect outliers in the dataset.
        
        Args:
            data: DataFrame to analyze
            columns: Specific columns to analyze (if None, analyze all numeric columns)
            
        Returns:
            Outlier detection results
        """
        self.logger.info(f"Detecting outliers using method: {self.method}")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_results = {
            'method': self.method,
            'columns_analyzed': columns,
            'column_results': {},
            'summary': {}
        }
        
        total_outliers = 0
        
        for column in columns:
            if column not in data.columns:
                continue
                
            column_data = data[column].dropna()
            
            if len(column_data) == 0:
                continue
            
            if self.method == 'iqr':
                outliers = self._detect_iqr_outliers(column_data)
            elif self.method == 'zscore':
                outliers = self._detect_zscore_outliers(column_data)
            elif self.method == 'isolation_forest':
                outliers = self._detect_isolation_forest_outliers(column_data)
            else:
                outliers = self._detect_iqr_outliers(column_data)  # Default
            
            outlier_count = outliers.sum()
            total_outliers += outlier_count
            
            outlier_results['column_results'][column] = {
                'outlier_count': int(outlier_count),
                'outlier_percentage': float(outlier_count / len(column_data) * 100),
                'outlier_indices': outliers[outliers].index.tolist(),
                'data_points_analyzed': len(column_data)
            }
        
        outlier_results['summary'] = {
            'total_outliers': total_outliers,
            'columns_with_outliers': len([col for col, result in outlier_results['column_results'].items() 
                                        if result['outlier_count'] > 0])
        }
        
        self.logger.info(f"Outlier detection completed. Found {total_outliers} outliers across {len(columns)} columns")
        
        return outlier_results
    
    def _detect_iqr_outliers(self, data: pd.Series) -> pd.Series:
        """Detect outliers using Interquartile Range method.
        
        Args:
            data: Series to analyze
            
        Returns:
            Boolean series indicating outliers
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.outlier_bounds[data.name] = {
            'method': 'iqr',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
        
        return (data < lower_bound) | (data > upper_bound)
    
    def _detect_zscore_outliers(self, data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method.
        
        Args:
            data: Series to analyze
            threshold: Z-score threshold
            
        Returns:
            Boolean series indicating outliers
        """
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return pd.Series([False] * len(data), index=data.index)
        
        z_scores = np.abs((data - mean) / std)
        
        self.outlier_bounds[data.name] = {
            'method': 'zscore',
            'mean': mean,
            'std': std,
            'threshold': threshold
        }
        
        return z_scores > threshold
    
    def _detect_isolation_forest_outliers(self, data: pd.Series, contamination: float = 0.1) -> pd.Series:
        """Detect outliers using Isolation Forest.
        
        Args:
            data: Series to analyze
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean series indicating outliers
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            data_reshaped = data.values.reshape(-1, 1)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(data_reshaped) == -1
            
            self.outlier_models[data.name] = iso_forest
            
            return pd.Series(outliers, index=data.index)
            
        except ImportError:
            self.logger.warning("scikit-learn not available, falling back to IQR method")
            return self._detect_iqr_outliers(data)
    
    def handle_outliers(self, data: pd.DataFrame, method: str = 'clip', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle detected outliers.
        
        Args:
            data: DataFrame with outliers
            method: Handling method ('clip', 'remove', 'transform')
            columns: Columns to process
            
        Returns:
            DataFrame with outliers handled
        """
        self.logger.info(f"Handling outliers using method: {method}")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_processed = data.copy()
        
        for column in columns:
            if column not in self.outlier_bounds:
                continue
            
            bounds = self.outlier_bounds[column]
            
            if method == 'clip':
                if bounds['method'] == 'iqr':
                    data_processed[column] = data_processed[column].clip(
                        lower=bounds['lower_bound'],
                        upper=bounds['upper_bound']
                    )
                
            elif method == 'remove':
                # Mark outliers for removal (will be handled at DataFrame level)
                if bounds['method'] == 'iqr':
                    outlier_mask = (
                        (data_processed[column] < bounds['lower_bound']) |
                        (data_processed[column] > bounds['upper_bound'])
                    )
                    data_processed = data_processed[~outlier_mask]
                    
            elif method == 'transform':
                # Apply log transformation to reduce impact of outliers
                if (data_processed[column] > 0).all():
                    data_processed[column] = np.log1p(data_processed[column])
        
        return data_processed

class DataScaler(LoggerMixin):
    """Data scaling and normalization."""
    
    def __init__(self, method: str = 'standard'):
        """Initialize data scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'quantile')
        """
        self.method = method
        self.scalers = {}
    
    def fit_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit scaler and transform data.
        
        Args:
            data: DataFrame to scale
            columns: Columns to scale (if None, scale all numeric columns)
            
        Returns:
            Scaled DataFrame
        """
        self.logger.info(f"Scaling data using method: {self.method}")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_scaled = data.copy()
        
        if self.method == 'standard':
            scaler = StandardScaler()
        elif self.method == 'minmax':
            scaler = MinMaxScaler()
        elif self.method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()  # Default
        
        if columns:
            # Fit and transform numeric columns
            scaled_values = scaler.fit_transform(data[columns])
            data_scaled[columns] = scaled_values
            
            # Store scaler for later use
            self.scalers[f'{self.method}_scaler'] = scaler
        
        self.logger.info(f"Data scaling completed for {len(columns)} columns")
        
        return data_scaled
    
    def transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Transform data using fitted scaler.
        
        Args:
            data: DataFrame to transform
            columns: Columns to transform
            
        Returns:
            Transformed DataFrame
        """
        if f'{self.method}_scaler' not in self.scalers:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_scaled = data.copy()
        scaler = self.scalers[f'{self.method}_scaler']
        
        if columns:
            scaled_values = scaler.transform(data[columns])
            data_scaled[columns] = scaled_values
        
        return data_scaled

class FeatureEngineer(LoggerMixin):
    """Feature engineering and transformation."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.encoders = {}
        self.feature_stats = {}
    
    def create_datetime_features(self, data: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
        """Create features from datetime columns.
        
        Args:
            data: DataFrame with datetime columns
            datetime_columns: List of datetime column names
            
        Returns:
            DataFrame with additional datetime features
        """
        self.logger.info(f"Creating datetime features for {len(datetime_columns)} columns")
        
        data_enhanced = data.copy()
        
        for column in datetime_columns:
            if column not in data.columns:
                continue
            
            # Convert to datetime if not already
            datetime_series = pd.to_datetime(data[column], errors='coerce')
            
            # Extract features
            data_enhanced[f'{column}_year'] = datetime_series.dt.year
            data_enhanced[f'{column}_month'] = datetime_series.dt.month
            data_enhanced[f'{column}_day'] = datetime_series.dt.day
            data_enhanced[f'{column}_dayofweek'] = datetime_series.dt.dayofweek
            data_enhanced[f'{column}_quarter'] = datetime_series.dt.quarter
            data_enhanced[f'{column}_is_weekend'] = datetime_series.dt.dayofweek.isin([5, 6]).astype(int)
            
            # Add hour for datetime (not just date)
            if datetime_series.dt.hour.notna().any():
                data_enhanced[f'{column}_hour'] = datetime_series.dt.hour
        
        return data_enhanced
    
    def encode_categorical_features(self, data: pd.DataFrame, categorical_columns: Optional[List[str]] = None,
                                  encoding_method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            data: DataFrame with categorical columns
            categorical_columns: List of categorical column names
            encoding_method: Encoding method ('onehot', 'label', 'target')
            
        Returns:
            DataFrame with encoded features
        """
        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.logger.info(f"Encoding {len(categorical_columns)} categorical features using {encoding_method}")
        
        data_encoded = data.copy()
        
        for column in categorical_columns:
            if column not in data.columns:
                continue
            
            if encoding_method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(data[column], prefix=column, dummy_na=True)
                data_encoded = pd.concat([data_encoded.drop(column, axis=1), dummies], axis=1)
                
            elif encoding_method == 'label':
                # Label encoding
                encoder = LabelEncoder()
                # Handle missing values
                data_filled = data[column].fillna('missing')
                encoded_values = encoder.fit_transform(data_filled)
                data_encoded[f'{column}_encoded'] = encoded_values
                self.encoders[f'{column}_label'] = encoder
        
        return data_encoded
    
    def create_interaction_features(self, data: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between specified pairs.
        
        Args:
            data: Input DataFrame
            feature_pairs: List of feature pairs to create interactions
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info(f"Creating {len(feature_pairs)} interaction features")
        
        data_enhanced = data.copy()
        
        for feature1, feature2 in feature_pairs:
            if feature1 in data.columns and feature2 in data.columns:
                # Check if both features are numeric
                if (pd.api.types.is_numeric_dtype(data[feature1]) and 
                    pd.api.types.is_numeric_dtype(data[feature2])):
                    # Multiplicative interaction
                    data_enhanced[f'{feature1}_x_{feature2}'] = data[feature1] * data[feature2]
                    
                    # Ratio (avoid division by zero)
                    data_enhanced[f'{feature1}_div_{feature2}'] = data[feature1] / (data[feature2] + 1e-8)
        
        return data_enhanced

class DataPreprocessor(LoggerMixin):
    """Main data preprocessing pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.missing_handler = MissingValueHandler()
        self.outlier_detector = OutlierDetector()
        self.scaler = DataScaler()
        self.feature_engineer = FeatureEngineer()
        self.preprocessing_steps = []
    
    def preprocess_data(self, data: pd.DataFrame, target_column: Optional[str] = None,
                       steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run complete preprocessing pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Target column for supervised preprocessing
            steps: Specific preprocessing steps to run
            
        Returns:
            Preprocessing results including cleaned data and metadata
        """
        self.logger.info("Starting data preprocessing pipeline")
        
        if steps is None:
            steps = ['missing_values', 'outliers', 'scaling', 'feature_engineering']
        
        # Initialize results
        results = {
            'original_shape': data.shape,
            'steps_completed': [],
            'preprocessing_metadata': {},
            'data': data.copy()
        }
        
        # Step 1: Handle missing values
        if 'missing_values' in steps:
            self.logger.info("Step 1: Handling missing values")
            
            missing_analysis = self.missing_handler.analyze_missing_patterns(data)
            results['preprocessing_metadata']['missing_analysis'] = missing_analysis
            
            if missing_analysis['total_missing_values'] > 0:
                results['data'] = self.missing_handler.impute_missing_values(
                    results['data'], target_column
                )
            
            results['steps_completed'].append('missing_values')
        
        # Step 2: Handle outliers
        if 'outliers' in steps:
            self.logger.info("Step 2: Handling outliers")
            
            numeric_columns = results['data'].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                outlier_analysis = self.outlier_detector.detect_outliers(results['data'])
                results['preprocessing_metadata']['outlier_analysis'] = outlier_analysis
                
                # Handle outliers if found
                if outlier_analysis['summary']['total_outliers'] > 0:
                    results['data'] = self.outlier_detector.handle_outliers(
                        results['data'], method='clip'
                    )
            
            results['steps_completed'].append('outliers')
        
        # Step 3: Feature engineering
        if 'feature_engineering' in steps:
            self.logger.info("Step 3: Feature engineering")
            
            # Detect datetime columns
            datetime_columns = []
            for column in results['data'].columns:
                if pd.api.types.is_datetime64_any_dtype(results['data'][column]):
                    datetime_columns.append(column)
                else:
                    # Try to parse as datetime
                    try:
                        pd.to_datetime(results['data'][column].dropna().head(100), errors='raise')
                        datetime_columns.append(column)
                    except:
                        pass
            
            if datetime_columns:
                results['data'] = self.feature_engineer.create_datetime_features(
                    results['data'], datetime_columns
                )
            
            # Encode categorical features
            categorical_columns = results['data'].select_dtypes(include=['object']).columns.tolist()
            if categorical_columns:
                results['data'] = self.feature_engineer.encode_categorical_features(
                    results['data'], categorical_columns, encoding_method='label'
                )
            
            results['steps_completed'].append('feature_engineering')
        
        # Step 4: Scaling
        if 'scaling' in steps:
            self.logger.info("Step 4: Scaling features")
            
            numeric_columns = results['data'].select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_columns:
                numeric_columns.remove(target_column)  # Don't scale target
            
            if numeric_columns:
                results['data'] = self.scaler.fit_transform(results['data'], numeric_columns)
            
            results['steps_completed'].append('scaling')
        
        # Final metadata
        results['final_shape'] = results['data'].shape
        results['shape_change'] = {
            'rows_change': results['final_shape'][0] - results['original_shape'][0],
            'columns_change': results['final_shape'][1] - results['original_shape'][1]
        }
        
        self.logger.info(f"Preprocessing completed. Shape: {results['original_shape']} -> {results['final_shape']}")
        
        return results
    
    def get_preprocessing_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable preprocessing summary.
        
        Args:
            results: Preprocessing results
            
        Returns:
            Summary string
        """
        summary_lines = [
            "Data Preprocessing Summary",
            "=" * 30,
            f"Original shape: {results['original_shape']}",
            f"Final shape: {results['final_shape']}",
            f"Steps completed: {', '.join(results['steps_completed'])}",
            ""
        ]
        
        if 'missing_analysis' in results['preprocessing_metadata']:
            missing = results['preprocessing_metadata']['missing_analysis']
            summary_lines.extend([
                "Missing Values:",
                f"  Total missing: {missing['total_missing_values']}",
                f"  Percentage: {missing['overall_missing_percentage']:.2f}%",
                ""
            ])
        
        if 'outlier_analysis' in results['preprocessing_metadata']:
            outliers = results['preprocessing_metadata']['outlier_analysis']
            summary_lines.extend([
                "Outliers:",
                f"  Total outliers: {outliers['summary']['total_outliers']}",
                f"  Columns affected: {outliers['summary']['columns_with_outliers']}",
                ""
            ])
        
        return "\n".join(summary_lines)