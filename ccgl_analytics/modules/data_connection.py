"""
CCGL Analytics - Data Connection Module
Comprehensive database connectivity and data quality assessment
"""

import os
import time
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Optional, Union, Tuple
from contextlib import contextmanager
from urllib.parse import quote_plus
import warnings

# Database connectivity
try:
    from sqlalchemy import create_engine, text, pool, event
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from ..utils.logger import get_logger, LoggerMixin

class DatabaseConnectionPool(LoggerMixin):
    """Database connection pool manager."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize connection pool.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.engine = None
        self.session_factory = None
        self._connection_string = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize database connection pool."""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for database connectivity")
        
        db_config = self.config.get('database', {}).get('primary', {})
        pool_config = self.config.get('database', {}).get('pool', {})
        
        # Build connection string
        db_type = db_config.get('type', 'mysql')
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 3306)
        username = db_config.get('username', '')
        password = db_config.get('password', '')
        database = db_config.get('database', '')
        charset = db_config.get('charset', 'utf8mb4')
        
        if db_type == 'mysql':
            if not PYMYSQL_AVAILABLE:
                raise ImportError("pymysql is required for MySQL connectivity")
            
            self._connection_string = (
                f"mysql+pymysql://{username}:{quote_plus(password)}"
                f"@{host}:{port}/{database}?charset={charset}"
            )
        
        elif db_type == 'postgresql':
            if not PSYCOPG2_AVAILABLE:
                raise ImportError("psycopg2 is required for PostgreSQL connectivity")
            
            self._connection_string = (
                f"postgresql+psycopg2://{username}:{quote_plus(password)}"
                f"@{host}:{port}/{database}"
            )
        
        elif db_type == 'sqlite':
            database_path = database if database else 'ccgl_analytics.db'
            self._connection_string = f"sqlite:///{database_path}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self._connection_string,
            pool_size=pool_config.get('size', 10),
            max_overflow=pool_config.get('max_overflow', 20),
            pool_timeout=pool_config.get('timeout', 30),
            pool_recycle=pool_config.get('recycle', 3600),
            pool_pre_ping=pool_config.get('pre_ping', True),
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)
        
        # Add connection pool event listeners
        self._setup_event_listeners()
        
        self.logger.info(f"Database connection pool initialized for {db_type}")
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            self.logger.debug("New database connection established")
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            self.logger.debug("Connection checked out from pool")
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Handle connection checkin to pool."""
            self.logger.debug("Connection checked in to pool")
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool.
        
        Yields:
            Database connection
        """
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    @contextmanager
    def get_session(self):
        """Get database session.
        
        Yields:
            Database session
        """
        session = None
        try:
            session = self.session_factory()
            yield session
            session.commit()
        except Exception as e:
            if session:
                session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def test_connection(self) -> bool:
        """Test database connection.
        
        Returns:
            True if connection successful
        """
        try:
            with self.get_connection() as conn:
                # Execute a simple query
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            self.logger.info("Database connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status.
        
        Returns:
            Pool status information
        """
        pool = self.engine.pool
        return {
            'pool_size': pool.size(),
            'connections_checked_out': pool.checkedout(),
            'connections_checked_in': pool.checkedin(),
            'connections_invalid': pool.invalid(),
            'overflow_connections': pool.overflow(),
        }

class DataQualityAssessor(LoggerMixin):
    """Data quality assessment and scoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data quality assessor.
        
        Args:
            config: Quality assessment configuration
        """
        self.config = config or {}
        self.quality_config = self.config.get('analysis', {}).get('data_quality', {})
    
    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Quality assessment results
        """
        self.logger.info(f"Assessing data quality for {data.shape[0]} rows, {data.shape[1]} columns")
        
        start_time = time.time()
        
        # Basic information
        assessment = {
            'metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'shape': data.shape,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict()
            }
        }
        
        # Completeness assessment
        assessment['completeness'] = self._assess_completeness(data)
        
        # Uniqueness assessment  
        assessment['uniqueness'] = self._assess_uniqueness(data)
        
        # Validity assessment
        assessment['validity'] = self._assess_validity(data)
        
        # Consistency assessment
        assessment['consistency'] = self._assess_consistency(data)
        
        # Overall quality score
        assessment['overall_score'] = self._calculate_overall_score(assessment)
        
        # Performance metrics
        assessment['assessment_time_seconds'] = time.time() - start_time
        
        self.logger.info(f"Data quality assessment completed. Score: {assessment['overall_score']:.2f}")
        
        return assessment
    
    def _assess_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Completeness assessment
        """
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        
        # Per-column analysis
        column_completeness = {}
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            column_completeness[column] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_count / len(data) * 100),
                'completeness_score': float(1 - missing_count / len(data))
            }
        
        # Overall completeness
        overall_completeness = 1 - (missing_cells / total_cells)
        
        return {
            'overall_completeness': float(overall_completeness),
            'missing_cells_total': int(missing_cells),
            'missing_percentage': float(missing_cells / total_cells * 100),
            'column_analysis': column_completeness,
            'columns_with_missing': [col for col, info in column_completeness.items() 
                                   if info['missing_count'] > 0]
        }
    
    def _assess_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Uniqueness assessment
        """
        # Overall duplicate analysis
        duplicate_rows = data.duplicated().sum()
        
        # Per-column uniqueness
        column_uniqueness = {}
        for column in data.columns:
            unique_count = data[column].nunique()
            total_count = len(data) - data[column].isnull().sum()
            
            column_uniqueness[column] = {
                'unique_count': int(unique_count),
                'total_non_null': int(total_count),
                'uniqueness_ratio': float(unique_count / total_count) if total_count > 0 else 0,
                'duplicate_count': int(total_count - unique_count)
            }
        
        return {
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': float(duplicate_rows / len(data) * 100),
            'unique_rows': int(len(data) - duplicate_rows),
            'overall_uniqueness': float(1 - duplicate_rows / len(data)),
            'column_analysis': column_uniqueness
        }
    
    def _assess_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Validity assessment
        """
        column_validity = {}
        
        for column in data.columns:
            column_data = data[column].dropna()
            
            validity_info = {
                'data_type': str(data[column].dtype),
                'valid_count': len(column_data),
                'invalid_patterns': []
            }
            
            # Type-specific validity checks
            if pd.api.types.is_numeric_dtype(data[column]):
                # Numeric validity
                infinite_count = np.isinf(column_data).sum()
                validity_info['infinite_values'] = int(infinite_count)
                
                if infinite_count > 0:
                    validity_info['invalid_patterns'].append('infinite_values')
            
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                # Datetime validity
                validity_info['date_range'] = {
                    'min_date': str(column_data.min()),
                    'max_date': str(column_data.max())
                }
            
            elif pd.api.types.is_string_dtype(data[column]) or pd.api.types.is_object_dtype(data[column]):
                # String validity
                empty_strings = (column_data == '').sum()
                validity_info['empty_strings'] = int(empty_strings)
                
                if empty_strings > 0:
                    validity_info['invalid_patterns'].append('empty_strings')
            
            # Calculate validity score
            invalid_count = sum([
                validity_info.get('infinite_values', 0),
                validity_info.get('empty_strings', 0)
            ])
            
            validity_info['validity_score'] = float(
                1 - invalid_count / len(column_data) if len(column_data) > 0 else 1
            )
            
            column_validity[column] = validity_info
        
        # Overall validity score
        overall_validity = np.mean([info['validity_score'] for info in column_validity.values()])
        
        return {
            'overall_validity': float(overall_validity),
            'column_analysis': column_validity
        }
    
    def _assess_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Consistency assessment
        """
        consistency_checks = {
            'data_type_consistency': self._check_dtype_consistency(data),
            'format_consistency': self._check_format_consistency(data),
            'range_consistency': self._check_range_consistency(data)
        }
        
        # Overall consistency score
        consistency_scores = [check.get('score', 1.0) for check in consistency_checks.values()]
        overall_consistency = np.mean(consistency_scores)
        
        return {
            'overall_consistency': float(overall_consistency),
            'checks': consistency_checks
        }
    
    def _check_dtype_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data type consistency.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Data type consistency results
        """
        # Check for mixed types in object columns
        mixed_type_columns = []
        
        for column in data.select_dtypes(include=['object']).columns:
            try:
                # Try to infer a more specific type
                inferred_type = pd.api.types.infer_dtype(data[column].dropna())
                if inferred_type == 'mixed':
                    mixed_type_columns.append(column)
            except:
                pass
        
        score = 1.0 - (len(mixed_type_columns) / len(data.columns))
        
        return {
            'score': score,
            'mixed_type_columns': mixed_type_columns,
            'total_columns': len(data.columns)
        }
    
    def _check_format_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check format consistency.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Format consistency results
        """
        format_issues = []
        
        # Check string columns for format consistency
        for column in data.select_dtypes(include=['object', 'string']).columns:
            column_data = data[column].dropna().astype(str)
            
            if len(column_data) == 0:
                continue
            
            # Check for common format patterns
            if column.lower() in ['email', 'email_address']:
                # Email format check (simplified)
                email_pattern = column_data.str.contains('@', na=False)
                if not email_pattern.all():
                    format_issues.append(f"{column}: inconsistent email format")
            
            elif column.lower() in ['phone', 'phone_number']:
                # Phone format check (simplified)
                # Check for consistent length or pattern
                lengths = column_data.str.len()
                if lengths.std() > 2:  # Allow some variation
                    format_issues.append(f"{column}: inconsistent phone format")
        
        score = 1.0 - (len(format_issues) / max(1, len(data.select_dtypes(include=['object', 'string']).columns)))
        
        return {
            'score': score,
            'format_issues': format_issues
        }
    
    def _check_range_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check range consistency.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Range consistency results
        """
        range_issues = []
        
        # Check numeric columns for reasonable ranges
        for column in data.select_dtypes(include=['number']).columns:
            column_data = data[column].dropna()
            
            if len(column_data) == 0:
                continue
            
            # Check for extreme outliers (beyond 3 standard deviations)
            if len(column_data) > 10:  # Need sufficient data
                mean_val = column_data.mean()
                std_val = column_data.std()
                
                if std_val > 0:
                    outliers = np.abs(column_data - mean_val) > 3 * std_val
                    outlier_percentage = outliers.sum() / len(column_data)
                    
                    if outlier_percentage > 0.05:  # More than 5% outliers
                        range_issues.append(f"{column}: {outlier_percentage:.1%} extreme outliers")
        
        score = 1.0 - (len(range_issues) / max(1, len(data.select_dtypes(include=['number']).columns)))
        
        return {
            'score': score,
            'range_issues': range_issues
        }
    
    def _calculate_overall_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate overall quality score.
        
        Args:
            assessment: Quality assessment data
            
        Returns:
            Overall quality score (0-1)
        """
        weights = self.quality_config.get('quality_score_weights', {
            'completeness': 0.3,
            'uniqueness': 0.2,
            'validity': 0.3,
            'consistency': 0.2
        })
        
        # Get component scores
        completeness_score = assessment.get('completeness', {}).get('overall_completeness', 1.0)
        uniqueness_score = assessment.get('uniqueness', {}).get('overall_uniqueness', 1.0)
        validity_score = assessment.get('validity', {}).get('overall_validity', 1.0)
        consistency_score = assessment.get('consistency', {}).get('overall_consistency', 1.0)
        
        # Calculate weighted score
        overall_score = (
            completeness_score * weights.get('completeness', 0.3) +
            uniqueness_score * weights.get('uniqueness', 0.2) +
            validity_score * weights.get('validity', 0.3) +
            consistency_score * weights.get('consistency', 0.2)
        )
        
        return float(overall_score)

class DataConnectionManager(LoggerMixin):
    """Main data connection and quality management class."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize data connection manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.connection_pool = None
        self.quality_assessor = DataQualityAssessor(self.config)
        
        # Initialize connection pool if database is configured
        if self._is_database_configured():
            try:
                self.connection_pool = DatabaseConnectionPool(self.config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize database connection pool: {e}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not config_file or not os.path.exists(config_file):
            self.logger.warning("Configuration file not found, using defaults")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _is_database_configured(self) -> bool:
        """Check if database is configured.
        
        Returns:
            True if database configuration exists
        """
        db_config = self.config.get('database', {}).get('primary', {})
        return bool(db_config.get('host') and db_config.get('database'))
    
    def query_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results as DataFrame
        """
        if not self.connection_pool:
            raise ValueError("Database connection not available")
        
        self.logger.info(f"Executing query: {query[:100]}...")
        
        try:
            with self.connection_pool.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            self.logger.info(f"Query returned {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def load_data_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional parameters for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading data from file: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_ext == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Quality assessment results
        """
        return self.quality_assessor.assess_data_quality(data)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information.
        
        Returns:
            Connection status
        """
        status = {
            'database_configured': self._is_database_configured(),
            'connection_pool_available': self.connection_pool is not None,
            'connection_test_passed': False
        }
        
        if self.connection_pool:
            status['connection_test_passed'] = self.connection_pool.test_connection()
            status['pool_status'] = self.connection_pool.get_pool_status()
        
        return status