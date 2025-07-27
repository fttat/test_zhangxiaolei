"""
Data Connection Module for CCGL Analytics
Handles database connections and data loading
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False

from ..utils.logger import get_logger


class DataConnection:
    """
    Data connection handler for various data sources
    Supports MySQL, CSV, Excel files, and more
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data connection
        
        Args:
            config: Configuration dictionary containing connection parameters
        """
        self.config = config
        self.logger = get_logger("data_connection")
        self.engine = None
        self.connection = None
        
        # Initialize connection based on type
        self.connection_type = config.get('type', 'file')
        if self.connection_type == 'mysql':
            self._init_mysql_connection()
    
    def _init_mysql_connection(self):
        """Initialize MySQL database connection"""
        if not SQLALCHEMY_AVAILABLE:
            self.logger.error("SQLAlchemy not available. Please install: pip install sqlalchemy")
            return
        
        try:
            db_config = self.config.get('database', {})
            
            # Build connection string
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 3306)
            user = db_config.get('user', 'root')
            password = db_config.get('password', '')
            database = db_config.get('database', 'ccgl_warehouse')
            
            # Use PyMySQL driver if available
            driver = 'pymysql' if PYMYSQL_AVAILABLE else 'mysql+mysqlconnector'
            
            connection_string = f"mysql+{driver}://{user}:{password}@{host}:{port}/{database}"
            
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                self.logger.info(f"Successfully connected to MySQL database: {database}")
                
        except Exception as e:
            self.logger.error(f"Failed to connect to MySQL database: {str(e)}")
            self.engine = None
    
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various sources
        
        Args:
            source: Data source (table name, file path, etc.)
            **kwargs: Additional parameters for data loading
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            if self.connection_type == 'mysql':
                return self._load_from_mysql(source, **kwargs)
            elif self.connection_type == 'file':
                return self._load_from_file(source, **kwargs)
            else:
                raise ValueError(f"Unsupported connection type: {self.connection_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to load data from {source}: {str(e)}")
            raise
    
    def _load_from_mysql(self, table_name: str, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from MySQL database"""
        if not self.engine:
            raise RuntimeError("MySQL connection not established")
        
        try:
            if query:
                # Execute custom query
                df = pd.read_sql(query, self.engine, **kwargs)
                self.logger.info(f"Loaded data using custom query: {len(df)} rows")
            else:
                # Load entire table
                df = pd.read_sql_table(table_name, self.engine, **kwargs)
                self.logger.info(f"Loaded data from table '{table_name}': {len(df)} rows")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from MySQL: {str(e)}")
            raise
    
    def _load_from_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file (CSV, Excel, etc.)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_extension == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.logger.info(f"Loaded data from file '{file_path}': {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from file: {str(e)}")
            raise
    
    def save_data(self, df: pd.DataFrame, destination: str, **kwargs):
        """
        Save data to various destinations
        
        Args:
            df: DataFrame to save
            destination: Destination (table name, file path, etc.)
            **kwargs: Additional parameters for saving
        """
        try:
            if self.connection_type == 'mysql':
                self._save_to_mysql(df, destination, **kwargs)
            elif self.connection_type == 'file':
                self._save_to_file(df, destination, **kwargs)
            else:
                raise ValueError(f"Unsupported connection type: {self.connection_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to save data to {destination}: {str(e)}")
            raise
    
    def _save_to_mysql(self, df: pd.DataFrame, table_name: str, **kwargs):
        """Save data to MySQL database"""
        if not self.engine:
            raise RuntimeError("MySQL connection not established")
        
        try:
            # Default parameters
            if_exists = kwargs.get('if_exists', 'replace')
            index = kwargs.get('index', False)
            
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index, **kwargs)
            self.logger.info(f"Saved {len(df)} rows to table '{table_name}'")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to MySQL: {str(e)}")
            raise
    
    def _save_to_file(self, df: pd.DataFrame, file_path: str, **kwargs):
        """Save data to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_extension == '.json':
                df.to_json(file_path, **kwargs)
            elif file_extension == '.parquet':
                df.to_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.logger.info(f"Saved {len(df)} rows to file '{file_path}'")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to file: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a database table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing table information
        """
        if self.connection_type != 'mysql' or not self.engine:
            raise RuntimeError("MySQL connection required for table information")
        
        try:
            with self.engine.connect() as conn:
                # Get table schema
                schema_query = text(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = '{table_name}'
                    AND TABLE_SCHEMA = DATABASE()
                """)
                
                schema_result = conn.execute(schema_query)
                columns = [dict(row._mapping) for row in schema_result]
                
                # Get row count
                count_query = text(f"SELECT COUNT(*) as row_count FROM {table_name}")
                count_result = conn.execute(count_query)
                row_count = count_result.fetchone()[0]
                
                return {
                    'table_name': table_name,
                    'columns': columns,
                    'row_count': row_count,
                    'column_count': len(columns)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get table info: {str(e)}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query
        
        Args:
            query: SQL query to execute
            
        Returns:
            pandas.DataFrame: Query results
        """
        if self.connection_type != 'mysql' or not self.engine:
            raise RuntimeError("MySQL connection required for query execution")
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test the database connection
        
        Returns:
            bool: True if connection is successful
        """
        try:
            if self.connection_type == 'mysql' and self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return True
            else:
                # For file connections, just check if we can create a DataFrame
                test_data = pd.DataFrame({'test': [1, 2, 3]})
                return not test_data.empty
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")


def create_sample_data() -> pd.DataFrame:
    """
    Create sample warehouse data for testing
    
    Returns:
        pandas.DataFrame: Sample warehouse data
    """
    np.random.seed(42)
    n_records = 1000
    
    # Generate sample data
    data = {
        'product_id': [f'P{i:04d}' for i in range(1, n_records + 1)],
        'product_name': [f'Product {i}' for i in range(1, n_records + 1)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], n_records),
        'quantity': np.random.randint(1, 1000, n_records),
        'unit_price': np.round(np.random.uniform(5.0, 500.0, n_records), 2),
        'supplier_id': np.random.randint(1, 50, n_records),
        'warehouse_location': np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], n_records),
        'last_updated': pd.date_range('2024-01-01', periods=n_records, freq='h'),
        'stock_status': np.random.choice(['In Stock', 'Low Stock', 'Out of Stock'], n_records, p=[0.7, 0.25, 0.05])
    }
    
    df = pd.DataFrame(data)
    df['total_value'] = df['quantity'] * df['unit_price']
    
    return df