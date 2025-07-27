"""
CCGL Analytics - Relationship Extraction Module
Extract and analyze relationships between data entities
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import itertools

from ..utils.logger import get_logger, LoggerMixin

class RelationshipExtraction(LoggerMixin):
    """Extract and analyze relationships between data entities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize relationship extraction.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.extracted_relationships = {}
        self.relationship_strength_threshold = 0.3
        
    def extract_correlations(self, data: pd.DataFrame, threshold: float = 0.3) -> Dict[str, Any]:
        """Extract correlation relationships between numeric columns.
        
        Args:
            data: Input DataFrame
            threshold: Minimum correlation threshold
            
        Returns:
            Correlation relationships
        """
        self.logger.info("Extracting correlation relationships")
        
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return {
                    'status': 'insufficient_data',
                    'message': 'Need at least 2 numeric columns for correlation analysis'
                }
            
            # Calculate correlation matrix
            correlation_matrix = numeric_data.corr()
            
            # Extract significant correlations
            relationships = []
            
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicates and self-correlation
                        corr_value = correlation_matrix.loc[col1, col2]
                        
                        if abs(corr_value) >= threshold:
                            relationship = {
                                'entity1': col1,
                                'entity2': col2,
                                'relationship_type': 'correlation',
                                'strength': abs(corr_value),
                                'direction': 'positive' if corr_value > 0 else 'negative',
                                'correlation_value': float(corr_value),
                                'significance': self._assess_correlation_significance(abs(corr_value))
                            }
                            relationships.append(relationship)
            
            # Sort by strength
            relationships.sort(key=lambda x: x['strength'], reverse=True)
            
            result = {
                'status': 'success',
                'relationship_type': 'correlations',
                'total_relationships': len(relationships),
                'relationships': relationships,
                'correlation_matrix': correlation_matrix.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.extracted_relationships['correlations'] = result
            
            self.logger.info(f"Extracted {len(relationships)} correlation relationships")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract correlations: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def extract_categorical_associations(self, data: pd.DataFrame, 
                                       chi_square_threshold: float = 0.05) -> Dict[str, Any]:
        """Extract associations between categorical variables.
        
        Args:
            data: Input DataFrame
            chi_square_threshold: P-value threshold for chi-square test
            
        Returns:
            Categorical associations
        """
        self.logger.info("Extracting categorical associations")
        
        try:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_columns) < 2:
                return {
                    'status': 'insufficient_data',
                    'message': 'Need at least 2 categorical columns for association analysis'
                }
            
            relationships = []
            
            # Check associations between all pairs of categorical columns
            for col1, col2 in itertools.combinations(categorical_columns, 2):
                try:
                    # Create contingency table
                    contingency_table = pd.crosstab(data[col1], data[col2])
                    
                    # Calculate association strength using Cramér's V
                    cramers_v = self._calculate_cramers_v(contingency_table)
                    
                    # Perform chi-square test (simplified)
                    chi2_stat, p_value = self._chi_square_test_simplified(contingency_table)
                    
                    if p_value <= chi_square_threshold:
                        relationship = {
                            'entity1': col1,
                            'entity2': col2,
                            'relationship_type': 'categorical_association',
                            'strength': cramers_v,
                            'chi_square_statistic': chi2_stat,
                            'p_value': p_value,
                            'significance': 'significant' if p_value <= 0.05 else 'moderate',
                            'contingency_table': contingency_table.to_dict()
                        }
                        relationships.append(relationship)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to analyze association between {col1} and {col2}: {e}")
                    continue
            
            # Sort by strength
            relationships.sort(key=lambda x: x['strength'], reverse=True)
            
            result = {
                'status': 'success',
                'relationship_type': 'categorical_associations',
                'total_relationships': len(relationships),
                'relationships': relationships,
                'timestamp': datetime.now().isoformat()
            }
            
            self.extracted_relationships['categorical_associations'] = result
            
            self.logger.info(f"Extracted {len(relationships)} categorical associations")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract categorical associations: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def extract_temporal_relationships(self, data: pd.DataFrame, 
                                     date_column: str, 
                                     value_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract temporal relationships and trends.
        
        Args:
            data: Input DataFrame
            date_column: Name of the date column
            value_columns: List of value columns to analyze
            
        Returns:
            Temporal relationships
        """
        self.logger.info("Extracting temporal relationships")
        
        try:
            if date_column not in data.columns:
                return {
                    'status': 'error',
                    'error': f'Date column {date_column} not found'
                }
            
            # Convert date column
            data_copy = data.copy()
            data_copy[date_column] = pd.to_datetime(data_copy[date_column], errors='coerce')
            
            # Remove rows with invalid dates
            data_copy = data_copy.dropna(subset=[date_column])
            
            if data_copy.empty:
                return {
                    'status': 'error',
                    'error': 'No valid dates found'
                }
            
            # Sort by date
            data_copy = data_copy.sort_values(date_column)
            
            if value_columns is None:
                value_columns = data_copy.select_dtypes(include=[np.number]).columns.tolist()
            
            relationships = []
            
            for column in value_columns:
                if column in data_copy.columns:
                    # Calculate trend
                    trend_analysis = self._analyze_trend(data_copy[date_column], data_copy[column])
                    
                    # Calculate seasonality (simplified)
                    seasonality_analysis = self._analyze_seasonality(data_copy[date_column], data_copy[column])
                    
                    relationship = {
                        'entity': column,
                        'relationship_type': 'temporal',
                        'trend': trend_analysis,
                        'seasonality': seasonality_analysis,
                        'date_range': {
                            'start': data_copy[date_column].min().isoformat(),
                            'end': data_copy[date_column].max().isoformat()
                        },
                        'data_points': len(data_copy)
                    }
                    relationships.append(relationship)
            
            result = {
                'status': 'success',
                'relationship_type': 'temporal',
                'total_relationships': len(relationships),
                'relationships': relationships,
                'timestamp': datetime.now().isoformat()
            }
            
            self.extracted_relationships['temporal'] = result
            
            self.logger.info(f"Extracted temporal relationships for {len(relationships)} columns")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract temporal relationships: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def extract_hierarchical_relationships(self, data: pd.DataFrame, 
                                         hierarchy_columns: List[str]) -> Dict[str, Any]:
        """Extract hierarchical relationships between columns.
        
        Args:
            data: Input DataFrame
            hierarchy_columns: List of columns representing hierarchy levels
            
        Returns:
            Hierarchical relationships
        """
        self.logger.info("Extracting hierarchical relationships")
        
        try:
            # Validate columns exist
            missing_columns = [col for col in hierarchy_columns if col not in data.columns]
            if missing_columns:
                return {
                    'status': 'error',
                    'error': f'Missing columns: {missing_columns}'
                }
            
            relationships = []
            hierarchy_stats = {}
            
            # Analyze each level of hierarchy
            for i, column in enumerate(hierarchy_columns):
                level_stats = {
                    'level': i + 1,
                    'column': column,
                    'unique_values': data[column].nunique(),
                    'total_values': len(data[column].dropna()),
                    'sample_values': data[column].dropna().unique()[:10].tolist()
                }
                hierarchy_stats[column] = level_stats
                
                # Analyze relationships between consecutive levels
                if i > 0:
                    parent_column = hierarchy_columns[i - 1]
                    child_column = column
                    
                    # Calculate parent-child relationships
                    parent_child_mapping = data.groupby(parent_column)[child_column].nunique().to_dict()
                    
                    relationship = {
                        'parent': parent_column,
                        'child': child_column,
                        'relationship_type': 'hierarchical',
                        'parent_child_mapping': dict(list(parent_child_mapping.items())[:20]),  # Limit size
                        'avg_children_per_parent': np.mean(list(parent_child_mapping.values())),
                        'max_children_per_parent': max(parent_child_mapping.values()),
                        'min_children_per_parent': min(parent_child_mapping.values())
                    }
                    relationships.append(relationship)
            
            result = {
                'status': 'success',
                'relationship_type': 'hierarchical',
                'hierarchy_levels': len(hierarchy_columns),
                'hierarchy_stats': hierarchy_stats,
                'relationships': relationships,
                'timestamp': datetime.now().isoformat()
            }
            
            self.extracted_relationships['hierarchical'] = result
            
            self.logger.info(f"Extracted hierarchical relationships for {len(hierarchy_columns)} levels")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract hierarchical relationships: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_all_relationships(self) -> Dict[str, Any]:
        """Get all extracted relationships.
        
        Returns:
            All relationship data
        """
        return {
            'total_relationship_types': len(self.extracted_relationships),
            'relationships': self.extracted_relationships,
            'summary': self._generate_relationship_summary(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _assess_correlation_significance(self, correlation_value: float) -> str:
        """Assess significance of correlation value.
        
        Args:
            correlation_value: Absolute correlation value
            
        Returns:
            Significance level
        """
        if correlation_value >= 0.7:
            return 'strong'
        elif correlation_value >= 0.5:
            return 'moderate'
        elif correlation_value >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _calculate_cramers_v(self, contingency_table: pd.DataFrame) -> float:
        """Calculate Cramér's V statistic for categorical association.
        
        Args:
            contingency_table: Contingency table
            
        Returns:
            Cramér's V value
        """
        try:
            chi2 = ((contingency_table.values - np.outer(contingency_table.sum(axis=1), 
                                                         contingency_table.sum(axis=0)) / contingency_table.sum().sum()) ** 2 / 
                   np.outer(contingency_table.sum(axis=1), contingency_table.sum(axis=0)) * contingency_table.sum().sum()).sum()
            
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            
            if min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return float(cramers_v)
            
        except:
            return 0.0
    
    def _chi_square_test_simplified(self, contingency_table: pd.DataFrame) -> Tuple[float, float]:
        """Simplified chi-square test.
        
        Args:
            contingency_table: Contingency table
            
        Returns:
            Chi-square statistic and p-value
        """
        try:
            observed = contingency_table.values
            row_totals = contingency_table.sum(axis=1)
            col_totals = contingency_table.sum(axis=0)
            total = contingency_table.sum().sum()
            
            expected = np.outer(row_totals, col_totals) / total
            
            # Avoid division by zero
            expected = np.where(expected == 0, 1e-10, expected)
            
            chi2_stat = ((observed - expected) ** 2 / expected).sum()
            
            # Simplified p-value approximation
            df = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)
            
            # Very simplified p-value calculation (not accurate, just for demonstration)
            if chi2_stat > df * 3:
                p_value = 0.001
            elif chi2_stat > df * 2:
                p_value = 0.01
            elif chi2_stat > df:
                p_value = 0.05
            else:
                p_value = 0.1
            
            return float(chi2_stat), float(p_value)
            
        except:
            return 0.0, 1.0
    
    def _analyze_trend(self, dates: pd.Series, values: pd.Series) -> Dict[str, Any]:
        """Analyze trend in time series data.
        
        Args:
            dates: Date series
            values: Value series
            
        Returns:
            Trend analysis
        """
        try:
            # Simple linear trend analysis
            numeric_dates = pd.to_numeric(dates)
            
            # Calculate correlation between dates and values
            trend_correlation = np.corrcoef(numeric_dates, values)[0, 1]
            
            # Calculate percentage change
            first_value = values.iloc[0]
            last_value = values.iloc[-1]
            
            if first_value != 0:
                percent_change = ((last_value - first_value) / first_value) * 100
            else:
                percent_change = 0
            
            return {
                'direction': 'increasing' if trend_correlation > 0 else 'decreasing',
                'strength': abs(trend_correlation),
                'correlation': float(trend_correlation),
                'percent_change': float(percent_change),
                'significance': 'strong' if abs(trend_correlation) > 0.7 else 'moderate' if abs(trend_correlation) > 0.3 else 'weak'
            }
            
        except:
            return {
                'direction': 'unknown',
                'strength': 0,
                'correlation': 0,
                'percent_change': 0,
                'significance': 'none'
            }
    
    def _analyze_seasonality(self, dates: pd.Series, values: pd.Series) -> Dict[str, Any]:
        """Analyze seasonality in time series data.
        
        Args:
            dates: Date series
            values: Value series
            
        Returns:
            Seasonality analysis
        """
        try:
            # Simple seasonality analysis based on month
            data_with_month = pd.DataFrame({
                'date': dates,
                'value': values,
                'month': dates.dt.month
            })
            
            monthly_stats = data_with_month.groupby('month')['value'].agg(['mean', 'std', 'count']).to_dict()
            
            # Calculate coefficient of variation across months
            monthly_means = list(monthly_stats['mean'].values())
            cv = np.std(monthly_means) / np.mean(monthly_means) if np.mean(monthly_means) != 0 else 0
            
            return {
                'has_seasonality': cv > 0.1,
                'coefficient_of_variation': float(cv),
                'monthly_patterns': monthly_stats,
                'peak_month': max(monthly_stats['mean'], key=monthly_stats['mean'].get),
                'low_month': min(monthly_stats['mean'], key=monthly_stats['mean'].get)
            }
            
        except:
            return {
                'has_seasonality': False,
                'coefficient_of_variation': 0,
                'monthly_patterns': {},
                'peak_month': None,
                'low_month': None
            }
    
    def _generate_relationship_summary(self) -> Dict[str, Any]:
        """Generate summary of all relationships.
        
        Returns:
            Relationship summary
        """
        summary = {
            'total_relationship_types': len(self.extracted_relationships),
            'relationship_counts': {},
            'strongest_relationships': []
        }
        
        # Count relationships by type
        for rel_type, rel_data in self.extracted_relationships.items():
            if rel_data.get('status') == 'success':
                summary['relationship_counts'][rel_type] = rel_data.get('total_relationships', 0)
        
        # Find strongest relationships across all types
        all_relationships = []
        for rel_type, rel_data in self.extracted_relationships.items():
            if rel_data.get('status') == 'success' and 'relationships' in rel_data:
                for rel in rel_data['relationships']:
                    rel['source_type'] = rel_type
                    all_relationships.append(rel)
        
        # Sort by strength and take top 10
        if all_relationships:
            sorted_relationships = sorted(all_relationships, 
                                        key=lambda x: x.get('strength', 0), 
                                        reverse=True)
            summary['strongest_relationships'] = sorted_relationships[:10]
        
        return summary