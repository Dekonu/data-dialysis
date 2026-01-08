"""Change Detection Service.

This service detects field-level changes between existing and new records
using vectorized pandas operations for performance.

Security Impact:
    - Compares values that may contain PII (already redacted)
    - Change events are logged for audit trail
    - Handles sensitive data appropriately

Architecture:
    - Pure domain service with no infrastructure dependencies
    - Uses pandas for vectorized operations (performance-critical)
    - Returns domain models (ChangeEvent) for use by adapters
"""

import json
import logging
from typing import Any, List, Optional

import pandas as pd

from src.domain.cdc_models import ChangeEvent

logger = logging.getLogger(__name__)


class ChangeDetector:
    """Service for detecting field-level changes between records.
    
    This service uses vectorized pandas operations to efficiently compare
    large batches of records, identifying which fields have changed.
    """
    
    def __init__(self, ingestion_id: Optional[str] = None, source_adapter: Optional[str] = None):
        """Initialize change detector.
        
        Parameters:
            ingestion_id: ID of the current ingestion run
            source_adapter: Source adapter identifier
        """
        self.ingestion_id = ingestion_id
        self.source_adapter = source_adapter
    
    def detect_changes_vectorized(
        self,
        merged_df: pd.DataFrame,
        table_name: str,
        primary_key: str
    ) -> pd.DataFrame:
        """Detect field-level changes using vectorized pandas operations.
        
        This method is optimized for performance with large chunks (10k-50k rows).
        It uses pandas vectorized operations instead of row-by-row loops.
        
        Parameters:
            merged_df: DataFrame with merged old and new records (from pandas merge)
                      Should have columns with '_old' and '_new' suffixes
            table_name: Name of the table (patients, encounters, observations)
            primary_key: Name of the primary key column
            
        Returns:
            DataFrame with columns: table_name, record_id, field_name, old_value, new_value, change_type
        """
        if merged_df.empty:
            return pd.DataFrame(columns=[
                'table_name', 'record_id', 'field_name', 'old_value', 'new_value', 'change_type'
            ])
        
        changes_list = []
        
        # Get all data columns (exclude primary key and merge indicator)
        all_columns = set(merged_df.columns)
        
        # Find columns that have both _old and _new versions
        old_columns = {col.replace('_old', '') for col in all_columns if col.endswith('_old')}
        new_columns = {col.replace('_new', '') for col in all_columns if col.endswith('_new')}
        data_columns = old_columns & new_columns  # Columns present in both
        
        # Remove primary key from comparison
        data_columns.discard(primary_key)
        
        # Vectorized comparison for each field
        for col in data_columns:
            old_col = f"{col}_old"
            new_col = f"{col}_new"
            
            if old_col not in merged_df.columns or new_col not in merged_df.columns:
                continue
            
            # Get series for comparison
            old_series = merged_df[old_col]
            new_series = merged_df[new_col]
            
            # Normalize for comparison (handle NaN, None, arrays)
            old_normalized = self._normalize_for_comparison(old_series)
            new_normalized = self._normalize_for_comparison(new_series)
            
            # Vectorized comparison
            changed_mask = old_normalized != new_normalized
            
            if changed_mask.any():
                # Extract changed rows
                changed_rows = merged_df[changed_mask]
                
                # Build change records
                for idx, row in changed_rows.iterrows():
                    old_value = row[old_col]
                    new_value = row[new_col]
                    
                    changes_list.append({
                        'table_name': table_name,
                        'record_id': str(row[primary_key]),
                        'field_name': col,
                        'old_value': old_value,
                        'new_value': new_value,
                        'change_type': 'UPDATE'
                    })
        
        return pd.DataFrame(changes_list)
    
    def _normalize_for_comparison(self, series: pd.Series) -> pd.Series:
        """Normalize a pandas Series for comparison.
        
        Handles NaN, None, arrays, and other complex types by converting
        them to comparable string representations.
        
        Parameters:
            series: pandas Series to normalize
            
        Returns:
            Normalized Series with string values
        """
        # Handle NaN/None first
        normalized = series.fillna('__NULL__')
        
        # For object dtype (may contain lists, dicts, etc.)
        if normalized.dtype == 'object':
            def normalize_value(x):
                if x == '__NULL__':
                    return '__NULL__'
                if isinstance(x, (list, dict)):
                    try:
                        return json.dumps(x, sort_keys=True)
                    except (TypeError, ValueError):
                        return str(x)
                return str(x)
            
            normalized = normalized.apply(normalize_value)
        else:
            # For numeric types, convert to string
            normalized = normalized.astype(str)
        
        return normalized
    
    def changes_df_to_events(
        self,
        changes_df: pd.DataFrame
    ) -> List[ChangeEvent]:
        """Convert changes DataFrame to list of ChangeEvent objects.
        
        Parameters:
            changes_df: DataFrame with change information
            
        Returns:
            List of ChangeEvent objects
        """
        if changes_df.empty:
            return []
        
        events = []
        for _, row in changes_df.iterrows():
            try:
                event = ChangeEvent(
                    table_name=row['table_name'],
                    record_id=row['record_id'],
                    field_name=row['field_name'],
                    old_value=row.get('old_value'),
                    new_value=row.get('new_value'),
                    change_type=row.get('change_type', 'UPDATE'),
                    ingestion_id=self.ingestion_id,
                    source_adapter=self.source_adapter
                )
                events.append(event)
            except Exception as e:
                logger.warning(
                    f"Failed to create ChangeEvent from row: {e}. "
                    f"Row: {row.to_dict()}"
                )
                continue
        
        return events
    
    def generate_insert_changes(
        self,
        df: pd.DataFrame,
        table_name: str,
        primary_key: str
    ) -> List[ChangeEvent]:
        """Generate change events for new records (INSERTs).
        
        For new records, all fields are considered as INSERTs.
        
        Parameters:
            df: DataFrame with new records
            table_name: Name of the table
            primary_key: Name of the primary key column
            
        Returns:
            List of ChangeEvent objects for all fields in new records
        """
        if df.empty:
            return []
        
        events = []
        for _, row in df.iterrows():
            record_id = str(row[primary_key])
            
            # Create INSERT event for each field (except primary key)
            for col in df.columns:
                if col == primary_key:
                    continue
                
                value = row[col]
                
                try:
                    event = ChangeEvent(
                        table_name=table_name,
                        record_id=record_id,
                        field_name=col,
                        old_value=None,  # No old value for INSERTs
                        new_value=value,
                        change_type='INSERT',
                        ingestion_id=self.ingestion_id,
                        source_adapter=self.source_adapter
                    )
                    events.append(event)
                except Exception as e:
                    logger.warning(
                        f"Failed to create INSERT ChangeEvent for {col}: {e}"
                    )
                    continue
        
        return events
    
    @staticmethod
    def values_equal(old: Any, new: Any) -> bool:
        """Compare two values accounting for NaN, None, arrays, etc.
        
        This is a utility method for single-value comparison.
        For bulk operations, use detect_changes_vectorized() instead.
        
        Parameters:
            old: Old value
            new: New value
            
        Returns:
            True if values are equal, False otherwise
        """
        # Handle arrays/lists first (before NaN check, as pd.isna() doesn't work on lists)
        if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
            return old == new
        if isinstance(old, (list, tuple)) or isinstance(new, (list, tuple)):
            return False  # One is a list/tuple, the other is not
        
        # Handle dicts
        if isinstance(old, dict) and isinstance(new, dict):
            return old == new
        if isinstance(old, dict) or isinstance(new, dict):
            return False  # One is a dict, the other is not
        
        # Handle None/NaN (only for non-list/dict types)
        try:
            if pd.isna(old) and pd.isna(new):
                return True
            if pd.isna(old) or pd.isna(new):
                return False
        except (ValueError, TypeError):
            # pd.isna() may fail for some types (e.g., lists), skip NaN check
            pass
        
        # Handle other types
        return old == new
