# Change Data Capture (CDC) Implementation Plan

## Overview

This document outlines the implementation plan for Change Data Capture (CDC) in the Data-Dialysis system. CDC will track field-level changes to records, enabling audit trails, compliance reporting, and efficient updates that only modify changed fields.

**⚡ Performance-Optimized for Large-Scale Batch Processing**: This plan is specifically designed to work efficiently with the existing chunk-based batch processing architecture that handles millions of records. Key optimizations include:
- **Bulk operations** at chunk level (10k-50k rows)
- **Vectorized pandas operations** for change detection
- **Async change logging** (non-blocking, similar to redaction logs)
- **Chunk-level caching** to avoid redundant database queries
- **Selective UPDATEs** with only changed fields
- **Target**: < 10% overhead on ingestion throughput

## Objectives

1. **Track Field-Level Changes**: Record what fields changed, when, and from/to values
2. **Efficient Updates**: Only update fields that have actually changed
3. **Audit Compliance**: Maintain immutable audit trail for HIPAA/GDPR compliance
4. **Performance**: Minimize database writes and maintain ingestion throughput
5. **Queryability**: Enable queries for change history, data lineage, and compliance reports

## Architecture

### 1. Database Schema Changes

#### 1.1 Change Audit Log Table

```sql
CREATE TABLE IF NOT EXISTS change_audit_log (
    change_id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    table_name VARCHAR(50) NOT NULL,
    record_id VARCHAR(50) NOT NULL,  -- Primary key of the record (patient_id, encounter_id, observation_id)
    field_name VARCHAR(100) NOT NULL,
    old_value TEXT,  -- JSON representation for complex types (arrays, etc.)
    new_value TEXT,  -- JSON representation for complex types
    change_type VARCHAR(20) NOT NULL,  -- 'INSERT', 'UPDATE', 'DELETE'
    changed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingestion_id VARCHAR(50),  -- Link to ingestion run
    source_adapter VARCHAR(50),
    changed_by VARCHAR(100),  -- System/user identifier (optional)
    
    -- Indexes for efficient querying
    INDEX idx_change_audit_table_record (table_name, record_id),
    INDEX idx_change_audit_timestamp (changed_at),
    INDEX idx_change_audit_ingestion (ingestion_id)
);
```

#### 1.2 Record Version Table (Optional - for point-in-time queries)

```sql
CREATE TABLE IF NOT EXISTS record_versions (
    version_id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    table_name VARCHAR(50) NOT NULL,
    record_id VARCHAR(50) NOT NULL,
    version_number INTEGER NOT NULL,
    record_data JSONB NOT NULL,  -- Full record snapshot
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingestion_id VARCHAR(50),
    
    UNIQUE (table_name, record_id, version_number),
    INDEX idx_record_versions_lookup (table_name, record_id, version_number)
);
```

### 2. Core Components

#### 2.1 Change Detection Service

**File**: `src/domain/services/change_detector.py`

**Responsibilities**:
- Compare old vs new record values
- Identify changed fields
- Generate change events
- Handle complex types (arrays, nested objects)

**Key Methods**:
```python
def detect_changes(
    old_record: dict,
    new_record: dict,
    table_name: str,
    record_id: str
) -> List[ChangeEvent]

def compare_field_value(
    field_name: str,
    old_value: Any,
    new_value: Any
) -> Optional[ChangeEvent]
```

#### 2.2 Change Audit Logger

**File**: `src/infrastructure/audit/change_audit_logger.py`

**Responsibilities**:
- Persist change events to `change_audit_log` table
- Batch change events for performance
- Handle concurrent writes safely

**Key Methods**:
```python
def log_changes(
    changes: List[ChangeEvent],
    ingestion_id: str,
    source_adapter: str
) -> Result[int]

def flush_pending_changes() -> Result[int]
```

#### 2.3 Smart Update Service

**File**: `src/adapters/storage/smart_update_service.py`

**Responsibilities**:
- Fetch existing records from database
- Compare with new records
- Generate UPDATE statements with only changed fields
- Execute updates with change logging

**Key Methods**:
```python
def smart_update_dataframe(
    df: pd.DataFrame,
    table_name: str,
    primary_key: str
) -> Result[UpdateResult]

def get_existing_records(
    record_ids: List[str],
    table_name: str,
    primary_key: str
) -> pd.DataFrame
```

### 3. Implementation Phases

#### Phase 1: Foundation (Week 1)

**Tasks**:
1. Create `change_audit_log` table schema
2. Implement `ChangeEvent` Pydantic model
3. Implement `ChangeDetector` service
4. Add unit tests for change detection

**Deliverables**:
- Database migration script
- `ChangeDetector` service with tests
- Documentation

#### Phase 2: Change Logging (Week 2)

**Tasks**:
1. Implement `ChangeAuditLogger`
2. Integrate with PostgreSQL adapter
3. Add batch logging for performance
4. Test with sample data

**Deliverables**:
- `ChangeAuditLogger` implementation
- Integration tests
- Performance benchmarks

#### Phase 3: Smart Updates (Week 3)

**Tasks**:
1. Implement `SmartUpdateService`
2. Modify `PostgreSQLAdapter.persist_dataframe()` to use smart updates
3. Add change detection before updates
4. Optimize for bulk operations

**Deliverables**:
- Smart update logic
- Updated persistence methods
- Performance comparison (before/after)

#### Phase 4: Query & Reporting (Week 4)

**Tasks**:
1. Add API endpoints for change history
2. Create dashboard views for change audit
3. Add filtering and search capabilities
4. Generate compliance reports

**Deliverables**:
- API endpoints (`/api/change-history`)
- Dashboard UI components
- Report generation utilities

### 4. Data Models

#### 4.1 ChangeEvent Model

```python
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

class ChangeEvent(BaseModel):
    """Represents a single field-level change."""
    table_name: str
    record_id: str
    field_name: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    change_type: str  # 'INSERT', 'UPDATE', 'DELETE'
    changed_at: datetime = Field(default_factory=datetime.now)
    ingestion_id: Optional[str] = None
    source_adapter: Optional[str] = None
    
    def to_audit_dict(self) -> dict:
        """Convert to dictionary for audit log insertion."""
        return {
            'change_id': str(uuid.uuid4()),
            'table_name': self.table_name,
            'record_id': self.record_id,
            'field_name': self.field_name,
            'old_value': self._serialize_value(self.old_value),
            'new_value': self._serialize_value(self.new_value),
            'change_type': self.change_type,
            'changed_at': self.changed_at,
            'ingestion_id': self.ingestion_id,
            'source_adapter': self.source_adapter
        }
    
    def _serialize_value(self, value: Any) -> Optional[str]:
        """Serialize complex types to JSON string."""
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)
```

#### 4.2 UpdateResult Model

```python
class UpdateResult(BaseModel):
    """Result of a smart update operation."""
    records_processed: int
    records_inserted: int
    records_updated: int
    records_unchanged: int
    fields_changed: int
    changes_logged: int
```

### 5. Integration Points

#### 5.1 PostgreSQL Adapter Modifications (Optimized for Batch Processing)

**Current**: `persist_dataframe()` uses `ON CONFLICT DO UPDATE SET` which updates all fields.

**New (Optimized)**: 
1. **Bulk fetch** existing records for entire chunk (single query)
2. **Vectorized comparison** using pandas merge
3. **Split** into inserts (no existing) and updates (existing)
4. **Bulk UPDATE** with only changed fields (using CASE statements or bulk operations)
5. **Async log** changes to `change_audit_log` (non-blocking)

**Optimized Pseudocode**:
```python
def persist_dataframe(
    self, 
    df: pd.DataFrame, 
    table_name: str,
    chunk_cache: Optional[ChunkCache] = None,
    change_logger: Optional[AsyncChangeLogger] = None
) -> Result[int]:
    """Persist DataFrame with CDC, optimized for chunk-based batch processing."""
    
    if not ENABLE_CDC:
        # Fallback to existing UPSERT logic
        return self._persist_dataframe_upsert(df, table_name)
    
    # Get primary key for table
    primary_key = self._get_primary_key(table_name)
    
    if primary_key not in df.columns:
        return Result.failure_result(ValueError(f"Primary key {primary_key} not in DataFrame"))
    
    # Bulk fetch existing records (using cache if available)
    record_ids = df[primary_key].unique().tolist()
    
    if chunk_cache:
        existing_df = chunk_cache.get_existing_records(
            record_ids, table_name, primary_key, self
        )
    else:
        existing_df = self._bulk_fetch_existing_records(
            record_ids, table_name, primary_key
        )
    
    # Split into inserts and updates using pandas operations
    if existing_df.empty:
        # All new records - fast path
        result = self._insert_records_bulk(df, table_name)
        if change_logger:
            # Log all fields as INSERTs
            insert_changes = self._generate_insert_changes(df, table_name, primary_key)
            change_logger.log_changes_async(insert_changes)
        return result
    
    # Merge to identify inserts vs updates
    merged = existing_df.merge(
        df,
        on=primary_key,
        suffixes=('_old', '_new'),
        how='outer',
        indicator=True
    )
    
    # Process inserts (no existing record)
    inserts_df = merged[merged['_merge'] == 'right_only']
    if not inserts_df.empty:
        # Extract new columns only
        insert_cols = [col for col in df.columns if not col.endswith('_old')]
        inserts_df = inserts_df[[col for col in insert_cols if col in inserts_df.columns]]
        self._insert_records_bulk(inserts_df, table_name)
        
        if change_logger:
            insert_changes = self._generate_insert_changes(inserts_df, table_name, primary_key)
            change_logger.log_changes_async(insert_changes)
    
    # Process updates (existing record exists)
    updates_df = merged[merged['_merge'] == 'both']
    if not updates_df.empty:
        # Vectorized change detection
        changes_df = self._detect_changes_vectorized(
            updates_df, table_name, primary_key
        )
        
        if not changes_df.empty:
            # Build selective UPDATE with only changed fields
            self._update_changed_fields_bulk(
                updates_df, changes_df, table_name, primary_key
            )
            
            if change_logger:
                change_events = self._changes_df_to_events(changes_df)
                change_logger.log_changes_async(change_events)
        else:
            # No changes - skip UPDATE (performance optimization)
            logger.debug(f"No changes detected for {len(updates_df)} records in {table_name}")
    
    return Result.success_result(len(df))
```

**Key Optimizations**:
1. **Single bulk fetch** per chunk (not per record)
2. **Pandas merge** for vectorized comparison
3. **Skip UPDATE** if no changes detected
4. **Async change logging** (non-blocking)
5. **Chunk cache** to avoid re-fetching
6. **Fast path** for all-new records

#### 5.2 Change Detection Logic (Vectorized for Performance)

```python
def detect_changes_vectorized(
    merged_df: pd.DataFrame,
    table_name: str,
    primary_key: str
) -> pd.DataFrame:
    """Detect field-level changes using vectorized pandas operations.
    
    This is MUCH faster than row-by-row comparison for large chunks.
    """
    changes_list = []
    
    # Get all data columns (exclude primary key and merge indicator)
    data_columns = [
        col.replace('_old', '').replace('_new', '') 
        for col in merged_df.columns 
        if col.endswith('_old') or (col.endswith('_new') and not col.startswith('_'))
    ]
    data_columns = list(set(data_columns))  # Remove duplicates
    
    # Vectorized comparison for each field
    for col in data_columns:
        if col == primary_key:
            continue
            
        old_col = f"{col}_old"
        new_col = f"{col}_new"
        
        if old_col not in merged_df.columns or new_col not in merged_df.columns:
            continue
        
        # Handle different data types
        old_series = merged_df[old_col]
        new_series = merged_df[new_col]
        
        # Normalize for comparison (handle NaN, None, arrays)
        old_normalized = old_series.fillna('__NULL__')
        new_normalized = new_series.fillna('__NULL__')
        
        # For arrays/lists, convert to string for comparison
        if old_series.dtype == 'object':
            old_normalized = old_normalized.apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
            )
        if new_series.dtype == 'object':
            new_normalized = new_normalized.apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
            )
        
        # Vectorized comparison
        changed_mask = old_normalized != new_normalized
        
        if changed_mask.any():
            # Extract changed rows
            changed_rows = merged_df[changed_mask]
            
            # Build change records
            for idx, row in changed_rows.iterrows():
                changes_list.append({
                    'table_name': table_name,
                    'record_id': str(row[primary_key]),
                    'field_name': col,
                    'old_value': row[old_col],
                    'new_value': row[new_col],
                    'change_type': 'UPDATE'
                })
    
    return pd.DataFrame(changes_list)

def _values_equal(old: Any, new: Any) -> bool:
    """Compare two values accounting for NaN, None, arrays, etc.
    
    Note: This is used for single-value comparison. For bulk operations,
    use detect_changes_vectorized() instead.
    """
    # Handle None/NaN
    if pd.isna(old) and pd.isna(new):
        return True
    if pd.isna(old) or pd.isna(new):
        return False
    
    # Handle arrays/lists
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        return old == new
    
    # Handle dicts
    if isinstance(old, dict) and isinstance(new, dict):
        return old == new
    
    # Handle other types
    return old == new
```

### 6. Performance Considerations

#### 6.1 Batch Operations Strategy

**Problem**: Fetching existing records one-by-one would kill performance for millions of records.

**Solution**: Use bulk operations at the chunk level:

1. **Bulk Fetch Existing Records** (per chunk):
   ```python
   # Fetch ALL existing records for the chunk in ONE query
   record_ids = chunk_df[primary_key].unique().tolist()
   existing_df = pd.read_sql(
       f"SELECT * FROM {table_name} WHERE {primary_key} IN ({placeholders})",
       conn,
       params=record_ids
   )
   ```

2. **Pandas Merge for Bulk Comparison**:
   ```python
   # Vectorized comparison using pandas merge
   merged = existing_df.merge(
       chunk_df,
       on=primary_key,
       suffixes=('_old', '_new'),
       how='outer',
       indicator=True
   )
   
   # Identify inserts (no existing record)
   inserts = merged[merged['_merge'] == 'right_only']
   
   # Identify updates (existing record exists)
   updates = merged[merged['_merge'] == 'both']
   
   # Vectorized field comparison using pandas operations
   changes = detect_changes_vectorized(updates, table_name, primary_key)
   ```

3. **Bulk UPDATE with Changed Fields Only**:
   ```python
   # Use PostgreSQL UPDATE with CASE statements for bulk updates
   # Only update fields that actually changed
   update_sql = build_selective_update_sql(changes, table_name, primary_key)
   ```

#### 6.2 Chunk-Level Caching

**Problem**: Re-fetching the same records across chunks wastes time.

**Solution**: Maintain a chunk-level cache (see detailed implementation in section 6.1 above).

#### 6.3 Async Change Log Writing

**Problem**: Writing change logs synchronously would slow down ingestion.

**Solution**: Use async background writer (similar to redaction logs) - see detailed implementation in section 6.1 above.

#### 6.4 Performance Targets

For a file with **1 million records** in chunks of **50k rows**:

- **Change Detection**: < 2 seconds per chunk (vectorized pandas)
- **Bulk Fetch**: < 1 second per chunk (using temp tables)
- **Selective UPDATE**: < 3 seconds per chunk (bulk operations)
- **Change Log Write**: Async, non-blocking (batched every 10k changes)
- **Overall Impact**: < 10% overhead on ingestion throughput

#### 6.5 Indexing Strategy

```sql
-- Critical indexes for performance
CREATE INDEX idx_change_audit_table_record ON change_audit_log(table_name, record_id);
CREATE INDEX idx_change_audit_timestamp ON change_audit_log(changed_at);
CREATE INDEX idx_change_audit_ingestion ON change_audit_log(ingestion_id);

-- Covering index for common queries
CREATE INDEX idx_change_audit_covering ON change_audit_log(table_name, record_id, changed_at, field_name);
```

#### 6.6 Configuration & Feature Flags

Make CDC optional and configurable:

```python
# In settings or environment variables
ENABLE_CDC = os.getenv("ENABLE_CDC", "false").lower() == "true"
CDC_BATCH_SIZE = int(os.getenv("CDC_BATCH_SIZE", "10000"))
CDC_CACHE_SIZE = int(os.getenv("CDC_CACHE_SIZE", "100000"))
CDC_ASYNC_LOGGING = os.getenv("CDC_ASYNC_LOGGING", "true").lower() == "true"
```

### 7. Security & Compliance

#### 7.1 Immutable Audit Trail

- `change_audit_log` is append-only (no UPDATE/DELETE)
- Use database triggers to prevent modifications
- Regular backups for compliance retention

#### 7.2 PII Handling

- Change audit log may contain PII (old/new values)
- Ensure redaction is applied before logging
- Consider separate PII audit log with stricter access controls

#### 7.3 Access Control

- Read-only access to `change_audit_log` for most users
- Write access only to system processes
- Audit who queries change history

### 8. API Endpoints

#### 8.1 Change History Endpoint

```
GET /api/change-history
Query Parameters:
  - table_name: Filter by table (patients, encounters, observations)
  - record_id: Filter by specific record
  - field_name: Filter by field
  - start_date: Start timestamp
  - end_date: End timestamp
  - ingestion_id: Filter by ingestion run
  - limit: Max results (default: 1000)
  - offset: Pagination offset
```

#### 8.2 Record Change Summary

```
GET /api/change-history/{table_name}/{record_id}/summary
Returns: Summary of all changes for a specific record
```

### 9. Testing Strategy

#### 9.1 Unit Tests

- Change detection logic
- Value comparison (NaN, None, arrays)
- Change event serialization

#### 9.2 Integration Tests

- End-to-end update with change logging
- Batch processing with change detection
- Performance under load

#### 9.3 Compliance Tests

- Audit trail completeness
- Immutability verification
- Query performance benchmarks

### 10. Migration Plan

#### 10.1 Database Migration

1. Create `change_audit_log` table
2. Add indexes
3. Create database functions/triggers if needed
4. Test with sample data

#### 10.2 Code Migration

1. Implement change detection service
2. Update PostgreSQL adapter incrementally
3. Add feature flag to enable/disable CDC
4. Monitor performance impact
5. Roll out gradually

#### 10.3 Rollback Plan

- Feature flag to disable CDC
- Keep old UPSERT logic as fallback
- Ability to disable change logging if performance issues

### 11. Monitoring & Observability

#### 11.1 Metrics

- Number of changes detected per ingestion
- Average fields changed per record
- Change logging latency
- Database query performance

#### 11.2 Alerts

- High change rate (potential data quality issue)
- Change logging failures
- Performance degradation

### 12. Future Enhancements

1. **Point-in-Time Queries**: Use `record_versions` table to query historical state
2. **Change Replication**: Stream changes to external systems (Kafka, etc.)
3. **Change Analytics**: Dashboard for change patterns and trends
4. **Automated Alerts**: Notify on significant changes (e.g., patient address change)
5. **Change Approval Workflow**: Require approval for certain field changes

## Implementation Checklist

- [ ] Phase 1: Foundation
  - [ ] Create `change_audit_log` table schema
  - [ ] Implement `ChangeEvent` model
  - [ ] Implement `ChangeDetector` service
  - [ ] Add unit tests
- [ ] Phase 2: Change Logging
  - [ ] Implement `ChangeAuditLogger`
  - [ ] Integrate with PostgreSQL adapter
  - [ ] Add batch logging
  - [ ] Integration tests
- [ ] Phase 3: Smart Updates
  - [ ] Implement `SmartUpdateService`
  - [ ] Modify `persist_dataframe()` method
  - [ ] Add change detection
  - [ ] Performance optimization
- [ ] Phase 4: Query & Reporting
  - [ ] Add API endpoints
  - [ ] Create dashboard views
  - [ ] Add filtering/search
  - [ ] Generate reports

## Success Criteria

1. ✅ All field-level changes are logged to `change_audit_log`
2. ✅ Updates only modify changed fields (verified via SQL logs)
3. ✅ Change history queries return results in < 100ms for single record
4. ✅ Ingestion throughput remains > 80% of baseline
5. ✅ Audit trail is immutable and tamper-proof
6. ✅ Compliance reports can be generated from change audit log
