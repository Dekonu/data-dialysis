# Ingestion Performance Optimization Proposal

## Current Bottleneck Analysis

From the logs, we can see:
- **4-5 minute gap** between batches (NOT database persistence!)
- **Batch 1**: 10k patients (~2s), 10,641 encounters (~3s), 30,061 observations (~8-9s)
- **Total per batch**: ~13-14 seconds (DB persistence is fast!)
- **The real bottleneck**: 4-5 minutes of processing BEFORE each batch

### Root Cause: Sequential Pydantic Validation with SpaCy NER

**The Problem**:
1. **Line 176** (`json_ingester.py`): Loads entire JSON file into memory (`json.load(f)`)
2. **Lines 204-228**: Normalizes all records sequentially (500k records)
3. **Line 414**: `for idx, row in df.iterrows()` - **SEQUENTIAL ROW-BY-ROW PROCESSING**
4. **Line 536**: For each observation, creates `ClinicalObservation(**filtered_dict)`
5. **Line 625-636** (`golden_record.py`): `redact_notes` validator calls `RedactorService.redact_observation_notes()`
6. **Line 467-469** (`services.py`): Calls `_redact_names_with_ner()` which uses **SpaCy NER**
7. **Line 490**: `ner_adapter.extract_person_names(text)` - **EXPENSIVE!** Runs for EVERY observation's notes

**The Math**:
- 10k patients × ~3 observations each = 30k observations
- Each observation with notes triggers SpaCy NER (sequential, per-row)
- SpaCy NER: ~8-10ms per text → 30k × 8ms = **240 seconds = 4 minutes** ✅

### Current Flow (Sequential)
```
Chunk 1 Processing (4-5 minutes):
  ├─ Load entire JSON file into memory
  ├─ Normalize 500k records sequentially
  ├─ For each of 10k patients (row-by-row):
  │   ├─ Create PatientRecord (Pydantic validation)
  │   ├─ For each encounter: Create EncounterRecord
  │   └─ For each observation: Create ClinicalObservation
  │       └─ redact_notes() → SpaCy NER (8-10ms each!) ⚠️
  └─ Create DataFrames from validated records

Chunk 1 Persistence (13-14s):
  ├─ Persist 10k patients (2s)
  ├─ Persist 10,641 encounters (3s)
  └─ Persist 30,061 observations (8-9s)

Chunk 2: (same 4-5 minute processing delay)
  ...
```

## Optimization Strategies

### Strategy 0: Batch NER Processing (CRITICAL - Highest Priority)

**Concept**: Instead of calling SpaCy NER for each observation's notes individually during Pydantic validation, batch all notes and process them together.

**Current**: 
```python
# In _validate_dataframe_chunk, for each row:
observation = ClinicalObservation(**filtered_dict)  # Triggers redact_notes validator
# redact_notes() calls SpaCy NER for THIS ONE text (8-10ms)
```

**Proposed**:
```python
# Collect all notes first
all_notes = []
note_indices = []  # Track which observation each note belongs to

# Then batch process with NER
if ner_adapter:
    # Process all notes at once (SpaCy can batch process)
    redacted_notes = ner_adapter.extract_person_names_batch(all_notes)
    # Apply redactions
```

**Alternative (Simpler)**: Defer NER to after DataFrame creation, then apply vectorized redaction.

**Expected Improvement**: 
- Current: 30k observations × 8ms = 240s (4 minutes)
- Optimized: Batch NER ~10-20s total
- **~90% faster** (saves 3.5-4 minutes per chunk!)

**Implementation**:
1. Skip NER during Pydantic validation (use regex-only for notes)
2. After creating observations DataFrame, batch process all notes with NER
3. Update DataFrame with redacted notes

**Pros**:
- Massive performance gain (4 minutes → 20 seconds)
- SpaCy supports batch processing (more efficient)
- Minimal code changes

**Cons**:
- Notes redaction happens after validation (acceptable trade-off)
- Need to ensure notes are still redacted before persistence

---

### Strategy 1: Parallel Table Persistence (Recommended - Quick Win)

**Concept**: When processing a chunk, persist patients, encounters, and observations in parallel since they're independent operations.

**Implementation**:
- Collect all 3 DataFrames for a chunk before persisting
- Use `concurrent.futures.ThreadPoolExecutor` to persist all 3 tables simultaneously
- PostgreSQL connection pool handles concurrent connections

**Expected Improvement**: 
- Current: 2s + 3s + 9s = 14s per chunk
- Optimized: max(2s, 3s, 9s) = 9s per chunk
- **~35% faster** (5s saved per chunk)

**Pros**:
- Simple to implement
- Low risk (each table is independent)
- Works with existing connection pool
- No schema changes needed

**Cons**:
- Still sequential chunk processing
- Limited by slowest table (observations)

---

### Strategy 2: Adaptive Chunk Sizing (Recommended - Medium Effort)

**Concept**: Calculate optimal chunk size based on total row count (patients + encounters + observations) to keep processing time consistent.

**Current**: Fixed 10k patients per chunk → variable total rows (50k-60k)
**Proposed**: Target total rows (e.g., 50k) → variable patient chunk size

**Implementation**:
```python
# Calculate average ratios from first chunk
patients_per_chunk = 10000
encounters_per_patient = 1.06  # 10,641 / 10,000
observations_per_patient = 3.0  # 30,061 / 10,000

# Target total rows per chunk
target_total_rows = 50000

# Calculate optimal patient chunk size
optimal_patient_chunk = target_total_rows / (1 + encounters_per_patient + observations_per_patient)
# = 50000 / (1 + 1.06 + 3.0) = ~9,900 patients
```

**Expected Improvement**:
- More consistent processing times
- Better resource utilization
- Can be combined with Strategy 1

**Pros**:
- Better resource utilization
- More predictable performance
- Works with any data distribution

**Cons**:
- Requires initial analysis pass or adaptive learning
- More complex logic

---

### Strategy 3: Parallel Chunk Processing (Advanced - High Effort)

**Concept**: Process multiple chunks in parallel using worker threads/processes.

**Implementation**:
- Use `concurrent.futures.ThreadPoolExecutor` with 2-4 workers
- Each worker processes a full chunk (patients + encounters + observations)
- Coordinate with connection pool to avoid exhaustion

**Expected Improvement**:
- With 2 workers: ~2x faster
- With 4 workers: ~3-4x faster (diminishing returns due to I/O)

**Pros**:
- Maximum throughput
- Best for large files

**Cons**:
- Complex coordination
- Risk of connection pool exhaustion
- Harder to debug
- Circuit breaker coordination needed

---

### Strategy 4: Batch Size Optimization for Observations (Quick Win)

**Concept**: Use larger batch sizes for observations since they're the bottleneck.

**Current**: All tables use same batch size (10k patients → ~30k observations)
**Proposed**: Use smaller patient chunks but keep observation batches larger, or use database-specific bulk insert optimizations

**Implementation**:
- Use PostgreSQL `COPY` command for observations (faster than INSERT)
- Or increase observation batch size to 50k-100k rows

**Expected Improvement**:
- COPY is 2-3x faster than INSERT for bulk data
- **~20-30% faster** for observations

**Pros**:
- Database-level optimization
- No code architecture changes
- Works with existing flow

**Cons**:
- PostgreSQL-specific (need DuckDB equivalent)
- Requires testing

---

## Recommended Implementation Plan

### Phase 0: Critical Fix (1 day) - DO THIS FIRST!
1. **Strategy 0**: Batch NER processing for observation notes

**Expected Result**: **4-5 minutes → 20 seconds** per chunk (90% improvement!)

### Phase 1: Quick Wins (1-2 days)
1. **Strategy 4**: Implement PostgreSQL `COPY` for bulk inserts
2. **Strategy 1**: Parallel table persistence within chunks

**Expected Result**: Additional 40-50% improvement on persistence (but persistence is already fast)

### Phase 2: Adaptive Optimization (2-3 days)
3. **Strategy 2**: Adaptive chunk sizing based on total row count

**Expected Result**: More consistent performance, additional 10-15% improvement

### Phase 3: Advanced (Optional, 1 week)
4. **Strategy 3**: Parallel chunk processing (if needed for very large files)

---

## Implementation Details

### Strategy 1: Parallel Table Persistence

```python
# In main.py, modify process_ingestion():
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Collect all DataFrames for a chunk
chunk_dataframes = {
    'patients': None,
    'encounters': None,
    'observations': None
}

for result in adapter.ingest(source):
    if result.is_success() and hasattr(result.value, 'shape'):
        df = result.value
        # Determine table name
        if 'observation_id' in df.columns:
            table_name = 'observations'
        elif 'encounter_id' in df.columns:
            table_name = 'encounters'
        elif 'patient_id' in df.columns:
            table_name = 'patients'
        
        chunk_dataframes[table_name] = df
        
        # When we have all 3, persist in parallel
        if all(chunk_dataframes.values()):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(storage.persist_dataframe, df, table): (df, table)
                    for table, df in chunk_dataframes.items()
                    if df is not None
                }
                
                for future in as_completed(futures):
                    df, table = futures[future]
                    try:
                        persist_result = future.result()
                        if persist_result.is_success():
                            success_count += persist_result.value
                        else:
                            failure_count += len(df)
                    except Exception as e:
                        logger.error(f"Failed to persist {table}: {e}")
                        failure_count += len(df)
            
            # Reset for next chunk
            chunk_dataframes = {k: None for k in chunk_dataframes}
```

### Strategy 2: Adaptive Chunk Sizing

```python
# In JSONIngester.__init__():
def __init__(self, max_record_size: int = 10 * 1024 * 1024, 
             chunk_size: int = 10000,
             target_total_rows: int = 50000):
    self.chunk_size = chunk_size
    self.target_total_rows = target_total_rows
    self.ratios = None  # Will be calculated after first chunk

# After first chunk, calculate ratios:
if self.ratios is None:
    total_patients = len(validated_df)
    total_encounters = len(encounters_df)
    total_observations = len(observations_df)
    
    self.ratios = {
        'encounters_per_patient': total_encounters / total_patients if total_patients > 0 else 0,
        'observations_per_patient': total_observations / total_patients if total_patients > 0 else 0
    }
    
    # Calculate optimal chunk size
    total_per_patient = 1 + self.ratios['encounters_per_patient'] + self.ratios['observations_per_patient']
    self.chunk_size = int(self.target_total_rows / total_per_patient)
    logger.info(f"Adaptive chunk sizing: {self.chunk_size} patients per chunk (target: {self.target_total_rows} total rows)")
```

### Strategy 4: PostgreSQL COPY Optimization

```python
# In postgresql_adapter.py, modify persist_dataframe():
def persist_dataframe(self, df: pd.DataFrame, table_name: str) -> Result[int]:
    # For large DataFrames, use COPY instead of INSERT
    if len(df) > 10000:
        return self._persist_dataframe_copy(df, table_name)
    else:
        return self._persist_dataframe_insert(df, table_name)

def _persist_dataframe_copy(self, df: pd.DataFrame, table_name: str) -> Result[int]:
    """Use PostgreSQL COPY for bulk inserts (2-3x faster)."""
    from io import StringIO
    import csv
    
    conn = self._get_connection()
    try:
        cursor = conn.cursor()
        
        # Convert DataFrame to CSV in memory
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
        buffer.seek(0)
        
        # Use COPY FROM
        columns = ', '.join(df.columns)
        cursor.copy_expert(
            f"COPY {table_name} ({columns}) FROM STDIN WITH (FORMAT csv)",
            buffer
        )
        
        conn.commit()
        row_count = len(df)
        return Result.success_result(row_count)
    except Exception as e:
        conn.rollback()
        return Result.failure_result(...)
```

---

## Testing Plan

1. **Baseline**: Measure current performance (500k records)
2. **Strategy 1**: Test parallel persistence (should see ~35% improvement)
3. **Strategy 4**: Test COPY optimization (should see additional 20-30% for observations)
4. **Strategy 2**: Test adaptive chunk sizing (should see more consistent times)
5. **Combined**: Test all strategies together (target: 50-60% total improvement)

---

## Risk Assessment

| Strategy | Risk Level | Mitigation |
|----------|-----------|------------|
| Parallel Persistence | Low | Connection pool handles concurrency, each table is independent |
| Adaptive Chunking | Low | Fallback to fixed size if ratios can't be calculated |
| Parallel Chunks | Medium | Monitor connection pool, limit workers to pool size |
| COPY Optimization | Low | Fallback to INSERT if COPY fails, test thoroughly |

---

## Next Steps

1. Review and approve this proposal
2. Implement Strategy 1 (parallel persistence) - quick win
3. Implement Strategy 4 (COPY) for PostgreSQL
4. Measure improvements
5. Decide on Strategy 2 and 3 based on results

