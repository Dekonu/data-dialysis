# Examples

This directory contains example scripts demonstrating how to use Data-Dialysis.

## End-to-End Flow Example

`end_to_end_flow.py` demonstrates the complete data flow from raw files to DuckDB:

### What It Shows

1. **CSV Flow**: CSV → Pandas DataFrame → PII Redaction → DuckDB
2. **JSON Flow**: JSON → Pandas DataFrame → PII Redaction → DuckDB
3. **XML Flow**: XML → GoldenRecord → PII Redaction → DuckDB

### Running the Example

```bash
python examples/end_to_end_flow.py
```

### What You'll See

The example will:
1. Create sample data files (CSV, JSON, XML) with PII
2. Process each file through the ingestion pipeline
3. Show PII redaction at each step
4. Persist data to DuckDB
5. Verify data in the database
6. Display summary statistics

### Sample Output

```
======================================================================
Data-Dialysis: End-to-End Data Flow Demonstration
======================================================================

Creating Sample Data Files
----------------------------------------------------------------------
✓ Created CSV file: /tmp/patients.csv
  Records: 3 (excluding header)
  Contains PII: Names, SSNs, Phone, Email, Address, DOB

======================================================================
CSV Flow: CSV -> Pandas -> Redaction -> DuckDB
======================================================================

[Step 1] Initializing DuckDB adapter...
✓ Schema initialized

[Step 2] Getting CSV ingestion adapter...
✓ Adapter: CSVIngester

[Step 3] Ingesting CSV file...
  → CSV file is read in chunks
  → Each chunk is converted to pandas DataFrame
  → PII redaction is applied (vectorized)
  → DataFrames are validated

  Chunk 1: 3 rows
    Columns: ['patient_id', 'family_name', 'given_names', ...]
    Names redacted: 3/3
  → Persisting to DuckDB...
    ✓ Persisted 3 rows

[Step 4] Verifying data in DuckDB...
  Total patients in DB: 3

  Sample record (MRN001):
    Patient ID: MRN001 (preserved)
    Family Name: None (redacted)
    Phone: None (redacted)
    Email: None (redacted)
    DOB: None (redacted)

  Audit log entries: 1

✓ CSV flow completed successfully!
```

## Storage Adapter Usage Example

`storage_adapter_usage.py` (if created) shows how to use storage adapters with the configuration manager.

