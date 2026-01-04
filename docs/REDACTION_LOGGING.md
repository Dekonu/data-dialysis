# Redaction Logging and Security Reporting

## Overview

The Data-Dialysis pipeline includes comprehensive redaction logging and security reporting capabilities. Every PII redaction event is logged to a `logs` table in DuckDB with detailed information for compliance and auditing.

## Architecture

### Components

1. **`logs` Table** (DuckDB)
   - Stores all redaction events
   - Fields: `log_id`, `field_name`, `original_hash`, `timestamp`, `rule_triggered`, `record_id`, `source_adapter`, `ingestion_id`, `redacted_value`, `original_value_length`

2. **RedactionLogger** (`src/infrastructure/redaction_logger.py`)
   - In-memory buffer for redaction events
   - Thread-safe logging mechanism
   - Flushes to storage after ingestion

3. **LoggingRedactorService** (`src/infrastructure/redaction_service_wrapper.py`)
   - Wrapper around `RedactorService` that logs all redactions
   - Maintains same interface as `RedactorService`
   - Can be used in place of `RedactorService` for logging

4. **Security Report Generator** (`src/infrastructure/security_report.py`)
   - Generates comprehensive security reports
   - Summarizes redactions by field, rule, and adapter
   - Saves reports as JSON files

## Database Schema

### `logs` Table

```sql
CREATE TABLE IF NOT EXISTS logs (
    log_id VARCHAR PRIMARY KEY,
    field_name VARCHAR NOT NULL,
    original_hash VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    rule_triggered VARCHAR NOT NULL,
    record_id VARCHAR,
    source_adapter VARCHAR,
    ingestion_id VARCHAR,
    redacted_value VARCHAR,
    original_value_length INTEGER
);
```

**Indexes:**
- `idx_logs_field_name` on `field_name`
- `idx_logs_timestamp` on `timestamp`
- `idx_logs_rule_triggered` on `rule_triggered`
- `idx_logs_record_id` on `record_id`
- `idx_logs_ingestion_id` on `ingestion_id`

## Usage

### Automatic Logging (via main.py)

When using `main.py` for ingestion, redaction logging and security reports are generated automatically:

```bash
python -m src.main --input data/patients.csv
```

This will:
1. Initialize redaction logger with unique ingestion ID
2. Process all records (redactions are logged automatically if using `LoggingRedactorService`)
3. Flush redaction logs to database
4. Generate and print security report
5. Save report to `reports/security_report_{ingestion_id}.json` (if enabled)

**Configuration:**
- `DD_SAVE_SECURITY_REPORT=true|false` - Enable/disable saving report to file (default: `true`)
- `DD_SECURITY_REPORT_DIR=reports` - Directory to save reports (default: `reports`)

**Example:**
```bash
# Disable saving report to file (report still generated and printed)
export DD_SAVE_SECURITY_REPORT=false
python -m src.main --input data/patients.csv

# Use custom report directory
export DD_SECURITY_REPORT_DIR=/var/log/security_reports
python -m src.main --input data/patients.csv
```

### Manual Logging

To use redaction logging in custom code:

```python
from src.infrastructure.redaction_logger import get_redaction_logger
from src.infrastructure.redaction_service_wrapper import LoggingRedactorService
from src.adapters.storage import DuckDBAdapter

# Initialize logger
redaction_logger = get_redaction_logger()
redaction_logger.set_ingestion_id("my-ingestion-001")

# Use logging redactor service
redactor = LoggingRedactorService(
    record_id="MRN001",
    source_adapter="csv_ingester"
)

# Redact values (automatically logged)
phone = redactor.redact_phone("555-123-4567")
email = redactor.redact_email("patient@example.com")

# Flush logs to storage
storage = DuckDBAdapter(db_path="data.duckdb")
storage.initialize_schema()
storage.flush_redaction_logs(redaction_logger.get_logs())
```

### Generating Security Reports

```python
from src.infrastructure.security_report import generate_security_report, print_security_report_summary
from src.adapters.storage import DuckDBAdapter

storage = DuckDBAdapter(db_path="data.duckdb")

# Generate report for specific ingestion
report_result = generate_security_report(
    storage=storage,
    ingestion_id="my-ingestion-001"
)

if report_result.is_success():
    report = report_result.value
    print_security_report_summary(report)
    
    # Save to file
    generate_security_report(
        storage=storage,
        output_path="reports/my_report.json",
        ingestion_id="my-ingestion-001"
    )
```

### Querying Redaction Logs Directly

```python
import duckdb

conn = duckdb.connect("data.duckdb")

# Get all redactions for a specific field
redactions = conn.execute("""
    SELECT * FROM logs
    WHERE field_name = 'phone'
    ORDER BY timestamp DESC
""").fetchall()

# Get redactions by rule
by_rule = conn.execute("""
    SELECT rule_triggered, COUNT(*) as count
    FROM logs
    GROUP BY rule_triggered
    ORDER BY count DESC
""").fetchall()

# Get redactions for a specific ingestion
ingestion_logs = conn.execute("""
    SELECT * FROM logs
    WHERE ingestion_id = ?
    ORDER BY timestamp
""", ["my-ingestion-001"]).fetchall()
```

## Security Report Format

Security reports are JSON files with the following structure:

```json
{
  "report_timestamp": "2025-01-15T10:30:00",
  "ingestion_id": "abc-123-def",
  "start_timestamp": null,
  "end_timestamp": null,
  "summary": {
    "total_redactions": 150,
    "redactions_by_field": {
      "phone": 45,
      "email": 38,
      "ssn": 12,
      "name": 55
    },
    "redactions_by_rule": {
      "PHONE_PATTERN": 45,
      "EMAIL_PATTERN": 38,
      "SSN_PATTERN": 12,
      "NAME_PATTERN": 55
    },
    "redactions_by_adapter": {
      "csv_ingester": 100,
      "json_ingester": 50
    }
  },
  "events": [
    {
      "log_id": "uuid-here",
      "field_name": "phone",
      "original_hash": "sha256-hash",
      "timestamp": "2025-01-15T10:25:00",
      "rule_triggered": "PHONE_PATTERN",
      "record_id": "MRN001",
      "source_adapter": "csv_ingester",
      "ingestion_id": "abc-123-def",
      "redacted_value": "***-***-****",
      "original_value_length": 12
    }
  ]
}
```

## Redaction Rules

The following redaction rules are tracked:

- `SSN_PATTERN` - Social Security Number detection
- `PHONE_PATTERN` - Phone number detection
- `EMAIL_PATTERN` - Email address detection
- `NAME_PATTERN` - Name detection
- `ADDRESS_PATTERN` - Address detection
- `DATE_REDACTION` - Date of birth redaction
- `ZIP_CODE_PARTIAL_REDACTION` - ZIP code partial redaction
- `UNSTRUCTURED_TEXT_PII_DETECTION` - PII found in unstructured text
- `OBSERVATION_NOTES_PII_DETECTION` - PII found in observation notes

## Security Considerations

1. **Original Value Hashing**: Original PII values are never stored in plaintext. Only SHA256 hashes are stored, allowing verification without exposing sensitive data.

2. **Immutable Logs**: The `logs` table is append-only. Once a redaction is logged, it cannot be modified or deleted.

3. **Audit Trail**: Every redaction includes:
   - Timestamp
   - Field name
   - Rule that triggered the redaction
   - Record identifier
   - Source adapter
   - Ingestion ID for grouping

4. **Compliance**: Reports can be generated for specific time periods or ingestion runs, supporting HIPAA/GDPR compliance requirements.

## Integration with Ingesters

To enable redaction logging in custom ingesters, replace `RedactorService` with `LoggingRedactorService`:

```python
# Before
from src.domain.services import RedactorService
df['phone'] = RedactorService.redact_phone(df['phone'])

# After
from src.infrastructure.redaction_service_wrapper import LoggingRedactorService
redactor = LoggingRedactorService(record_id=row['patient_id'], source_adapter="my_ingester")
df['phone'] = redactor.redact_phone(df['phone'])
```

## Future Enhancements

- Real-time redaction logging (currently batched)
- Integration with external SIEM systems
- Automated anomaly detection (unusual redaction patterns)
- Compliance report templates (HIPAA, GDPR)
- Redaction verification workflows

