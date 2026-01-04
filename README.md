# 🛡️ Data-Dialysis: Self-Securing Data Ingestion Engine

<div align="center">

**A production-ready, security-first data pipeline for ingesting clinical and sensitive data with automatic PII redaction, schema validation, and threat protection.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

A **"Safety-First"** data pipeline designed to ingest disparate clinical-style datasets while automatically redacting PII (Personally Identifiable Information) and enforcing schema strictness before data ever touches a persistent database.

**🎯 Showcase Feature:** Processes **100MB+ XML files** with constant memory usage (~50-100MB peak) using streaming architecture - demonstrating production-ready scalability.

**Key Features:**
- 🔒 **Automatic PII Redaction** - HIPAA/GDPR compliant with audit trails
- 🛡️ **Security-First Architecture** - Protection against XML attacks, injection, and resource exhaustion
- 🏗️ **Hexagonal Architecture** - Clean separation of concerns, highly testable
- ⚡ **High Performance** - Streaming processing for large files (handles 100MB+ files efficiently)
- 📊 **Scalable Processing** - O(record_size) memory usage with streaming, not O(file_size)
- ✅ **Strict Validation** - Pydantic V2 schemas with fail-fast error handling
- 🔄 **Circuit Breaker** - Automatic quality gates to prevent bad data ingestion

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install CLI (optional, for development)
pip install -e .
```

### Basic Usage

```bash
# Ingest a CSV file
datadialysis ingest data/patients.csv

# Ingest an XML file (requires config)
datadialysis ingest data/encounters.xml --xml-config xml_config.json

# Ingest with custom batch size
datadialysis ingest data/observations.json --batch-size 5000

# View system information
datadialysis info

# Run performance benchmarks
datadialysis benchmark test_data/ xml_config.json
```

---

## 🛡️ Threat Model & Security

**This system is designed to process data from untrusted sources while maintaining HIPAA/GDPR compliance.**

See **[THREAT_MODEL.md](THREAT_MODEL.md)** for comprehensive documentation of:
- **Attack vectors** (XML attacks, PII leakage, injection, resource exhaustion)
- **Defense mechanisms** (defusedxml, streaming, validation, circuit breakers)
- **Security layers** (defense in depth architecture)
- **Compliance** (HIPAA, GDPR audit trails)

### Key Security Features

✅ **XML Attack Prevention**
- Billion Laughs attack protection via `defusedxml`
- Quadratic blowup prevention with streaming parser
- XXE (XML External Entity) attack blocking

✅ **PII Redaction**
- Automatic detection and redaction of SSNs, phone numbers, emails
- Name entity recognition in unstructured text
- Irreversible redaction with audit trail

✅ **Data Poisoning Protection**
- Strict Pydantic schema validation
- Circuit breaker halts ingestion if error rate >10%
- SQL/XSS injection prevention via parameterized queries

✅ **Resource Exhaustion Protection**
- Streaming processing for large files (O(record_size) memory)
- Record size limits (default: 10MB per record)
- Event/depth limits prevent CPU exhaustion

---

## 🏗️ Architecture

### Hexagonal (Ports & Adapters) Architecture

This engine uses **Hexagonal Architecture** to decouple business logic from infrastructure:

```
┌─────────────────────────────────────────┐
│         Domain Core (Pure Python)       │
│  - GoldenRecord (Pydantic schemas)     │
│  - RedactorService (PII redaction)      │
│  - CircuitBreaker (Quality gates)      │
└─────────────────────────────────────────┘
           ↕ Ports (Interfaces)
┌─────────────────────────────────────────┐
│      Adapters (Infrastructure)          │
│  - CSV/JSON/XML Ingestion Adapters     │
│  - DuckDB/PostgreSQL Storage Adapters  │
└─────────────────────────────────────────┘
```

**Benefits:**
- **Testability:** Core logic can be unit-tested without databases
- **Flexibility:** Swap adapters without changing business logic
- **Security:** Safety layer is isolated and cannot be bypassed

---

## 📋 Data Flow

### Verify-Then-Load Pattern

Unlike traditional ELT pipelines, Data-Dialysis follows a **Verify-Then-Load** pattern:

```
Input File (CSV/JSON/XML)
    ↓
[1] Secure Parsing
    - defusedxml for XML
    - Streaming for large files
    - Event/depth limits
    ↓
[2] PII Redaction
    - Regex-based detection
    - Field-level validation
    - Unstructured text scanning
    ↓
[3] Schema Validation
    - Pydantic strict validation
    - Type coercion
    - Pattern matching
    ↓
[4] Circuit Breaker Check
    - Failure rate monitoring
    - Auto-halt on threshold
    ↓
[5] Secure Persistence
    - Parameterized queries
    - Transaction safety
    - Audit logging
    ↓
Database (DuckDB/PostgreSQL)
```

**Security Boundary:** The Safety Layer (steps 2-3) is the **hard security boundary**. Data cannot reach the database without passing through validation and redaction.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Validation:** Pydantic V2 (strict schema enforcement)
- **Data Processing:** Pandas (vectorized operations)
- **XML Security:** defusedxml + lxml (streaming)
- **Database:** DuckDB (analytical) / PostgreSQL (production)
- **CLI:** Typer + Rich (modern, type-safe CLI)
- **Testing:** pytest + hypothesis (property-based testing)

---

## 📖 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive architecture overview and design decisions
- **[THREAT_MODEL.md](THREAT_MODEL.md)** - Detailed threat model and security architecture
- **[docs/XML_STREAMING_DESIGN.md](docs/XML_STREAMING_DESIGN.md)** - XML streaming implementation details
- **[docs/REDACTION_LOGGING.md](docs/REDACTION_LOGGING.md)** - PII redaction and audit trail design
- **[docs/CONFIG_MANAGER_DESIGN.md](docs/CONFIG_MANAGER_DESIGN.md)** - Configuration management
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guidelines for contributing to the project

---

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
export DD_DB_TYPE=duckdb                    # or postgresql
export DD_DB_PATH=./data/clinical.db        # DuckDB path
export DD_DB_HOST=localhost                  # PostgreSQL host
export DD_DB_NAME=clinical_db               # PostgreSQL database

# Processing Configuration
export DD_BATCH_SIZE=1000                   # Batch size for processing
export DD_CHUNK_SIZE=5000                   # Chunk size for CSV/JSON
export DD_MAX_RECORD_SIZE=10485760          # Max record size (10MB)

# Security Configuration
export DD_CIRCUIT_BREAKER_ENABLED=true      # Enable circuit breaker
export DD_CIRCUIT_BREAKER_THRESHOLD=0.1     # 10% failure threshold
export DD_XML_STREAMING_ENABLED=true        # Enable XML streaming
export DD_XML_STREAMING_THRESHOLD=104857600 # 100MB threshold

# Logging
export DD_LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
export DD_SAVE_SECURITY_REPORT=true         # Save security reports
```

### XML Configuration

For XML ingestion, create a JSON configuration file mapping XPath expressions to fields:

```json
{
  "root_element": "./PatientRecord",
  "fields": {
    "mrn": "./MRN",
    "patient_name": "./Demographics/FullName",
    "patient_dob": "./Demographics/BirthDate",
    "patient_gender": "./Demographics/Gender",
    "ssn": "./Demographics/SSN",
    "phone": "./Demographics/Phone",
    "email": "./Demographics/Email",
    "address_line1": "./Demographics/Address/Street",
    "city": "./Demographics/Address/City",
    "state": "./Demographics/Address/State",
    "postal_code": "./Demographics/Address/ZIP"
  }
}
```

---

## 📊 Performance & Scalability

### Large File Processing Capability

**Clinical-Sieve can efficiently process files up to 100MB+ using streaming architecture:**

- ✅ **100MB XML files** processed with constant memory usage
- ✅ **Streaming parser** prevents memory exhaustion on large datasets
- ✅ **Automatic mode selection** - uses streaming for files >100MB
- ✅ **O(record_size) memory** - not O(file_size) - scales to any file size

This capability demonstrates production-ready data pipeline engineering, handling real-world clinical data volumes without resource exhaustion.

### Benchmarking

Generate test files and benchmark performance:

```bash
# Generate test XML files (1MB, 5MB, 10MB, 25MB, 50MB, 75MB, 100MB)
python scripts/generate_xml_test_files.py

# Run benchmarks (includes 100MB file)
python scripts/benchmark_xml_ingestion.py test_data/ xml_config.json --iterations 3

# Or use CLI
datadialysis benchmark test_data/ xml_config.json --iterations 3
```

### Expected Performance

- **Small files (<10MB):** Traditional mode, ~2,000-5,000 records/sec
- **Large files (100MB+):** Streaming mode, ~1,400-1,800 records/sec
- **Memory usage:** O(record_size) with streaming, not O(file_size)
- **100MB file processing:** Constant memory (~50-100MB peak) regardless of file size

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_xml_ingestion.py -v
```

**Test Coverage:**
- ✅ 183 tests covering ingestion, validation, redaction
- ✅ Adversarial tests for security (XML attacks, injection attempts)
- ✅ Property-based tests with Hypothesis
- ✅ Integration tests with DuckDB

---

## 📝 Examples

### Example 1: Ingest CSV File

```bash
datadialysis ingest data/patients.csv
```

### Example 2: Ingest XML with Custom Config

```bash
datadialysis ingest data/encounters.xml \
    --xml-config custom_mappings.json \
    --batch-size 2000 \
    --verbose
```

### Example 3: Programmatic Usage

```python
from src.adapters.ingesters import get_adapter
from src.adapters.storage import DuckDBAdapter

# Get ingestion adapter
adapter = get_adapter("data/patients.csv")

# Process records
for result in adapter.ingest("data/patients.csv"):
    if result.is_success():
        print(f"Processed: {result.value.patient.patient_id}")
    else:
        print(f"Failed: {result.error}")
```

---

## 🔒 Security Best Practices

1. **Never disable security features** in production
2. **Monitor circuit breaker statistics** for data quality issues
3. **Review security reports** regularly for anomalies
4. **Use environment variables** for sensitive configuration
5. **Enable audit logging** for compliance requirements
6. **Keep dependencies updated** (especially defusedxml, lxml)

---

## 🤝 Contributing

We welcome contributions! This is a security-critical system, so please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

**Key Requirements:**
- Include tests (especially adversarial tests)
- Document security impact
- Follow Hexagonal Architecture principles
- Maintain backward compatibility
- Update documentation for new features

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **defusedxml** - XML attack prevention
- **Pydantic** - Schema validation
- **DuckDB** - High-performance analytical database
- **Typer** - Modern CLI framework

---

---

**Version:** 1.0.0 | **Last Updated:** January 2026

> **Note:** This project demonstrates production-ready patterns for secure data pipelines, including Hexagonal Architecture, PII redaction, threat protection, and comprehensive testing strategies. It serves as a reference implementation for building secure, maintainable data ingestion systems.
