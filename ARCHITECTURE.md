# Architecture Overview

This document provides a comprehensive overview of the Data-Dialysis architecture, design decisions, and patterns used throughout the codebase.

## 🏗️ High-Level Architecture

Data-Dialysis follows **Hexagonal Architecture** (also known as Ports and Adapters), which separates business logic from infrastructure concerns.

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Boundary                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Domain Core (Business Logic)                │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │  GoldenRecord (Pydantic Schemas)             │   │  │
│  │  │  - Patient, Encounter, Observation models   │   │  │
│  │  │  - Strict validation rules                   │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │  RedactorService (PII Redaction)              │   │  │
│  │  │  - Regex-based detection                      │   │  │
│  │  │  - Field-level redaction                     │   │  │
│  │  │  - Audit trail generation                    │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │  CircuitBreaker (Quality Gates)              │   │  │
│  │  │  - Failure rate monitoring                   │   │  │
│  │  │  - Automatic batch abortion                  │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │  Ports (Interfaces)                          │   │  │
│  │  │  - IngestionPort                             │   │  │
│  │  │  - StoragePort                               │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
│                        ↕ Ports                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Adapters (Infrastructure)                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │  │
│  │  │ CSV Ingester │  │ JSON Ingester│  │XML Ingester│ │  │
│  │  └──────────────┘  └──────────────┘  └───────────┘ │  │
│  │  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │DuckDB Adapter│  │PostgreSQL Adp│                │  │
│  │  └──────────────┘  └──────────────┘                │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Infrastructure (Cross-Cutting Concerns)          │  │
│  │  - Configuration Manager                             │  │
│  │  - Redaction Logger                                  │  │
│  │  - Security Report Generator                        │  │
│  │  - XML Streaming Parser                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Layer Breakdown

### Domain Core (`src/domain/`)

The **Domain Core** contains pure business logic with no external dependencies. This is the "inside" of the hexagon.

#### Components

**`golden_record.py`** - Pydantic V2 schemas defining the canonical data model:
- `Patient` - Patient demographics (with PII fields)
- `Encounter` - Clinical encounters
- `Observation` - Clinical observations
- `GoldenRecord` - Complete record combining all entities

**`services.py`** - Business logic services:
- `RedactorService` - PII redaction logic
- Field-level redaction rules
- Audit trail generation

**`guardrails.py`** - Quality control:
- `CircuitBreaker` - Monitors failure rates
- `CircuitBreakerConfig` - Configuration
- Automatic batch abortion on threshold breach

**`ports.py`** - Protocol definitions (interfaces):
- `IngestionPort` - Interface for data ingestion
- `StoragePort` - Interface for data persistence
- `Result[T]` - Functional error handling type

**`enums.py`** - Domain enumerations:
- Data source types
- Field types
- Redaction types
- Status codes

**`utils.py`** - Pure utility functions:
- Date parsing
- String normalization
- Type conversions

### Adapters (`src/adapters/`)

**Adapters** implement the Port interfaces and handle I/O operations. They can be swapped without changing domain logic.

#### Ingestion Adapters

**`csv_ingester.py`** - CSV file ingestion:
- Chunked reading for large files
- Pandas DataFrame processing
- Vectorized PII redaction

**`json_ingester.py`** - JSON file ingestion:
- Streaming JSON parsing
- Chunked processing
- Error handling

**`xml_ingester.py`** - XML file ingestion:
- Streaming parser for large files
- XPath-based field extraction
- XML attack prevention (defusedxml)
- Configurable field mapping

#### Storage Adapters

**`duckdb_adapter.py`** - DuckDB storage:
- High-performance analytical database
- Schema initialization
- Batch inserts
- Query interface

**`postgresql_adapter.py`** - PostgreSQL storage:
- Production-ready relational database
- Connection pooling
- Transaction management
- Parameterized queries (SQL injection prevention)

### Infrastructure (`src/infrastructure/`)

**Infrastructure** handles cross-cutting concerns and configuration.

**`settings.py`** - Application settings:
- Environment variable loading
- Default values
- Type-safe configuration

**`config_manager.py`** - Configuration management:
- Database configuration
- Adapter selection
- Environment-based overrides

**`redaction_logger.py`** - Audit trail:
- PII redaction logging
- Immutable audit records
- Compliance reporting

**`redaction_context.py`** - Context management:
- Thread-local redaction state
- Request-scoped logging

**`security_report.py`** - Security reporting:
- Ingestion statistics
- Security event aggregation
- Report generation

**`xml_streaming_parser.py`** - XML streaming:
- Memory-efficient parsing
- Event-based processing
- Resource limits

## 🔄 Data Flow

### Verify-Then-Load Pattern

Data-Dialysis uses a **Verify-Then-Load** pattern, ensuring data is validated and redacted before persistence:

```
1. Input File (CSV/JSON/XML)
   ↓
2. Adapter Selection (based on file extension)
   ↓
3. Secure Parsing
   - defusedxml for XML (prevents attacks)
   - Streaming for large files (prevents exhaustion)
   - Event/depth limits (prevents CPU exhaustion)
   ↓
4. PII Redaction (RedactorService)
   - Regex-based detection (SSN, phone, email)
   - Field-level validation
   - Unstructured text scanning
   - Audit trail generation
   ↓
5. Schema Validation (Pydantic)
   - Strict type checking
   - Pattern matching
   - Custom validators
   - Fail-fast on errors
   ↓
6. Circuit Breaker Check
   - Failure rate calculation
   - Threshold comparison
   - Batch abortion if exceeded
   ↓
7. Secure Persistence (Storage Adapter)
   - Parameterized queries (SQL injection prevention)
   - Transaction safety
   - Audit logging
   ↓
8. Database (DuckDB/PostgreSQL)
```

### Security Boundary

The **Security Boundary** is between steps 3-5. Data cannot reach the database without:
1. Passing through secure parsing
2. Having PII redacted
3. Passing schema validation
4. Passing circuit breaker checks

## 🔌 Ports and Adapters Pattern

### Ports (Interfaces)

Ports define **what** the system needs, not **how** it's implemented:

```python
# src/domain/ports.py
class IngestionPort(Protocol):
    def ingest(self, source: str) -> Iterator[Result[GoldenRecord]]: ...

class StoragePort(Protocol):
    def initialize_schema(self) -> Result[None]: ...
    def persist(self, records: list[GoldenRecord]) -> Result[int]: ...
    def query(self, sql: str) -> Result[list[dict]]: ...
```

### Adapters (Implementations)

Adapters implement ports and can be swapped:

```python
# src/adapters/ingesters/csv_ingester.py
class CSVIngester:
    def ingest(self, source: str) -> Iterator[Result[GoldenRecord]]:
        # CSV-specific implementation
        pass

# src/adapters/storage/duckdb_adapter.py
class DuckDBAdapter:
    def persist(self, records: list[GoldenRecord]) -> Result[int]:
        # DuckDB-specific implementation
        pass
```

### Benefits

1. **Testability**: Domain logic can be tested without databases/files
2. **Flexibility**: Swap DuckDB for PostgreSQL without changing domain code
3. **Security**: Safety layer is isolated and cannot be bypassed
4. **Maintainability**: Clear separation of concerns

## 🛡️ Security Architecture

### Defense in Depth

Multiple security layers protect the system:

1. **Input Validation** - Secure parsing prevents malformed input
2. **PII Redaction** - Automatic detection and redaction
3. **Schema Validation** - Strict Pydantic validation
4. **Circuit Breaker** - Quality gates prevent bad data
5. **Parameterized Queries** - SQL injection prevention
6. **Audit Logging** - Compliance and forensics

### Threat Protection

See [THREAT_MODEL.md](THREAT_MODEL.md) for detailed threat analysis.

**XML Attacks:**
- Billion Laughs → defusedxml
- XXE → defusedxml
- Quadratic Blowup → Streaming parser

**Injection:**
- SQL Injection → Parameterized queries
- XSS → Output encoding (if web interface added)

**Resource Exhaustion:**
- Memory → Streaming processing
- CPU → Event/depth limits
- Disk → Record size limits

## 📊 Performance Considerations

### Large File Processing (100MB+)

Clinical-Sieve can efficiently process files up to **100MB+** using streaming architecture, demonstrating production-ready scalability:

- **Memory Usage**: O(record_size) - constant ~50-100MB peak memory regardless of file size
- **Scalability**: Can handle files of any size without memory exhaustion
- **Automatic Mode Selection**: Uses streaming mode for files >100MB
- **Benchmarking**: Test files up to 100MB available in `test_data/` directory

### Streaming Processing

Large files are processed in streams to maintain O(record_size) memory:

```python
# XML Streaming
for event, element in xml_streaming_parser.parse(file_path):
    record = extract_record(element)
    yield process_record(record)
```

**Example**: A 100MB XML file with 111,313 records can be processed with only ~68MB peak memory, demonstrating true streaming architecture.

### Vectorized Operations

Pandas operations are vectorized for performance:

```python
# Vectorized redaction
df['ssn'] = df['ssn'].apply(redact_ssn)
```

### Batch Processing

Records are processed in batches to balance memory and throughput:

```python
batch = []
for record in records:
    batch.append(record)
    if len(batch) >= batch_size:
        persist_batch(batch)
        batch = []
```

## 🧪 Testing Strategy

### Test Pyramid

```
        /\
       /  \  E2E Tests (few)
      /____\
     /      \  Integration Tests (some)
    /________\
   /          \  Unit Tests (many)
  /____________\
```

### Test Types

1. **Unit Tests** - Test domain logic in isolation
2. **Integration Tests** - Test adapter interactions
3. **Adversarial Tests** - Test security features
4. **Property-Based Tests** - Test with Hypothesis

### Mocking Strategy

- **Domain Core**: No mocks needed (pure functions)
- **Adapters**: Mock file I/O and database
- **Infrastructure**: Mock external services

## 🔧 Configuration Management

### Environment Variables

Configuration is loaded from environment variables with sensible defaults:

```python
# src/infrastructure/settings.py
class Settings:
    db_type: str = "duckdb"
    batch_size: int = 1000
    circuit_breaker_enabled: bool = True
    # ...
```

### Configuration Manager

The `ConfigManager` provides:
- Type-safe configuration access
- Environment-based overrides
- Validation
- Default values

## 📈 Scalability

### Horizontal Scaling

The architecture supports horizontal scaling:

1. **Stateless Processing** - No shared state between requests
2. **Adapter Pattern** - Can swap storage backends
3. **Streaming** - Can process files in parallel

### Vertical Scaling

Optimized for single-machine performance:

1. **Streaming** - O(record_size) memory
2. **Vectorization** - Efficient Pandas operations
3. **Batch Processing** - Balanced throughput

## 🔮 Future Enhancements

Potential architectural improvements:

1. **Message Queue Integration** - For distributed processing
2. **Web API** - REST/GraphQL interface
3. **Real-time Processing** - Stream processing with Kafka
4. **Multi-tenant Support** - Tenant isolation
5. **Plugin System** - Custom adapters via plugins

## 📚 References

- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Pydantic V2](https://docs.pydantic.dev/)
- [Threat Modeling](THREAT_MODEL.md)

---

**Last Updated:** January 2025

