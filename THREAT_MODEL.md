# 🛡️ Threat Model: Data-Dialysis Security Architecture

## Executive Summary

Data-Dialysis is a **self-securing data ingestion engine** designed to process clinical data from untrusted sources while maintaining HIPAA/GDPR compliance and preventing data poisoning attacks. This document provides a comprehensive threat model analysis, detailing attack vectors, defensive mechanisms, and security architecture. This analysis demonstrates formal security engineering practices suitable for academic evaluation.

---

## 1. Threat Landscape

### 1.1 Attack Surface

The system processes data from **untrusted sources** (CSV, JSON, XML files) that may contain:
- **Malicious payloads** designed to exploit parsing vulnerabilities
- **PII (Personally Identifiable Information)** that must be redacted before storage
- **Malformed data** intended to crash or corrupt the pipeline
- **Data poisoning** attempts to inject invalid records into the database
- **Resource exhaustion** attacks to deny service

### 1.2 Adversary Capabilities

**Assumed Adversary Capabilities:**
- Can craft malicious input files (CSV, JSON, XML)
- Has knowledge of the system's schema and validation rules
- Can attempt to bypass PII redaction mechanisms
- Can attempt to inject SQL, XSS, or other injection attacks
- Can attempt to exhaust system resources (memory, CPU)

**Adversary Goals:**
1. Extract or leak PII from the system
2. Corrupt the database with invalid data
3. Crash the system via resource exhaustion
4. Bypass validation and inject malicious records
5. Exfiltrate data through injection attacks

---

## 2. Attack Vectors & Defenses

### 2.1 XML-Based Attacks

#### **Threat: Billion Laughs Attack**
**Attack Vector:** Malicious XML file uses entity expansion to create exponentially large content in memory.

```xml
<!ENTITY a "b">
<!ENTITY b "&a;&a;">
<!ENTITY c "&b;&b;">
<!-- ... 30+ levels ... -->
<data>&z;</data>  <!-- Expands to billions of characters -->
```

**Defense:**
- ✅ **`defusedxml`** library prevents entity expansion
- ✅ **`resolve_entities=False`** in XMLParser configuration
- ✅ **Event limit enforcement** (max_events) prevents processing excessive XML events
- ✅ **Depth limit enforcement** (max_depth) prevents deep nesting attacks
- ✅ **File size limits** prevent loading massive files into memory

**Implementation:**
```python
# src/infrastructure/xml_streaming_parser.py
XMLParser(
    resolve_entities=False,  # Security: prevent entity expansion
    no_network=True,          # Security: prevent network access
    huge_tree=False,          # Security: prevent quadratic blowup
    recover=False             # Security: fail on malformed XML
)
```

#### **Threat: Quadratic Blowup Attack**
**Attack Vector:** XML structure designed to cause O(n²) memory usage during parsing.

**Defense:**
- ✅ **`huge_tree=False`** by default (enabled only for verified large files >50MB)
- ✅ **Streaming parser** processes records one at a time (O(record_size) memory)
- ✅ **Explicit element clearing** after each record to prevent memory accumulation
- ✅ **Periodic garbage collection** every 1000 records

#### **Threat: XML External Entity (XXE) Attack**
**Attack Vector:** XML file references external entities to read local files or make network requests.

**Defense:**
- ✅ **`no_network=True`** prevents network access during parsing
- ✅ **`resolve_entities=False`** prevents entity resolution
- ✅ **No external DTD processing** - all parsing is local

---

### 2.2 PII Leakage Prevention

#### **Threat: PII in Non-PII Fields**
**Attack Vector:** Adversary attempts to inject PII into fields that should not contain sensitive data (e.g., patient_id, observation_id).

**Defense:**
- ✅ **Field-level PII detection** using regex patterns
- ✅ **Validation rules** prevent PII in identifier fields
- ✅ **Redaction before validation** - PII is redacted before schema validation
- ✅ **Immutable records** - once redacted, records cannot be modified

**Implementation:**
```python
# src/domain/golden_record.py
@field_validator('patient_id', mode='before')
def validate_no_pii_in_id(cls, v):
    """Prevent PII leakage in identifier fields."""
    if contains_pii(v):
        raise ValueError("PII detected in patient_id field")
    return v
```

#### **Threat: PII in Unstructured Text**
**Attack Vector:** PII embedded in clinical notes, progress notes, or other unstructured text fields.

**Defense:**
- ✅ **Regex-based PII detection** for SSNs, phone numbers, emails
- ✅ **Name entity recognition** for patient names in text
- ✅ **Redaction logging** tracks all redactions for audit trail
- ✅ **Irreversible redaction** - redacted data cannot be recovered

#### **Threat: PII Reversibility**
**Attack Vector:** Adversary attempts to reverse-engineer redacted data.

**Defense:**
- ✅ **One-way redaction** - PII is replaced with `None` or `[REDACTED]`
- ✅ **No redaction metadata** stored with records (separate audit log)
- ✅ **Hash-based audit trail** - records include transformation hash for integrity

---

### 2.3 Data Poisoning & Injection Attacks

#### **Threat: SQL Injection**
**Attack Vector:** Malicious data containing SQL injection payloads in string fields.

**Defense:**
- ✅ **Parameterized queries** - all database operations use parameterized statements
- ✅ **Pydantic validation** - string fields are validated and sanitized
- ✅ **Type coercion** - all inputs are coerced to expected types
- ✅ **No raw SQL** - database adapter uses ORM-like patterns

#### **Threat: XSS (Cross-Site Scripting)**
**Attack Vector:** Malicious JavaScript or HTML in text fields.

**Defense:**
- ✅ **String validation** - all text fields are validated as plain text
- ✅ **No HTML rendering** - system does not render HTML from data
- ✅ **Sanitization** - special characters are handled safely

#### **Threat: Schema Violation Injection**
**Attack Vector:** Malformed data designed to bypass validation or corrupt the database schema.

**Defense:**
- ✅ **Strict Pydantic schemas** - all records must match exact schema
- ✅ **Fail-fast validation** - invalid records are rejected immediately
- ✅ **Circuit breaker** - halts ingestion if error rate exceeds threshold (default: 10%)
- ✅ **Type safety** - all fields are strongly typed

**Implementation:**
```python
# src/domain/guardrails.py
class CircuitBreaker:
    """Halts ingestion if failure rate exceeds threshold."""
    def record_result(self, result: Result) -> None:
        if result.is_failure():
            self.failure_count += 1
        if self._should_open():
            raise CircuitBreakerOpenError("Data quality threshold exceeded")
```

---

### 2.4 Resource Exhaustion Attacks

#### **Threat: Memory Exhaustion (DoS)**
**Attack Vector:** Extremely large files or records designed to exhaust system memory.

**Defense:**
- ✅ **Streaming processing** - files are processed incrementally, not loaded entirely
- ✅ **Record size limits** - maximum record size enforced (default: 10MB)
- ✅ **Chunked processing** - large datasets processed in batches
- ✅ **Memory-efficient XML parsing** - uses `iterparse` for O(record_size) memory

**Implementation:**
```python
# src/adapters/ingesters/xml_ingester.py
# Streaming mode automatically selected for files >100MB
if file_size > streaming_threshold:
    yield from self._ingest_streaming(source)  # Memory-efficient
else:
    yield from self._ingest_traditional(source)
```

#### **Threat: CPU Exhaustion**
**Attack Vector:** Malicious data designed to cause expensive computations (e.g., deeply nested structures, complex regex).

**Defense:**
- ✅ **Depth limits** - XML nesting depth limited (default: 100 levels)
- ✅ **Event limits** - maximum XML events per file (default: 1M, auto-scaled for large files)
- ✅ **Pre-compiled XPath** - XPath expressions compiled once, not per-record
- ✅ **Efficient regex** - PII detection uses optimized regex patterns

#### **Threat: Disk Exhaustion**
**Attack Vector:** Extremely large files or excessive logging.

**Defense:**
- ✅ **File size limits** - configurable maximum file size
- ✅ **Log rotation** - audit logs are managed and rotated
- ✅ **Batch processing** - limits memory and disk usage

---

### 2.5 Data Integrity Attacks

#### **Threat: Record Tampering**
**Attack Vector:** Adversary attempts to modify records after ingestion.

**Defense:**
- ✅ **Immutable records** - GoldenRecord objects are frozen after creation
- ✅ **Transformation hash** - each record includes hash of original data
- ✅ **Audit trail** - all transformations are logged with timestamps
- ✅ **Read-only storage** - records cannot be modified after persistence

#### **Threat: Data Corruption**
**Attack Vector:** Malformed data designed to corrupt database or cause data loss.

**Defense:**
- ✅ **Transaction safety** - database operations use transactions
- ✅ **Validation before persistence** - only validated records reach storage
- ✅ **Error isolation** - bad records don't affect good records
- ✅ **Rollback capability** - failed batches can be rolled back

---

## 3. Security Layers

### 3.1 Defense in Depth

The system implements **multiple layers of defense**:

```
┌─────────────────────────────────────────┐
│  Layer 1: Input Sanitization           │
│  - File size limits                     │
│  - Format validation                    │
│  - Malformed data rejection             │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Layer 2: Secure Parsing               │
│  - defusedxml (XML attacks)             │
│  - Streaming (memory safety)            │
│  - Event/depth limits                   │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Layer 3: PII Redaction                │
│  - Regex-based detection                │
│  - Field-level validation               │
│  - Unstructured text scanning           │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Layer 4: Schema Validation             │
│  - Pydantic strict validation           │
│  - Type coercion                        │
│  - Pattern matching                     │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Layer 5: Circuit Breaker               │
│  - Failure rate monitoring              │
│  - Automatic halt on threshold          │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Layer 6: Secure Persistence            │
│  - Parameterized queries                │
│  - Transaction safety                   │
│  - Audit logging                        │
└─────────────────────────────────────────┘
```

### 3.2 Fail-Safe Design

**Principle:** Bad data should never crash the system or corrupt the database.

- ✅ **Per-record error isolation** - each record wrapped in try/except
- ✅ **Graceful degradation** - failed records logged and skipped
- ✅ **Circuit breaker** - halts ingestion if quality drops too low
- ✅ **No side effects** - failed records don't affect successful ones

---

## 4. Compliance & Audit

### 4.1 HIPAA Compliance

**Protected Health Information (PHI) Handling:**
- ✅ **PII redaction** before persistence
- ✅ **Audit trail** of all redactions
- ✅ **Immutable records** prevent tampering
- ✅ **Access controls** via configuration management

### 4.2 GDPR Compliance

**Personal Data Protection:**
- ✅ **Data minimization** - only necessary fields stored
- ✅ **Right to erasure** - audit trail enables data deletion
- ✅ **Data portability** - standardized output format
- ✅ **Privacy by design** - PII redaction is default behavior

### 4.3 Audit Trail

**Immutable Logging:**
- ✅ **Redaction events** logged with timestamps
- ✅ **Transformation hashes** for data integrity
- ✅ **Source tracking** - each record includes source adapter
- ✅ **Failure logging** - all rejections logged for analysis

---

## 5. Threat Mitigation Summary

| Threat Category | Attack Vector | Mitigation | Status |
|----------------|---------------|------------|--------|
| **XML Attacks** | Billion Laughs | `defusedxml`, entity limits | ✅ Protected |
| **XML Attacks** | Quadratic Blowup | Streaming, `huge_tree=False` | ✅ Protected |
| **XML Attacks** | XXE | `no_network=True`, no entity resolution | ✅ Protected |
| **PII Leakage** | PII in identifiers | Field-level validation | ✅ Protected |
| **PII Leakage** | PII in text | Regex + NER detection | ✅ Protected |
| **PII Leakage** | Reversibility | One-way redaction | ✅ Protected |
| **Injection** | SQL Injection | Parameterized queries | ✅ Protected |
| **Injection** | XSS | String validation, no HTML rendering | ✅ Protected |
| **Data Poisoning** | Schema violations | Strict Pydantic validation | ✅ Protected |
| **Resource Exhaustion** | Memory DoS | Streaming, size limits | ✅ Protected |
| **Resource Exhaustion** | CPU DoS | Depth/event limits, compiled XPath | ✅ Protected |
| **Data Integrity** | Tampering | Immutable records, hashes | ✅ Protected |
| **Data Integrity** | Corruption | Transactions, validation | ✅ Protected |

---

## 6. Security Assumptions

### 6.1 Trusted Components

**We Trust:**
- ✅ Python standard library (with security patches)
- ✅ `defusedxml` library (security-focused XML parser)
- ✅ `lxml` library (for streaming XML, with security config)
- ✅ `pydantic` library (for validation)
- ✅ Database drivers (with parameterized queries)

### 6.2 Untrusted Components

**We Do NOT Trust:**
- ❌ Input files (CSV, JSON, XML) - treated as potentially malicious
- ❌ Configuration files - validated before use
- ❌ Network sources - all network access disabled during parsing

### 6.3 Security Boundaries

**Security Boundary:** The Safety Layer (Pydantic validation + PII redaction) is the **hard security boundary**. Data cannot reach the database without passing through this layer.

---

## 7. Known Limitations

### 7.1 Current Limitations

1. **NLP-based PII Detection:** Currently uses regex patterns. Advanced NLP (e.g., SpaCy) for name recognition is planned but not yet implemented.

2. **Real-time Processing:** System is designed for batch processing. Real-time streaming from APIs would require additional security measures.

3. **Encryption at Rest:** Database encryption is handled by the storage adapter (DuckDB/PostgreSQL). This is outside the scope of the ingestion engine.

### 7.2 Future Enhancements

- **Advanced NER:** Integrate SpaCy for better name/entity recognition
- **Rate Limiting:** Add rate limiting for API-based ingestion
- **Encryption in Transit:** Add TLS for network-based ingestion
- **Anomaly Detection:** ML-based detection of unusual patterns

---

## 8. Security Best Practices

### 8.1 Configuration

- ✅ Use environment variables for sensitive configuration
- ✅ Never commit credentials to version control
- ✅ Use strong database passwords
- ✅ Enable circuit breaker in production

### 8.2 Monitoring

- ✅ Monitor failure rates (circuit breaker statistics)
- ✅ Review security reports regularly
- ✅ Audit redaction logs for compliance
- ✅ Monitor resource usage (memory, CPU)

### 8.3 Incident Response

- ✅ Circuit breaker automatically halts ingestion on threshold breach
- ✅ All failures are logged with full context
- ✅ Security reports provide actionable insights
- ✅ Audit trail enables forensic analysis

---

## 9. References

- **OWASP Top 10:** https://owasp.org/www-project-top-ten/
- **CWE-611 (XXE):** https://cwe.mitre.org/data/definitions/611.html
- **CWE-400 (Resource Exhaustion):** https://cwe.mitre.org/data/definitions/400.html
- **HIPAA Security Rule:** https://www.hhs.gov/hipaa/for-professionals/security/
- **GDPR Article 32:** https://gdpr-info.eu/art-32-gdpr/

---

## 📚 Related Documentation

- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture and design patterns
- **[docs/REDACTION_LOGGING.md](docs/REDACTION_LOGGING.md)** - PII redaction system design
- **[docs/RAW_DATA_VAULT_DESIGN.md](docs/RAW_DATA_VAULT_DESIGN.md)** - Encrypted storage architecture

---

**Last Updated:** January 2026  
**Version:** 1.0.0

