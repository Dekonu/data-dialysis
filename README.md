# 🛡️ Data-Dialysis: Self-Securing Data Ingestion Engine

<div align="center">

**A production-ready, security-first data pipeline for ingesting clinical and sensitive data with automatic PII redaction, schema validation, and real-time observability.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## About This Project

Data-Dialysis is a **portfolio project** demonstrating production-grade software engineering: Hexagonal Architecture, security-first design (HIPAA/GDPR), streaming data pipelines, and a full-stack observability dashboard. It showcases current industry practices and technologies—Pydantic V2, FastAPI, Next.js App Router, TypeScript, DuckDB, WebSockets—applied to a realistic clinical data ingestion problem with PII redaction, change data capture, and circuit breakers.

---

## Abstract

Data-Dialysis is a **production-grade data ingestion system** that implements Hexagonal (Ports & Adapters) architecture to process clinical and sensitive datasets with automatic PII redaction, schema validation, and comprehensive threat protection. Highlights:

- **Security-first design**: Multi-layer defense (defusedxml, streaming, validation, circuit breakers) against XML attacks, injection, and resource exhaustion
- **Scalable processing**: Streaming architecture with O(record_size) memory usage—validated with 100MB+ files and a full benchmarking suite
- **Change Data Capture**: Field-level change tracking with encrypted raw data vault for audit trails
- **Real-time dashboard**: FastAPI backend + Next.js 16 frontend with WebSocket-driven metrics, audit logs, and security views
- **Quantified performance**: Academic-style benchmark suite (CSV/JSON/XML, multiple sizes) with automated visualizations (throughput, memory, latency, format comparison)

**Key technical achievements:**
- Processes **100MB+ XML files** with constant memory (~50–100MB peak) via streaming
- **Verify-then-load** pipeline: data cannot reach persistence without passing validation and redaction
- **Benchmark suite** with throughput, memory profiling, batch statistics, and publication-quality charts

---

## 🚀 Quick Start

### Installation

```bash
# Clone and enter project
cd DataDialysis

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```bash
# Ingest a CSV file
datadialysis ingest data/patients.csv

# Ingest XML (requires config)
datadialysis ingest data/encounters.xml --xml-config xml_config.json

# Ingest with custom batch size
datadialysis ingest data/observations.json --batch-size 5000

# Run performance benchmarks
datadialysis benchmark test_data/ xml_config.json

# Start dashboard (backend + frontend via Docker, or run separately)
docker-compose up -d
```

---

## 🛠️ Tech Stack (Current Technologies)

| Layer | Technologies |
|-------|--------------|
| **Language & validation** | Python 3.11+, **Pydantic V2** (strict schemas, field validators) |
| **API & async** | **FastAPI**, async/await, lifespan, dependency injection |
| **Data processing** | **Pandas** (vectorized), streaming parsers, chunked I/O |
| **Databases** | **DuckDB** (analytical), **PostgreSQL** + **SQLAlchemy 2.0** |
| **XML security** | **defusedxml**, lxml (streaming) |
| **CLI** | **Typer**, **Rich** (type-safe, modern CLI) |
| **Dashboard backend** | FastAPI, **WebSockets**, Pydantic response models |
| **Dashboard frontend** | **Next.js 16** (App Router), **React 19**, **TypeScript**, **Tailwind CSS**, **Radix UI**, **Recharts** |
| **Testing** | **pytest**, **pytest-asyncio**, **Hypothesis** (property-based) |
| **Benchmarking** | **tracemalloc**, **psutil**, **Matplotlib**, **Seaborn** (visualizations) |
| **Security** | **cryptography**, NER (e.g. spaCy) for PII in free text |

---

## 🏗️ Architecture

### Hexagonal (Ports & Adapters)

Business logic lives in the **domain core**; infrastructure (ingestion, storage, API) is behind **ports** and swappable **adapters**.

```
┌─────────────────────────────────────────┐
│         Domain Core (Pure Python)       │
│  GoldenRecord (Pydantic), Redactor,     │
│  CircuitBreaker, Change Detector         │
└─────────────────────────────────────────┘
           ↕ Ports (protocols)
┌─────────────────────────────────────────┐
│      Adapters (Infrastructure)          │
│  CSV/JSON/XML ingesters • DuckDB/       │
│  PostgreSQL • FastAPI dashboard API     │
└─────────────────────────────────────────┘
```

**Benefits:** Testable core without DBs, pluggable adapters, clear security boundary.

### Verify-Then-Load Data Flow

```
Input (CSV/JSON/XML) → Secure parsing → PII redaction → Schema validation (Pydantic)
    → Circuit breaker check → Persistence (parameterized, transactional) → DB
```

Data cannot reach the database without passing validation and redaction.

---

## 🛡️ Security & Threat Model

The system is designed to process **untrusted input** while supporting HIPAA/GDPR expectations. See **[THREAT_MODEL.md](THREAT_MODEL.md)** for attack vectors and defenses.

- **XML**: defusedxml (Billion Laughs, XXE), streaming to limit memory and CPU blowup  
- **PII**: Regex + NER redaction, irreversible with audit trail  
- **Data quality**: Strict Pydantic validation; **circuit breaker** stops ingestion if failure rate exceeds threshold  
- **Persistence**: Parameterized queries, transaction safety, audit logging  

---

## 📊 Performance & Benchmarking

### Benchmark Suite

The **`performance_benchmark/`** suite provides repeatable, multi-format evaluation:

- **Scripts**: `academic_benchmark_suite.py` (orchestrates runs), `visualize_benchmark_results.py` (charts from CSV)
- **Formats**: CSV, JSON, XML (configurable sizes, e.g. 1MB–100MB+)
- **Metrics**: Throughput (records/s, MB/s), peak/avg memory, processing/upload/ingestion times, batch stats, success rate
- **Output**: `benchmark_results.csv` plus `benchmark_visualizations/` (throughput vs size, memory efficiency, format comparison, scalability, heatmaps, etc.)

```bash
# From project root (with test data and xml_config.json in place)
python performance_benchmark/academic_benchmark_suite.py test_data/ xml_config.json --output benchmark_results.csv

# Generate visualizations from existing results
python performance_benchmark/visualize_benchmark_results.py benchmark_results.csv --output-dir benchmark_visualizations
```

**See [docs/PERFORMANCE_BENCHMARKING.md](docs/PERFORMANCE_BENCHMARKING.md)** for methodology and interpretation.

### Scalability

- **Streaming XML**: O(record_size) memory; 100MB+ files at ~50–100MB peak RAM  
- **Batch tuning**: Configurable batch sizes; benchmark suite includes batch-size and format comparison  

---

## 📺 Real-Time Dashboard

The **dashboard** gives operational visibility over the pipeline:

- **Backend**: FastAPI app in `src/dashboard/api/` — REST endpoints for metrics, audit log, change history, circuit breaker status; **WebSockets** for live updates  
- **Frontend**: **Next.js 16** (App Router), **React 19**, **TypeScript**, **Tailwind**, **Radix UI**, **Recharts** in `dashboard-frontend/`  
- **Views**: Overview, performance, security metrics, audit log, change history, circuit breaker status  

Run with **Docker** (`docker-compose up`) or run backend and frontend separately (see [docs/DASHBOARD_DESIGN.md](docs/DASHBOARD_DESIGN.md)).

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers, ports, adapters, design rationale |
| [THREAT_MODEL.md](THREAT_MODEL.md) | Threat model, defenses, compliance notes |
| [docs/PERFORMANCE_BENCHMARKING.md](docs/PERFORMANCE_BENCHMARKING.md) | Benchmark methodology and metrics |
| [docs/README.md](docs/README.md) | Design docs index |
| [docs/XML_STREAMING_DESIGN.md](docs/XML_STREAMING_DESIGN.md) | Streaming XML parser design |
| [docs/REDACTION_LOGGING.md](docs/REDACTION_LOGGING.md) | PII redaction and audit trail |
| [docs/CHANGE_DATA_CAPTURE_PLAN.md](docs/CHANGE_DATA_CAPTURE_PLAN.md) | CDC and field-level change tracking |
| [docs/DASHBOARD_DESIGN.md](docs/DASHBOARD_DESIGN.md) | Dashboard architecture and APIs |

---

## 🔧 Configuration

Key environment variables (see `.env.example` or inline below):

```bash
# Database
DD_DB_TYPE=duckdb
DD_DB_PATH=./data/clinical.db

# Processing
DD_BATCH_SIZE=1000
DD_XML_STREAMING_ENABLED=true
DD_XML_STREAMING_THRESHOLD=104857600   # 100MB

# Safety
DD_CIRCUIT_BREAKER_ENABLED=true
DD_CIRCUIT_BREAKER_THRESHOLD=0.1
DD_LOG_LEVEL=INFO
```

XML ingestion uses a JSON config for XPath → field mapping; see **`xml_config.json`** in the repo for the structure (`root_element`, `fields` with XPath values).

---

## 🧪 Testing

```bash
pytest
pytest --cov=src --cov-report=html
pytest tests/integration/ -v
```

- **Unit**: Domain, adapters, infrastructure  
- **Integration**: CSV/JSON/XML ingestion, DuckDB/PostgreSQL, security (bad data, circuit breaker)  
- **Adversarial**: Malformed XML, injection attempts, schema violations  
- **Property-based**: Hypothesis where applicable  

---

## 📝 Examples

**CLI:**

```bash
datadialysis ingest data/patients.csv
datadialysis ingest data/encounters.xml --xml-config xml_config.json --batch-size 2000
datadialysis info
```

**Programmatic:**

```python
from src.adapters.ingesters import get_adapter

adapter = get_adapter("data/patients.csv")
for result in adapter.ingest("data/patients.csv"):
    if result.is_success():
        print(result.value.patient.patient_id)
    else:
        print(result.error)
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions should include tests (including adversarial cases where relevant), preserve Hexagonal boundaries, and document security impact where applicable.

---

## 📄 License

See [LICENSE](LICENSE).

---

## Portfolio & Skills Demonstrated

This project illustrates practices and technologies relevant to **mid- to senior-level** roles in data engineering, backend services, and platform/security-aware applications:

- **Architecture**: Hexagonal/ports-and-adapters, clear boundaries, testability  
- **Security**: Threat-aware design, PII handling, secure parsing, circuit breakers, audit trails  
- **Data engineering**: Streaming pipelines, CDC, multi-format ingestion, benchmarking  
- **Modern Python**: 3.11+, Pydantic V2, async FastAPI, type hints  
- **Full-stack**: FastAPI + Next.js, TypeScript, REST + WebSockets  
- **Quality**: pytest, Hypothesis, benchmark suite, reproducible visualizations  

**Version:** 1.0.0 · **Last updated:** February 2026
