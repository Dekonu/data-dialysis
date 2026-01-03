🛡️ Project Roadmap: DataDialysis - Self-Securing Clinical Pipeline
This project is currently in active development. The goal is to move from a local prototype to a "Clinical-Grade" ingestion engine in 10 weeks, focusing on the Hexagonal Architecture and the "Safety-First" Sieve.

📍 Implementation Status
[x] Phase 1: Core Scaffolding (Current)

[ ] Phase 2: The Sieve (PII Redaction)

[ ] Phase 3: Persistence & Audit

[ ] Phase 4: Optimization & CLI

🛠️ Weekly Milestones
Phase 1: Architecture & Safety Layer (Weeks 1-2)
Goal: Establish the "Golden Record" and strict schema enforcement.

[x] Define Pydantic "Golden Record" schemas for clinical entities.

[ ] Implement defusedxml safety wrappers to prevent XML-based attacks.

[ ] Build the JSON Adapter (Input Port).

[ ] Achievement: Successfully block malformed records before they reach the core.

Phase 2: The Sieve - Static Analysis (Weeks 3-5)
Goal: Automated PII detection and data standardization.

[ ] Develop a Regex-based Redactor for SSNs, phone numbers, and IDs.

[ ] Integrate Pandas Vectorized Transformation for batch cleaning.

[ ] Implement a Circuit Breaker: Auto-kill ingestion if error rates exceed 10%.

[ ] Achievement: Ingest "dirty" data and output "clean" standardized data.

Phase 3: Persistence & Observability (Weeks 6-8)
Goal: Clinical-grade audit trails and analytical storage.

[ ] Implement DuckDB Adapter (Output Port) for local analytical storage.

[ ] Create an Immutable Audit Log: Track every redaction event with timestamps and hashes.

[ ] Build a Streaming XML Adapter using lxml.iterparse for memory efficiency.

[ ] Achievement: Process a 100MB+ file with a complete transformation history.

Phase 4: Refinement & Demo (Weeks 9-10)
Goal: Portfolio readiness and performance tuning.

[ ] Build a CLI Interface using Typer for easy engine execution.

[ ] Implement a Project Dashboard (Streamlit) to visualize ingestion health.

[ ] Comprehensive documentation of the Threat Model and Security Decisions.

[ ] Achievement: A production-ready portfolio piece.

🚀 Future Enhancements (Post-MVP)
NLP Integration: Use SpaCy or Med7 for advanced Named Entity Recognition in clinical notes.

Containerization: Dockerize the pipeline for deployment as a microservice.

FHIR Compatibility: Full mapping to HL7 FHIR R4 standards for industry interoperability.

How to follow this Roadmap
If you are an employer reviewing this project, you can track my progress through the docs/changelog.md or by exploring the tests/ directory, where I verify each new "Security Gate" as it is built.

🏗️ Architectural Decisions: Why Hexagonal?
This engine is built using the Hexagonal (Ports and Adapters) Architecture. In a clinical environment, requirements for data sources (XML, FHIR APIs) and storage (SQL, Cloud Buckets) change constantly, but the rules for patient safety and PII redaction remain a constant legal requirement.

1. Decoupling the "Sieve" (Core Logic)
The "Sieve"—my PII redaction and validation layer—is a Pure Python Domain. It has zero dependencies on databases or external APIs.

The Benefit: I can test the most high-stakes security logic in milliseconds without spinning up a database.

2. Interchangeable Adapters
By defining strict Ports, I can swap infrastructure without rewriting business logic:

Input Ports: Easily switch from a local .json file to a streaming Kafka queue or a hospital's SFTP server.

Output Ports: Currently uses DuckDB for local speed, but can be swapped for a HIPAA-compliant AWS RDS instance simply by writing a new Adapter.

3. Security as a Middleware
In this architecture, data cannot reach the database without passing through the Safety Layer. By placing the Sieve at the center of the Hexagon, security isn't a "feature" added at the end; it is the gatekeeper of the entire system.

🛡️ The "Self-Securing" Philosophy
Unlike traditional pipelines that "Extract, Load, then Transform" (ELT), this engine follows a Verify-Then-Load pattern.

Static Analysis: Pydantic enforces type safety and schema constraints at the boundary.

PII Sanitization: The Sieve redacts sensitive identifiers in-memory.

Persistence: Only "Sanitized" objects are ever permitted to touch the storage adapter.
