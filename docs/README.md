# Documentation

This directory contains detailed design and implementation documentation for Data-Dialysis.

## 📚 Documentation Index

### Design Documents

- **[XML_STREAMING_DESIGN.md](XML_STREAMING_DESIGN.md)** - Design rationale and architecture for XML streaming parser
- **[XML_STREAMING_IMPLEMENTATION.md](XML_STREAMING_IMPLEMENTATION.md)** - Implementation details and code walkthrough
- **[REDACTION_LOGGING.md](REDACTION_LOGGING.md)** - PII redaction system design and audit trail architecture
- **[HYBRID_REDACTION_LOGGING_IMPLEMENTATION.md](HYBRID_REDACTION_LOGGING_IMPLEMENTATION.md)** - Implementation details for hybrid redaction logging
- **[CONFIG_MANAGER_DESIGN.md](CONFIG_MANAGER_DESIGN.md)** - Configuration management system design

### Root-Level Documentation

- **[../README.md](../README.md)** - Project overview and quick start guide
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** - Comprehensive architecture overview
- **[../THREAT_MODEL.md](../THREAT_MODEL.md)** - Security threat model and defense mechanisms
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines

## 🎯 Documentation Organization

### Root-Level Documentation (High-Level, User-Facing)

These documents are in the repository root for maximum discoverability:

- **[../README.md](../README.md)** - Main entry point and quick start guide
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** - Comprehensive architecture overview (high-level)
- **[../THREAT_MODEL.md](../THREAT_MODEL.md)** - Security threat model and defense mechanisms
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines (GitHub standard location)

### Feature-Specific Documentation (Detailed Design & Implementation)

These documents in `docs/` provide deep-dive details on specific features:

- **Design Documents** - Explain *why* design decisions were made for specific features
- **Implementation Documents** - Explain *how* features are implemented in detail

## 📚 Documentation Purpose

These documents serve different purposes:

1. **Design Documents** - Explain *why* decisions were made
2. **Implementation Documents** - Explain *how* features are implemented
3. **Architecture Documents** - Explain the overall system structure
4. **Security Documents** - Explain threat model and defenses

## 📖 Reading Guide

**For New Contributors:**
1. Start with [README.md](../README.md)
2. Read [ARCHITECTURE.md](../ARCHITECTURE.md)
3. Review relevant design documents for features you're working on

**For Security Review:**
1. Read [THREAT_MODEL.md](../THREAT_MODEL.md)
2. Review [REDACTION_LOGGING.md](REDACTION_LOGGING.md)
3. Check implementation documents for security considerations

**For Performance Optimization:**
1. Read [XML_STREAMING_DESIGN.md](XML_STREAMING_DESIGN.md)
2. Review [XML_STREAMING_IMPLEMENTATION.md](XML_STREAMING_IMPLEMENTATION.md)

## 🔄 Keeping Documentation Updated

When making changes:

1. **Architectural Changes** → Update [ARCHITECTURE.md](../ARCHITECTURE.md)
2. **Security Changes** → Update [THREAT_MODEL.md](../THREAT_MODEL.md)
3. **New Features** → Create or update relevant design document
4. **Implementation Changes** → Update implementation documents

## 📝 Documentation Standards

- Use clear, concise language
- Include code examples where helpful
- Document security implications
- Explain design decisions
- Keep diagrams up to date

