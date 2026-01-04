# Contributing to Data-Dialysis

Thank you for your interest in contributing to Data-Dialysis! This document provides guidelines and instructions for contributing to this security-critical project.

## 🎯 Getting Started

1. **Fork the repository** and clone your fork
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```
4. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

## 📋 Development Workflow

### 1. Create a Branch

Create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow these principles:

- **Security First**: All changes must maintain or improve security posture
- **Hexagonal Architecture**: Keep domain logic separate from infrastructure
- **Type Hints**: Use Python type hints for all function signatures
- **Documentation**: Add docstrings explaining security impact
- **Tests**: Write tests before or alongside code changes

### 3. Write Tests

**Required Test Coverage:**
- Unit tests for domain logic
- Integration tests for adapters
- Adversarial tests for security features
- Property-based tests with Hypothesis where appropriate

**Test Structure:**
```python
def test_feature_name():
    """Test description explaining what is being tested."""
    # Arrange
    # Act
    # Assert
```

**Run Tests:**
```bash
# All tests
pytest

# Specific test file
pytest tests/test_feature.py

# With coverage
pytest --cov=src --cov-report=html
```

### 4. Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### 5. Commit Changes

Use clear, descriptive commit messages:
```
feat: Add XML streaming parser for large files
fix: Prevent XXE attacks in XML parsing
docs: Update architecture documentation
test: Add adversarial tests for circuit breaker
```

**Commit Message Format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `security:` - Security improvements
- `perf:` - Performance improvements

### 6. Submit Pull Request

1. **Push your branch** to your fork
2. **Create a Pull Request** with:
   - Clear description of changes
   - Security impact assessment
   - Test coverage summary
   - Any breaking changes

## 🛡️ Security Guidelines

### Security-Critical Areas

These areas require extra attention:

1. **PII Redaction** (`src/domain/services.py`)
   - Must be irreversible
   - Must maintain audit trail
   - Must handle edge cases

2. **XML Parsing** (`src/adapters/ingesters/xml_ingester.py`)
   - Must use `defusedxml`
   - Must handle streaming correctly
   - Must prevent resource exhaustion

3. **Schema Validation** (`src/domain/golden_record.py`)
   - Must be strict
   - Must fail fast
   - Must prevent injection

4. **Circuit Breaker** (`src/domain/guardrails.py`)
   - Must accurately track failures
   - Must prevent data poisoning
   - Must be configurable

### Security Review Process

All PRs will be reviewed for:
- Security vulnerabilities
- PII handling correctness
- Threat model compliance
- Test coverage (especially adversarial tests)

## 🏗️ Architecture Guidelines

### Hexagonal Architecture

**Domain Core** (`src/domain/`):
- Pure Python, no external dependencies
- Business logic only
- Testable without infrastructure

**Adapters** (`src/adapters/`):
- Implement Port interfaces
- Handle I/O (files, databases)
- Can be swapped without changing domain

**Infrastructure** (`src/infrastructure/`):
- Configuration management
- Logging and monitoring
- Cross-cutting concerns

### Adding a New Adapter

1. **Define Port** in `src/domain/ports.py`
2. **Implement Adapter** in `src/adapters/`
3. **Add Tests** in `tests/`
4. **Update Documentation**

Example:
```python
# src/domain/ports.py
class NewSourcePort(Protocol):
    def ingest(self, source: str) -> Iterator[Result[GoldenRecord]]: ...

# src/adapters/ingesters/new_ingester.py
class NewIngester:
    def ingest(self, source: str) -> Iterator[Result[GoldenRecord]]:
        # Implementation
        pass
```

## 📝 Documentation

### Code Documentation

- **Docstrings**: All public functions must have docstrings
- **Security Impact**: Document security implications
- **Parameters**: Document all parameters and return types
- **Examples**: Include usage examples for complex functions

### Architecture Documentation

- Update `ARCHITECTURE.md` for architectural changes
- Update `THREAT_MODEL.md` for security changes
- Update relevant docs in `docs/` directory

## 🧪 Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test adapter interactions
3. **Adversarial Tests**: Test security features
4. **Property-Based Tests**: Test with Hypothesis

### Adversarial Test Examples

```python
def test_xml_billion_laughs_attack():
    """Test that Billion Laughs attack is prevented."""
    # Test implementation
    pass

def test_pii_leakage_prevention():
    """Test that PII cannot leak through error messages."""
    # Test implementation
    pass

def test_circuit_breaker_prevents_poisoning():
    """Test that circuit breaker halts on bad data."""
    # Test implementation
    pass
```

## ✅ Checklist for PRs

Before submitting a PR, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black`)
- [ ] Linting passes (`ruff`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation is updated
- [ ] Security impact is documented
- [ ] Adversarial tests are included (if applicable)
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow conventions

## 🐛 Reporting Issues

### Security Issues

**Do not** open public issues for security vulnerabilities. Instead:
1. Email security concerns privately
2. Include detailed description
3. Include steps to reproduce
4. Include potential impact

### Bug Reports

Include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Relevant logs/error messages

### Feature Requests

Include:
- Use case description
- Proposed solution
- Security considerations
- Implementation approach

## 📚 Resources

- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/)
- [Threat Modeling Guide](THREAT_MODEL.md)
- [Architecture Overview](ARCHITECTURE.md)

## 🙏 Thank You

Your contributions help make Data-Dialysis more secure, robust, and useful. We appreciate your time and effort!

