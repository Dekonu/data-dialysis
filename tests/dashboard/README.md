# Dashboard API Tests

Comprehensive test suite for the Data-Dialysis Dashboard API.

## Test Coverage

### Phase 1 Endpoints (33 tests)

#### Root Endpoint (2 tests)
- ✅ Root endpoint returns API information
- ✅ Root endpoint has correct structure

#### Health Check Endpoint (8 tests)
- ✅ Returns 200 status
- ✅ Has all required fields
- ✅ Returns valid status values
- ✅ Timestamp format is valid
- ✅ Returns correct database type
- ✅ Handles connected database
- ✅ Handles disconnected database
- ✅ May include response time

#### Metrics Endpoints (6 tests)
- ✅ Overview metrics endpoint exists
- ✅ Overview metrics accepts time_range parameter
- ✅ Security metrics endpoint exists
- ✅ Security metrics accepts time_range parameter
- ✅ Performance metrics endpoint exists
- ✅ Performance metrics accepts time_range parameter

#### Audit Endpoints (5 tests)
- ✅ Audit logs endpoint exists
- ✅ Audit logs accepts pagination parameters
- ✅ Audit logs accepts filter parameters
- ✅ Redaction logs endpoint exists
- ✅ Redaction logs accepts filter parameters

#### Circuit Breaker Endpoint (1 test)
- ✅ Circuit breaker status endpoint exists

#### WebSocket Endpoint (1 test)
- ✅ WebSocket endpoint exists

#### Middleware (2 tests)
- ✅ Logging middleware adds process time header
- ✅ Error handling middleware handles errors

#### CORS (2 tests)
- ✅ CORS headers are present
- ✅ OPTIONS request works

#### API Documentation (3 tests)
- ✅ OpenAPI schema endpoint
- ✅ Swagger UI endpoint
- ✅ ReDoc endpoint

#### Error Handling (2 tests)
- ✅ Invalid endpoint returns 404
- ✅ Method not allowed returns 405

#### Response Format (1 test)
- ✅ All responses are JSON

## Running Tests

### Run all dashboard tests
```bash
pytest tests/dashboard/ -v
```

### Run specific test class
```bash
pytest tests/dashboard/test_dashboard_api.py::TestHealthEndpoint -v
```

### Run with coverage
```bash
pytest tests/dashboard/ --cov=src.dashboard --cov-report=html
```

## Test Structure

Tests are organized by endpoint/feature:
- `TestRootEndpoint` - Root endpoint tests
- `TestHealthEndpoint` - Health check tests
- `TestMetricsEndpoints` - Metrics endpoint tests
- `TestAuditEndpoints` - Audit log endpoint tests
- `TestCircuitBreakerEndpoint` - Circuit breaker tests
- `TestWebSocketEndpoint` - WebSocket tests
- `TestMiddleware` - Middleware tests
- `TestCORS` - CORS tests
- `TestAPIDocumentation` - Documentation endpoint tests
- `TestErrorHandling` - Error handling tests
- `TestResponseFormat` - Response format tests

## Fixtures

- `mock_storage_adapter` - Mock storage adapter for testing
- `client` - TestClient with mocked dependencies

## Test Principles

1. **Isolation**: Each test is independent
2. **Mocking**: External dependencies are mocked
3. **Coverage**: All endpoints and edge cases tested
4. **Assertions**: Clear, specific assertions
5. **Documentation**: Tests serve as documentation

## Future Tests (Phase 2+)

- Metrics aggregation service tests
- Security metrics calculation tests
- Performance metrics calculation tests
- Real WebSocket connection tests
- Integration tests with real database

