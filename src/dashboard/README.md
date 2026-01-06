# Data-Dialysis Dashboard API

FastAPI-based backend for the Data-Dialysis health monitoring dashboard.

## Quick Start

### Start the Development Server

```bash
# Activate virtual environment
.\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Start the server
uvicorn src.dashboard.api.main:app --reload --port 8000
```

Or run directly:

```bash
python -m src.dashboard.api.main
```

### Access the API

- **API Root**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/api/docs
- **API Docs (ReDoc)**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/api/health

## Project Structure

```
src/dashboard/
├── api/
│   ├── main.py              # FastAPI application
│   ├── dependencies.py      # Dependency injection
│   ├── middleware.py         # Request/response middleware
│   └── routes/
│       ├── health.py         # Health check endpoint
│       ├── metrics.py        # Metrics endpoints (Phase 2)
│       ├── audit.py          # Audit log endpoints (Phase 3)
│       ├── circuit_breaker.py # Circuit breaker status (Phase 3)
│       └── websocket.py      # WebSocket for real-time (Phase 4)
├── models/
│   └── health.py            # Pydantic models for health check
└── services/
    └── (to be implemented in Phase 2)
```

## API Endpoints

### Implemented (Phase 1)

- `GET /` - Root endpoint
- `GET /api/health` - Health check with database status

### Placeholder Endpoints (To be implemented)

- `GET /api/metrics/overview` - Overview metrics (Phase 2)
- `GET /api/metrics/security` - Security metrics (Phase 2)
- `GET /api/metrics/performance` - Performance metrics (Phase 2)
- `GET /api/audit-logs` - Audit log explorer (Phase 3)
- `GET /api/redaction-logs` - Redaction log explorer (Phase 3)
- `GET /api/circuit-breaker/status` - Circuit breaker status (Phase 3)
- `WS /ws/realtime` - WebSocket for real-time updates (Phase 4)

## Configuration

The dashboard uses the same configuration as the main Data-Dialysis pipeline:

- Database configuration via `DD_DB_TYPE`, `DD_DB_HOST`, etc.
- Environment variables from `.env` file (if present)
- Configuration manager from `src.infrastructure.config_manager`

## Development

### Testing

```bash
# Test API import
python test_dashboard_api.py

# Run tests (when implemented)
pytest tests/dashboard/
```

### Adding New Endpoints

1. Create route file in `src/dashboard/api/routes/`
2. Define router with `APIRouter(prefix="/api/...", tags=[...])`
3. Add route handlers with proper type hints
4. Include router in `src/dashboard/api/main.py`
5. Add Pydantic models in `src/dashboard/models/` if needed

## Architecture

The dashboard follows Hexagonal Architecture principles:

- **API Layer** (`api/`): FastAPI routes and middleware
- **Models Layer** (`models/`): Pydantic response models
- **Services Layer** (`services/`): Business logic (to be implemented)
- **Adapters**: Uses existing storage adapters via dependency injection

## Security

- CORS configured for frontend origins
- Request/response logging for audit trail
- Error handling without exposing sensitive information
- Uses existing secure storage adapters

## Next Steps

See `docs/DASHBOARD_IMPLEMENTATION_PLAN.md` for Phase 2 implementation details.

