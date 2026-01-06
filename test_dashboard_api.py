"""Quick test script to verify dashboard API setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from dashboard.api.main import app
    print("[OK] FastAPI app imported successfully")
    print(f"[OK] App title: {app.title}")
    print(f"[OK] App version: {app.version}")
    print(f"[OK] Number of routes: {len(app.routes)}")
    print("\nAvailable routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = ', '.join(route.methods)
            print(f"  {methods:10} {route.path}")
    print("\n[OK] Dashboard API setup complete!")
    print("\nTo start the server, run:")
    print("  uvicorn src.dashboard.api.main:app --reload --port 8000")
    print("\nOr:")
    print("  python -m src.dashboard.api.main")
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

