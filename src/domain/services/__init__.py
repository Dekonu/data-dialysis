"""Domain Services.

This package contains domain services that implement business logic
without infrastructure dependencies.

Note: RedactorService is still in src/domain/services.py (the old file)
for backward compatibility. This __init__.py re-exports it along with
new services in the package.
"""

# Import RedactorService from the parent module's services.py file
# We import the file directly to avoid circular imports (since this package
# has the same name as the file, we need to load it explicitly)
import importlib.util
from pathlib import Path

_parent_path = Path(__file__).parent.parent
_services_file = _parent_path / "services.py"

if _services_file.exists():
    _spec = importlib.util.spec_from_file_location("domain_services_file", _services_file)
    _services_file_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_services_file_module)
    RedactorService = _services_file_module.RedactorService
else:
    raise ImportError("Could not find src/domain/services.py file")

# Import from the new services package
from src.domain.services.change_detector import ChangeDetector

__all__ = ['RedactorService', 'ChangeDetector']
