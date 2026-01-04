"""Ingestion adapters for Data-Dialysis.

This module contains ingestion adapters that implement the IngestionPort interface
for reading data from various sources (CSV, JSON, XML, etc.).
"""

from pathlib import Path
from typing import Optional

from src.adapters.ingesters.csv_ingester import CSVIngester
from src.adapters.ingesters.json_ingester import JSONIngester
from src.adapters.ingesters.xml_ingester import XMLIngester
from src.domain.ports import IngestionPort, UnsupportedSourceError

__all__ = ["CSVIngester", "JSONIngester", "XMLIngester", "get_adapter"]


def get_adapter(source: str, **kwargs) -> IngestionPort:
    """Factory function to get the appropriate ingestion adapter for a source.
    
    This function automatically selects the correct adapter based on the source
    file extension or content. It implements the adapter registry pattern.
    
    Parameters:
        source: Source identifier (file path, URL, etc.)
        **kwargs: Additional arguments passed to adapter constructors
            - For XML: config_path or config_dict
            - For CSV/JSON: chunk_size, max_record_size, etc.
    
    Returns:
        IngestionPort: Appropriate adapter instance
    
    Raises:
        UnsupportedSourceError: If no adapter can handle the source
        ValueError: If source cannot be determined
    
    Example Usage:
        ```python
        # Automatic adapter selection
        adapter = get_adapter("data.csv")
        
        # With XML config
        adapter = get_adapter("data.xml", config_path="mappings.json")
        
        # With custom chunk size
        adapter = get_adapter("data.json", chunk_size=5000)
        ```
    """
    source_path = Path(source)
    extension = source_path.suffix.lower()
    
    # Try each adapter to see if it can handle the source
    adapters = [
        ("csv", CSVIngester),
        ("json", JSONIngester),
        ("xml", XMLIngester),
    ]
    
    # First, try by file extension
    for ext, adapter_class in adapters:
        if extension == f".{ext}":
            try:
                if ext == "xml":
                    # XML requires config_path or config_dict
                    if "config_path" not in kwargs and "config_dict" not in kwargs:
                        raise ValueError(
                            "XML ingester requires either 'config_path' or 'config_dict' parameter. "
                            "Use --xml-config argument or pass config_path in kwargs."
                        )
                return adapter_class(**kwargs)
            except Exception as e:
                raise UnsupportedSourceError(
                    f"Failed to create {ext.upper()} adapter: {str(e)}",
                    source=source,
                    adapter=adapter_class.__name__
                )
    
    # If extension doesn't match, try can_ingest() method
    for ext, adapter_class in adapters:
        if ext == "xml":
            # Skip XML for can_ingest check since it requires config
            continue
        try:
            adapter_instance = adapter_class(**kwargs)
            if adapter_instance.can_ingest(source):
                return adapter_instance
        except Exception:
            # If adapter creation fails, try next one
            continue
    
    # If no adapter can handle it, raise error
    raise UnsupportedSourceError(
        f"No adapter found for source: {source}. Supported formats: CSV, JSON, XML",
        source=source
    )
