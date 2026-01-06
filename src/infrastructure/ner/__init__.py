"""NER (Named Entity Recognition) Infrastructure Adapters.

This package contains adapters for Named Entity Recognition services,
following Hexagonal Architecture principles. The domain defines the
NERPort interface, and these adapters provide implementations.

Security Impact:
    - Enables accurate PII detection in unstructured clinical notes
    - Processes text in-memory (no external API calls)
    - Handles errors gracefully with fallback to regex
"""

from src.infrastructure.ner.spacy_adapter import SpaCyNERAdapter

__all__ = ["SpaCyNERAdapter"]

