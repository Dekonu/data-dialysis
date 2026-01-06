"""SpaCy NER Adapter for Named Entity Recognition.

This adapter implements the NERPort interface using SpaCy models.
It provides person name extraction from unstructured text.

Security Impact:
    - Processes text in-memory (no external API calls)
    - Model is loaded once and reused (efficient)
    - Handles errors gracefully with fallback to regex
    - Extracts person names for PII redaction in clinical notes

Architecture:
    - Infrastructure layer adapter (implements domain port)
    - Lazy model loading (only when needed)
    - Graceful degradation if model unavailable
"""

import logging
import sys
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except (ImportError, Exception) as e:
    # Catch all exceptions including Pydantic compatibility errors (Python 3.14+)
    SPACY_AVAILABLE = False
    Language = None  # type: ignore
    
    # Check if this is a Python 3.14 compatibility issue
    error_str = str(e).lower()
    if ("regex" in error_str or "pydantic" in error_str or 
        "unable to infer type" in error_str):
        python_version = sys.version_info
        if python_version >= (3, 14):
            logger.warning(
                f"⚠️ SpaCy is incompatible with Python {python_version.major}.{python_version.minor}. "
                f"SpaCy requires Python 3.11-3.13. "
                f"See docs/SPACY_PYTHON314_WARNING.md for solutions. "
                f"Falling back to regex-based redaction (this works fine!)."
            )

from src.domain.ports import NERPort

logger = logging.getLogger(__name__)


class SpaCyNERAdapter(NERPort):
    """SpaCy implementation of NER Port.
    
    This adapter loads a SpaCy model and uses it to extract person names
    from unstructured text. It handles model loading errors gracefully.
    
    Security Impact:
        - Identifies person names in clinical notes for redaction
        - Model processes text locally (no data sent externally)
        - Fails gracefully if model unavailable (falls back to regex)
    
    Example:
        ```python
        adapter = SpaCyNERAdapter(model_name="en_core_web_sm")
        if adapter.is_available():
            names = adapter.extract_person_names(
                "Patient John Smith visited Dr. Jane Doe."
            )
            # Returns: [("John Smith", 8, 18), ("Jane Doe", 30, 38)]
        ```
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize SpaCy NER adapter.
        
        Parameters:
            model_name: Name of SpaCy model to load
                        Options:
                        - "en_core_web_sm" (small, ~12 MB, fast)
                        - "en_core_web_md" (medium, ~40 MB, better accuracy)
                        - "en_core_web_lg" (large, ~560 MB, best accuracy)
        
        Security Impact:
            - Model is loaded lazily on first use
            - Errors are logged but don't crash the application
            - Falls back to regex if model unavailable
        """
        if not SPACY_AVAILABLE:
            logger.warning(
                "SpaCy is not installed. Install with: pip install spacy>=3.7.0. "
                "Falling back to regex-based redaction."
            )
            self.model_name = model_name
            self._nlp: Optional[Language] = None
            self._available = False
            return
        
        self.model_name = model_name
        self._nlp: Optional[Language] = None
        self._available = False
        
        try:
            # Try loading with minimal pipeline components to avoid Pydantic issues
            # This reduces dependencies and can help with compatibility
            # We only need NER for person name extraction, so disable:
            # - parser: not needed for NER
            # - tagger: not needed for NER (POS tagging)
            # - lemmatizer: not needed for NER (requires POS tags, causes warnings)
            try:
                self._nlp = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
            except (TypeError, ValueError):
                # If disable doesn't work, try standard load
                self._nlp = spacy.load(model_name)
            
            self._available = True
            logger.info(f"SpaCy model '{model_name}' loaded successfully")
        except OSError as e:
            logger.warning(
                f"SpaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}. "
                f"Falling back to regex-based redaction. Error: {e}"
            )
            self._available = False
        except (TypeError, AttributeError, ValueError) as e:
            # Handle Pydantic compatibility issues and type inference errors
            error_msg = str(e)
            if "REGEX" in error_msg or "unable to infer type" in error_msg.lower():
                logger.warning(
                    f"SpaCy model '{model_name}' has compatibility issues (likely Pydantic/Python version). "
                    f"Error: {error_msg}. "
                    f"Falling back to regex-based redaction. "
                    f"Consider: pip install --upgrade spacy pydantic, or use Python 3.11-3.13"
                )
            else:
                logger.warning(
                    f"Error loading SpaCy model '{model_name}': {error_msg}. "
                    f"Falling back to regex-based redaction."
                )
            self._available = False
        except Exception as e:
            logger.error(
                f"Unexpected error loading SpaCy model '{model_name}': {e}. "
                f"Falling back to regex-based redaction."
            )
            self._available = False
    
    def is_available(self) -> bool:
        """Check if NER service is available.
        
        Returns:
            True if NER can be used, False otherwise
        """
        return self._available and SPACY_AVAILABLE
    
    def extract_person_names(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract person names from text using SpaCy NER.
        
        Parameters:
            text: Input text to analyze (e.g., clinical notes)
        
        Returns:
            List of (name, start_pos, end_pos) tuples where:
                - name: The extracted person name string
                - start_pos: Character position where name starts
                - end_pos: Character position where name ends
            Returns empty list if NER is unavailable, text is empty, or on error.
        
        Security Impact:
            - Identifies person names that may be PII
            - Position information enables precise redaction
            - Handles errors gracefully (returns empty list on failure)
        
        Example:
            ```python
            names = adapter.extract_person_names("Patient John Smith visited.")
            # Returns: [("John Smith", 8, 18)]
            ```
        """
        if not self.is_available() or not self._nlp:
            return []
        
        if not text or not text.strip():
            return []
        
        try:
            doc = self._nlp(text)
            person_names: List[Tuple[str, int, int]] = []
            
            for ent in doc.ents:
                # SpaCy labels: PERSON, ORG, GPE (locations), etc.
                # We only extract PERSON entities for PII redaction
                if ent.label_ == "PERSON":
                    person_names.append((ent.text, ent.start_char, ent.end_char))
            
            return person_names
        except Exception as e:
            logger.error(f"Error extracting names with SpaCy: {e}")
            return []

