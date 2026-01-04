"""Secure Streaming XML Parser.

This module provides a secure streaming XML parser that combines the performance
benefits of lxml with security protections similar to defusedxml.

Security Impact:
    - Prevents XML-based attacks (Billion Laughs, quadratic blowup)
    - Enforces limits on events, depth, and file size
    - Disables entity expansion and network access
    - Memory-efficient streaming for large files

Architecture:
    - Wraps lxml.etree.iterparse with security limits
    - Provides defusedxml-style protections
    - Supports XPath-based record extraction
    - Memory usage: O(record_size) instead of O(file_size)
"""

import logging
import gc
from pathlib import Path
from typing import Iterator, Optional, Any
from contextlib import contextmanager

try:
    from lxml import etree
    from lxml.etree import iterparse, XMLParser
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    etree = None
    iterparse = None
    XMLParser = None

from src.domain.ports import TransformationError

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security violation in XML parsing."""
    pass


class StreamingXMLParser:
    """Secure streaming XML parser with lxml and security protections.
    
    This parser provides:
    - Streaming: Processes records one at a time (memory efficient)
    - Security: Enforces limits to prevent XML-based attacks
    - Performance: Uses lxml for better performance than ElementTree
    - XPath Support: Full XPath support for record extraction
    
    Security Impact:
        - Prevents memory exhaustion via event/depth limits
        - Prevents entity expansion attacks
        - Prevents network access during parsing
        - Fails fast on malformed XML
    
    Example Usage:
        ```python
        parser = StreamingXMLParser(
            max_events=1000000,
            max_depth=100
        )
        
        with parser.parse("large_file.xml", record_tag="PatientRecord") as records:
            for record in records:
                # Process record (memory released after)
                process(record)
        ```
    """
    
    def __init__(
        self,
        max_events: int = 1000000,
        max_depth: int = 100,
        max_file_size: Optional[int] = None,
        huge_tree: bool = False
    ):
        """Initialize streaming XML parser with security limits.
        
        Parameters:
            max_events: Maximum number of XML events to process (prevents DoS)
            max_depth: Maximum XML nesting depth (prevents deep recursion)
            max_file_size: Maximum file size in bytes (None = no limit)
            huge_tree: Allow huge XML trees (default: False for security)
        
        Security Impact:
            - Limits prevent memory exhaustion attacks
            - huge_tree=False prevents quadratic blowup attacks
        """
        if not LXML_AVAILABLE:
            raise ImportError(
                "lxml is required for streaming XML parsing. "
                "Install with: pip install lxml"
            )
        
        self.max_events = max_events
        self.max_depth = max_depth
        self.max_file_size = max_file_size
        self.huge_tree = huge_tree
        
        self.event_count = 0
        self._current_depth = 0
        self._depth_stack = []
    
    def _create_parser(self) -> XMLParser:
        """Create secure XML parser with security settings.
        
        Returns:
            XMLParser: Configured parser with security limits
        
        Security Impact:
            - resolve_entities=False: Prevents entity expansion attacks
            - no_network=True: Prevents network access during parsing
            - huge_tree=False: Prevents quadratic blowup attacks
            - recover=False: Fail fast on malformed XML
        """
        return XMLParser(
            huge_tree=self.huge_tree,
            strip_cdata=False,
            resolve_entities=False,  # Security: prevent entity expansion
            no_network=True,  # Security: prevent network access
            recover=False,  # Security: fail on malformed XML
            remove_blank_text=True  # Performance: remove whitespace
        )
    
    def _calculate_depth(self, elem: Any) -> int:
        """Calculate depth of XML element.
        
        Parameters:
            elem: XML element
        
        Returns:
            int: Depth of element (0 = root)
        """
        depth = 0
        parent = elem.getparent()
        while parent is not None:
            depth += 1
            parent = parent.getparent()
        return depth
    
    def _check_security_limits(self, elem: Any) -> None:
        """Check if parsing exceeds security limits.
        
        Parameters:
            elem: Current XML element
        
        Raises:
            SecurityError: If security limits are exceeded
        """
        # Check event limit
        self.event_count += 1
        
        # Log warning when approaching limit (at 80% and 90%)
        if self.event_count == int(self.max_events * 0.8):
            logger.warning(
                f"XML event count at 80% of limit: {self.event_count:,} / {self.max_events:,}"
            )
        elif self.event_count == int(self.max_events * 0.9):
            logger.warning(
                f"XML event count at 90% of limit: {self.event_count:,} / {self.max_events:,}"
            )
        
        if self.event_count > self.max_events:
            raise SecurityError(
                f"XML event limit exceeded: {self.event_count:,} > {self.max_events:,}. "
                f"This may indicate a very large file or a malicious XML file. "
                f"Consider increasing max_events for large files."
            )
        
        # Check depth limit
        depth = self._calculate_depth(elem)
        if depth > self.max_depth:
            raise SecurityError(
                f"XML depth limit exceeded: {depth} > {self.max_depth}. "
                "This may indicate a malicious XML file."
            )
    
    @contextmanager
    def parse(
        self,
        source: str,
        record_tag: Optional[str] = None,
        record_xpath: Optional[str] = None
    ) -> Iterator[Iterator[Any]]:
        """Parse XML file in streaming mode.
        
        Parameters:
            source: Path to XML file
            record_tag: XML tag name for records (e.g., "PatientRecord")
            record_xpath: XPath expression for records (alternative to record_tag)
        
        Yields:
            Iterator[Any]: Iterator of XML elements (one per record)
        
        Raises:
            SecurityError: If security limits are exceeded
            TransformationError: If parsing fails
        
        Security Impact:
            - File size checked before parsing
            - Security limits enforced during parsing
            - Memory released after each record
        
        Example:
            ```python
            parser = StreamingXMLParser()
            with parser.parse("file.xml", record_tag="PatientRecord") as records:
                for record in records:
                    # Process record
                    process(record)
                    # Memory automatically released
            ```
        """
        source_path = Path(source)
        
        # Check file exists
        if not source_path.exists():
            raise TransformationError(
                f"XML source not found: {source}",
                source=source
            )
        
        # Check file size limit
        if self.max_file_size:
            file_size = source_path.stat().st_size
            if file_size > self.max_file_size:
                raise SecurityError(
                    f"XML file size exceeds limit: {file_size} > {self.max_file_size} bytes"
                )
        
        # Reset counters
        self.event_count = 0
        self._current_depth = 0
        self._depth_stack = []
        
        # Determine record selector
        if record_tag and record_xpath:
            raise ValueError("Cannot specify both record_tag and record_xpath")
        if not record_tag and not record_xpath:
            raise ValueError("Must specify either record_tag or record_xpath")
        
        # Open file for streaming
        try:
            with open(source_path, 'rb') as xml_file:
                # Use iterparse for streaming with security options
                # Note: iterparse doesn't accept a parser object, so we pass options directly
                context = iterparse(
                    xml_file,
                    events=('end',),
                    tag=record_tag if record_tag else None,
                    huge_tree=self.huge_tree,
                    resolve_entities=False,  # Security: prevent entity expansion
                    no_network=True,  # Security: prevent network access
                    recover=False,  # Security: fail on malformed XML
                    remove_blank_text=True  # Performance: remove whitespace
                )
                
                def record_iterator() -> Iterator[Any]:
                    """Inner iterator that yields records and enforces security."""
                    root_element = None
                    record_counter = 0
                    
                    for event, elem in context:
                        try:
                            # Check security limits
                            self._check_security_limits(elem)
                            
                            # If using XPath, filter elements
                            if record_xpath:
                                # Check if element matches XPath
                                matches = elem.xpath(record_xpath)
                                if not matches:
                                    # Clear and skip - remove from parent to free memory
                                    elem.clear()
                                    parent = elem.getparent()
                                    if parent is not None:
                                        parent.remove(elem)
                                    continue
                            
                            # Get root element reference on first iteration
                            if root_element is None:
                                try:
                                    root_element = elem.getroottree().getroot()
                                except Exception:
                                    root_element = None
                            
                            # Yield record element
                            yield elem
                            
                            # After yielding, clear element to free memory
                            # Critical for large files: clear element and remove from parent tree
                            elem.clear(keep_tail=False)
                            parent = elem.getparent()
                            if parent is not None:
                                parent.remove(elem)
                            
                            # Root cleanup: Keep only last few elements to prevent memory growth
                            if root_element is not None:
                                try:
                                    # Keep only last 2 elements (aggressive cleanup)
                                    while len(root_element) > 2:
                                        root_element.remove(root_element[0])
                                except Exception:
                                    pass  # Ignore cleanup errors
                            
                            # Periodic garbage collection for very large files
                            record_counter += 1
                            if record_counter % 500 == 0:
                                gc.collect()
                            
                        except SecurityError:
                            # Re-raise security errors
                            raise
                        except Exception as e:
                            # Log and skip malformed records
                            logger.warning(
                                f"Error processing XML element: {str(e)}",
                                exc_info=True
                            )
                            # Clear element on error
                            try:
                                elem.clear(keep_tail=False)
                                parent = elem.getparent()
                                if parent is not None:
                                    parent.remove(elem)
                            except Exception:
                                pass
                            continue
                
                yield record_iterator()
                
        except SecurityError:
            raise
        except Exception as e:
            raise TransformationError(
                f"Failed to parse XML file: {str(e)}",
                source=source
            )
    
    def extract_with_xpath(self, elem: Any, xpath_expr: str) -> Optional[str]:
        """Extract value from XML element using XPath.
        
        Parameters:
            elem: XML element to extract from
            xpath_expr: XPath expression (relative to element)
        
        Returns:
            Optional[str]: Extracted value or None if not found
        
        Security Impact:
            - XPath expressions are evaluated safely
            - No network access or external entity resolution
        """
        try:
            # Evaluate XPath (relative to element)
            results = elem.xpath(xpath_expr)
            
            if not results:
                return None
            
            # Handle list of results
            if isinstance(results, list):
                if not results:
                    return None
                # Get first result
                result = results[0]
            else:
                result = results
            
            # Handle different result types
            if isinstance(result, str):
                return result.strip() if result else None
            elif isinstance(result, (int, float)):
                return str(result)
            elif hasattr(result, 'text'):
                # Element with text attribute
                if result.text:
                    return result.text.strip()
                # If no text, get all text content
                text = ''.join(result.itertext())
                return text.strip() if text else None
            elif hasattr(result, 'tag'):
                # Element, return text content
                text = ''.join(result.itertext())
                return text.strip() if text else None
            else:
                # Convert to string
                return str(result).strip() if result else None
                
        except Exception as e:
            logger.debug(f"XPath extraction failed for '{xpath_expr}': {str(e)}")
            return None
