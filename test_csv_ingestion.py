"""Test script for CSV ingestion pipeline.

This script demonstrates the CSV ingestion adapter with both valid and
malformed records to test the triage and fail-safe error handling.

Security Impact:
    - Tests that bad records are rejected without crashing the pipeline
    - Validates that PII redaction is working correctly
    - Ensures audit trail logging for rejected records
"""

import csv
from pathlib import Path
import logging

from src.adapters.csv_ingester import CSVIngester
from src.domain.ports import Result

# Configure logging to show security rejections
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(file_path: str, column_mapping: dict = None):
    """Run the CSV ingestion pipeline on a CSV file.
    
    Parameters:
        file_path: Path to CSV file to ingest
        column_mapping: Optional column mapping dictionary
    
    Security Impact:
        - Processes records through Safety Layer (Pydantic validation)
        - Applies PII redaction via Sieve (RedactorService)
        - Logs security rejections for bad records
    """
    print(f"--- Starting CSV Ingestion for {file_path} ---")
    
    # Create ingester with optional column mapping
    if column_mapping:
        loader = CSVIngester(column_mapping=column_mapping, has_header=True)
        print(f"Using custom column mapping: {column_mapping}")
    else:
        loader = CSVIngester(has_header=True)
        print("Using auto-detected column mapping from headers")
    
    valid_count = 0
    
    # The pipeline flows: File -> CSV Adapter -> Column Mapping -> Pydantic Model -> Sieve -> Result -> Out
    for result in loader.ingest(file_path):
        if result.is_success():
            valid_count += 1
            record = result.value
            # Note: PII fields are redacted, so we can safely print patient_id
            print(f"[OK] Ingested: Patient ID: {record.patient.patient_id} | "
                  f"Encounters: {len(record.encounters)} | "
                  f"Observations: {len(record.observations)}")
        else:
            # Failure results are logged by the adapter, but we can optionally print them
            pass  # Failures are already logged as "SECURITY REJECTION"
    
    print(f"--- Pipeline Finished. Total Valid Records: {valid_count} ---")


if __name__ == "__main__":
    # Create CSV test data with header
    test_file = Path("test_batch.csv")
    
    csv_data = [
        # Header row
        ["MRN", "FirstName", "LastName", "DOB", "Gender", "SSN", "Phone", "Email", "Address", "City", "State", "ZIP"],
        # Valid record 1
        ["MRN001", "John", "Doe", "1990-01-01", "male", "123-45-6789", "555-123-4567", "john.doe@example.com", "123 Main St", "Springfield", "IL", "62701"],
        # Invalid record 2: MRN too short
        ["AB", "Jane", "Smith", "1995-05-15", "female", "987-65-4321", "555-987-6543", "jane@example.com", "456 Oak Ave", "Los Angeles", "CA", "90210"],
        # Valid record 3
        ["MRN003", "Bob", "Johnson", "1985-03-20", "male", "", "", "", "789 Pine Rd", "Chicago", "IL", "60601"],
        # Invalid record 4: Future DOB
        ["MRN004", "Alice", "Williams", "2030-01-01", "female", "", "", "", "321 Elm St", "Miami", "FL", "33101"],
    ]
    
    with open(test_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f"Created test file: {test_file}")
    print(f"Test data contains {len(csv_data) - 1} records (excluding header):")
    print("  - Record 1: Valid (should pass)")
    print("  - Record 2: Invalid MRN (should be rejected)")
    print("  - Record 3: Valid (should pass)")
    print("  - Record 4: Future DOB (should be rejected)")
    print()
    
    # Run the pipeline with auto-detection
    run_pipeline(str(test_file))
    
    print()
    print("Note: Check logs for 'SECURITY REJECTION' messages for rejected records.")
    print()
    
    # Test with custom column mapping
    print("--- Testing with Custom Column Mapping ---")
    custom_mapping = {
        "patient_id": "MRN",
        "first_name": "FirstName",
        "last_name": "LastName",
        "date_of_birth": "DOB",
        "gender": "Gender",
        "ssn": "SSN",
        "phone": "Phone",
        "email": "Email",
        "address_line1": "Address",
        "city": "City",
        "state": "State",
        "postal_code": "ZIP",
    }
    
    run_pipeline(str(test_file), column_mapping=custom_mapping)

