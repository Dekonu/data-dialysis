"""Test script for JSON ingestion pipeline.

This script demonstrates the JSON ingestion adapter with both valid and
malformed records to test the triage and fail-safe error handling.

Security Impact:
    - Tests that bad records are rejected without crashing the pipeline
    - Validates that PII redaction is working correctly
    - Ensures audit trail logging for rejected records
"""

import json
from pathlib import Path

from src.adapters.json_ingester import JSONIngester


def run_pipeline(file_path: str):
    """Run the ingestion pipeline on a JSON file.
    
    Parameters:
        file_path: Path to JSON file to ingest
    
    Security Impact:
        - Processes records through Safety Layer (Pydantic validation)
        - Applies PII redaction via Sieve (RedactorService)
        - Logs security rejections for bad records
    """
    print(f"--- Starting Ingestion for {file_path} ---")
    
    loader = JSONIngester()
    valid_count = 0
    
    # The pipeline flows: File -> JSON Adapter -> Pydantic Model -> Sieve -> Out
    for record in loader.ingest(file_path):
        valid_count += 1
        # Note: PII fields are redacted, so we can safely print patient_id
        print(f"[OK] Ingested: Patient ID: {record.patient.patient_id} | "
              f"Encounters: {len(record.encounters)} | "
              f"Observations: {len(record.observations)}")
    
    print(f"--- Pipeline Finished. Total Valid Records: {valid_count} ---")


if __name__ == "__main__":
    # Create a dummy file for testing
    dummy_data = [
        {
            # Valid record - should pass validation and PII redaction
            "patient": {
                "patient_id": "MRN001",
                "first_name": "John",
                "last_name": "Doe",
                "date_of_birth": "1990-01-01",
                "gender": "male",
                "ssn": "123-45-6789",
                "phone": "555-123-4567",
                "email": "john.doe@example.com",
                "address_line1": "123 Main St",
                "city": "Springfield",
                "state": "IL",
                "postal_code": "62701",
            },
            "encounters": [
                {
                    "encounter_id": "ENC001",
                    "patient_id": "MRN001",
                    "class_code": "outpatient",
                    "period_start": "2023-01-01T10:00:00",
                    "period_end": "2023-01-01T11:00:00",
                    "diagnosis_codes": ["I10", "E11.9"],
                }
            ],
            "observations": [
                {
                    "observation_id": "OBS001",
                    "patient_id": "MRN001",
                    "category": "vital-signs",
                    "value": "120/80",
                    "unit": "mmHg",
                    "effective_date": "2023-01-01T10:30:00",
                    "notes": "Blood pressure normal. Patient SSN: 123-45-6789",
                }
            ],
        },
        {
            # Malformed record - should fail validation safely
            # This record has an invalid MRN (too short) and should be rejected
            "patient": {
                "patient_id": "AB",  # <--- THIS SHOULD FAIL SAFELY (MRN too short)
                "first_name": "Jane",
                "last_name": "Smith",
                "date_of_birth": "1995-05-15",
                "gender": "female",
                "ssn": "987-65-4321",
                "phone": "555-987-6543",
            },
            "encounters": [],
            "observations": [],
        },
        {
            # Another valid record - should pass
            "patient": {
                "patient_id": "MRN003",
                "first_name": "Bob",
                "last_name": "Johnson",
                "date_of_birth": "1985-03-20",
                "gender": "male",
                "state": "CA",
                "postal_code": "90210",
            },
            "encounters": [],
            "observations": [
                {
                    "observation_id": "OBS002",
                    "patient_id": "MRN003",
                    "category": "laboratory",
                    "code": "85354-9",
                    "value": "98.6",
                    "unit": "F",
                    "effective_date": "2023-01-02T14:00:00",
                }
            ],
        },
        {
            # Record with future date - should fail validation
            "patient": {
                "patient_id": "MRN004",
                "first_name": "Alice",
                "last_name": "Williams",
                "date_of_birth": "2030-01-01",  # <--- THIS SHOULD FAIL SAFELY (future date)
                "gender": "female",
            },
            "encounters": [],
            "observations": [],
        },
    ]
    
    # Write test file
    test_file = Path("test_batch.json")
    with open(test_file, "w") as f:
        json.dump(dummy_data, f, indent=2)
    
    print(f"Created test file: {test_file}")
    print(f"Test data contains {len(dummy_data)} records:")
    print("  - Record 1: Valid (should pass)")
    print("  - Record 2: Invalid MRN (should be rejected)")
    print("  - Record 3: Valid (should pass)")
    print("  - Record 4: Future DOB (should be rejected)")
    print()
    
    # Run the pipeline
    run_pipeline(str(test_file))
    
    print()
    print("Note: Check logs for 'SECURITY REJECTION' messages for rejected records.")

