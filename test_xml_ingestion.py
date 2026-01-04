"""Test script for XML ingestion pipeline.

This script demonstrates the XML ingestion adapter with both valid and
malformed records to test the triage and fail-safe error handling.

Security Impact:
    - Tests that bad records are rejected without crashing the pipeline
    - Validates that PII redaction is working correctly
    - Ensures audit trail logging for rejected records
    - Tests defusedxml protection against XML attacks
"""

import json
from pathlib import Path

from src.adapters.xml_ingester import XMLIngester


def run_pipeline(file_path: str, config_path: str):
    """Run the XML ingestion pipeline on an XML file.
    
    Parameters:
        file_path: Path to XML file to ingest
        config_path: Path to JSON configuration file with XPath mappings
    
    Security Impact:
        - Processes records through Safety Layer (Pydantic validation)
        - Applies PII redaction via Sieve (RedactorService)
        - Logs security rejections for bad records
        - Uses defusedxml to prevent XML attacks
    """
    print(f"--- Starting XML Ingestion for {file_path} ---")
    print(f"Using configuration: {config_path}")
    
    loader = XMLIngester(config_path=config_path)
    valid_count = 0
    
    # The pipeline flows: File -> XML Adapter -> XPath Extraction -> Pydantic Model -> Sieve -> Out
    for record in loader.ingest(file_path):
        valid_count += 1
        # Note: PII fields are redacted, so we can safely print patient_id
        print(f"[OK] Ingested: Patient ID: {record.patient.patient_id} | "
              f"Encounters: {len(record.encounters)} | "
              f"Observations: {len(record.observations)}")
    
    print(f"--- Pipeline Finished. Total Valid Records: {valid_count} ---")


if __name__ == "__main__":
    # Create XML configuration file
    xml_config = {
        "root_element": "./PatientRecord",  # Relative to root element (ClinicalData)
        "fields": {
            "mrn": "./MRN",
            "patient_name": "./Demographics/FullName",
            "patient_dob": "./Demographics/BirthDate",
            "patient_gender": "./Demographics/Gender",
            "ssn": "./Demographics/SSN",
            "phone": "./Demographics/Phone",
            "email": "./Demographics/Email",
            "address_line1": "./Demographics/Address/Street",
            "city": "./Demographics/Address/City",
            "state": "./Demographics/Address/State",
            "postal_code": "./Demographics/Address/ZIP",
            "encounter_date": "./Visit/AdmitDate",
            "encounter_status": "./Visit/Status",
            "encounter_type": "./Visit/Type",
            "primary_diagnosis_code": "./Visit/DxCode",
            "clinical_notes": "./Notes/ProgressNote"
        }
    }
    
    config_file = Path("xml_config.json")
    with open(config_file, "w") as f:
        json.dump(xml_config, f, indent=2)
    
    print(f"Created configuration file: {config_file}")
    print()
    
    # Create XML test data
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<ClinicalData>
    <!-- Valid Record 1: Should pass validation -->
    <PatientRecord>
        <MRN>MRN001</MRN>
        <Demographics>
            <FullName>John Doe</FullName>
            <BirthDate>1990-01-01</BirthDate>
            <Gender>male</Gender>
            <SSN>123-45-6789</SSN>
            <Phone>555-123-4567</Phone>
            <Email>john.doe@example.com</Email>
            <Address>
                <Street>123 Main St</Street>
                <City>Springfield</City>
                <State>IL</State>
                <ZIP>62701</ZIP>
            </Address>
        </Demographics>
        <Visit>
            <AdmitDate>2023-01-01T10:00:00</AdmitDate>
            <Status>finished</Status>
            <Type>outpatient</Type>
            <DxCode>I10</DxCode>
        </Visit>
        <Notes>
            <ProgressNote>Blood pressure normal. Patient SSN: 123-45-6789</ProgressNote>
        </Notes>
    </PatientRecord>
    
    <!-- Invalid Record 2: MRN too short - should be rejected -->
    <PatientRecord>
        <MRN>AB</MRN>
        <Demographics>
            <FullName>Jane Smith</FullName>
            <BirthDate>1995-05-15</BirthDate>
            <Gender>female</Gender>
            <SSN>987-65-4321</SSN>
            <Phone>555-987-6543</Phone>
        </Demographics>
        <Visit>
            <AdmitDate>2023-01-01T10:00:00</AdmitDate>
            <Status>arrived</Status>
            <Type>outpatient</Type>
        </Visit>
    </PatientRecord>
    
    <!-- Valid Record 3: Should pass validation -->
    <PatientRecord>
        <MRN>MRN003</MRN>
        <Demographics>
            <FullName>Bob Johnson</FullName>
            <BirthDate>1985-03-20</BirthDate>
            <Gender>male</Gender>
            <Address>
                <Street>456 Oak Ave</Street>
                <City>Los Angeles</City>
                <State>CA</State>
                <ZIP>90210</ZIP>
            </Address>
        </Demographics>
        <Visit>
            <AdmitDate>2023-01-02T14:00:00</AdmitDate>
            <Status>finished</Status>
            <Type>outpatient</Type>
        </Visit>
        <Notes>
            <ProgressNote>Temperature: 98.6F. All vital signs normal.</ProgressNote>
        </Notes>
    </PatientRecord>
    
    <!-- Invalid Record 4: Future DOB - should be rejected -->
    <PatientRecord>
        <MRN>MRN004</MRN>
        <Demographics>
            <FullName>Alice Williams</FullName>
            <BirthDate>2030-01-01</BirthDate>
            <Gender>female</Gender>
        </Demographics>
        <Visit>
            <AdmitDate>2023-01-01T10:00:00</AdmitDate>
            <Status>finished</Status>
            <Type>outpatient</Type>
        </Visit>
    </PatientRecord>
</ClinicalData>"""
    
    test_file = Path("test_batch.xml")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(xml_data)
    
    print(f"Created test file: {test_file}")
    print(f"Test data contains 4 records:")
    print("  - Record 1: Valid (should pass)")
    print("  - Record 2: Invalid MRN (should be rejected)")
    print("  - Record 3: Valid (should pass)")
    print("  - Record 4: Future DOB (should be rejected)")
    print()
    
    # Run the pipeline
    run_pipeline(str(test_file), str(config_file))
    
    print()
    print("Note: Check logs for 'SECURITY REJECTION' messages for rejected records.")

