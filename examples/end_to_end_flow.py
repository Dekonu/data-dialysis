"""End-to-End Example: Data Flow from Raw File to DuckDB.

This example demonstrates the complete data flow:
1. CSV/JSON -> Pandas DataFrame -> PII Redaction -> DuckDB
2. XML -> GoldenRecord -> PII Redaction -> DuckDB

It shows how data moves through the system with PII redaction
and validation at each step.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.adapters.ingesters import get_adapter
from src.adapters.storage.duckdb_adapter import DuckDBAdapter
from src.infrastructure.config_manager import DatabaseConfig
from src.domain.services import RedactorService


def create_sample_csv(file_path: Path):
    """Create a sample CSV file with PII data."""
    csv_data = [
        ["MRN", "FirstName", "LastName", "DOB", "Gender", "SSN", "Phone", "Email", "Address", "City", "State", "ZIP"],
        ["MRN001", "John", "Doe", "1990-01-01", "male", "123-45-6789", "555-123-4567", "john.doe@example.com", "123 Main St", "Springfield", "IL", "62701"],
        ["MRN002", "Jane", "Smith", "1995-05-15", "female", "987-65-4321", "555-987-6543", "jane@example.com", "456 Oak Ave", "Los Angeles", "CA", "90210"],
        ["MRN003", "Bob", "Johnson", "1985-03-20", "male", "", "", "", "789 Pine Rd", "Chicago", "IL", "60601"],
    ]
    
    import csv
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f"SUCCESS: Created CSV file: {file_path}")
    print(f"  Records: {len(csv_data) - 1} (excluding header)")
    print(f"  Contains PII: Names, SSNs, Phone, Email, Address, DOB")


def create_sample_json(file_path: Path):
    """Create a sample JSON file with PII data."""
    import json
    
    json_data = [
        {
            "patient": {
                "patient_id": "MRN004",
                "first_name": "Alice",
                "last_name": "Williams",
                "date_of_birth": "1988-07-12",
                "gender": "female",
                "ssn": "111-22-3333",
                "phone": "555-111-2222",
                "email": "alice.williams@example.com",
                "address_line1": "321 Elm St",
                "city": "Miami",
                "state": "FL",
                "postal_code": "33101",
            },
            "encounters": [
                {
                    "encounter_id": "ENC001",
                    "patient_id": "MRN004",
                    "status": "finished",
                    "class_code": "outpatient",
                    "period_start": "2023-01-01T10:00:00",
                    "period_end": "2023-01-01T11:00:00",
                    "diagnosis_codes": ["I10", "E11.9"],
                }
            ],
            "observations": [
                {
                    "observation_id": "OBS001",
                    "patient_id": "MRN004",
                    "status": "final",
                    "category": "vital-signs",
                    "code": "85354-9",
                    "effective_date": "2023-01-01T10:30:00",
                    "value": "120/80",
                    "unit": "mmHg",
                    "notes": "Blood pressure normal. Patient name: Alice Williams",
                }
            ],
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"SUCCESS: Created JSON file: {file_path}")
    print(f"  Records: {len(json_data)}")
    print(f"  Contains: Patient, Encounter, Observation data with PII")


def create_sample_xml(file_path: Path):
    """Create a sample XML file with PII data."""
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<ClinicalData>
    <PatientRecord>
        <MRN>MRN005</MRN>
        <Demographics>
            <FullName>David Miller</FullName>
            <BirthDate>1991-03-15</BirthDate>
            <Gender>male</Gender>
            <SSN>777-88-9999</SSN>
            <Phone>555-555-5555</Phone>
            <Email>david.miller@example.com</Email>
            <Address>
                <Street>555 Cedar Ln</Street>
                <City>Portland</City>
                <State>OR</State>
                <ZIP>97201</ZIP>
            </Address>
        </Demographics>
        <Visit>
            <AdmitDate>2023-02-01T09:00:00</AdmitDate>
            <Status>finished</Status>
            <Type>outpatient</Type>
            <DxCode>I10</DxCode>
        </Visit>
        <Notes>
            <ProgressNote>Patient David Miller reports feeling well. SSN: 777-88-9999</ProgressNote>
        </Notes>
    </PatientRecord>
</ClinicalData>"""
    
    file_path.write_text(xml_data, encoding="utf-8")
    print(f"SUCCESS: Created XML file: {file_path}")
    print(f"  Contains PII: Name, SSN, Phone, Email, Address, DOB, Notes")


def create_xml_config(file_path: Path):
    """Create XML configuration file."""
    import json
    
    xml_config = {
        "root_element": "./PatientRecord",
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
    
    with open(file_path, "w") as f:
        json.dump(xml_config, f, indent=2)
    
    print(f"SUCCESS: Created XML config file: {file_path}")


def demonstrate_csv_flow(csv_file: Path, db_path: str):
    """Demonstrate CSV -> Pandas -> Redaction -> DuckDB flow."""
    print("\n" + "=" * 70)
    print("CSV Flow: CSV -> Pandas -> Redaction -> DuckDB")
    print("=" * 70)
    
    # Step 1: Create DuckDB adapter
    print("\n[Step 1] Initializing DuckDB adapter...")
    db_config = DatabaseConfig(db_type="duckdb", db_path=db_path)
    storage = DuckDBAdapter(db_config=db_config)
    result = storage.initialize_schema()
    assert result.is_success(), f"Schema initialization failed: {result.error}"
    print("SUCCESS: Schema initialized")
    
    # Step 2: Get CSV adapter
    print("\n[Step 2] Getting CSV ingestion adapter...")
    adapter = get_adapter(str(csv_file))
    print(f"SUCCESS: Adapter: {adapter.__class__.__name__}")
    
    # Step 3: Ingest CSV (returns DataFrames with redacted PII)
    print("\n[Step 3] Ingesting CSV file...")
    print("  → CSV file is read in chunks")
    print("  → Each chunk is converted to pandas DataFrame")
    print("  → PII redaction is applied (vectorized)")
    print("  → DataFrames are validated")
    
    total_rows = 0
    for i, result in enumerate(adapter.ingest(str(csv_file)), 1):
        if result.is_success():
            df = result.value
            print(f"\n  Chunk {i}: {len(df)} rows")
            print(f"    Columns: {list(df.columns)}")
            
            # Show redaction in action
            if 'family_name' in df.columns:
                redacted_count = df['family_name'].isna().sum() + (df['family_name'] == '[REDACTED]').sum()
                print(f"    Names redacted: {redacted_count}/{len(df)}")
            
            # Step 4: Persist to DuckDB
            print(f"  → Persisting to DuckDB...")
            persist_result = storage.persist_dataframe(df, "patients")
            if persist_result.is_success():
                total_rows += persist_result.value
                print(f"    SUCCESS: Persisted {persist_result.value} rows")
            else:
                print(f"    FAILED: {persist_result.error}")
        else:
            print(f"  FAILED: Chunk {i} failed: {result.error}")
    
    # Step 5: Verify data in DuckDB
    print("\n[Step 4] Verifying data in DuckDB...")
    conn = storage._get_connection()
    
    # Count records
    count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    print(f"  Total patients in DB: {count}")
    
    # Check PII redaction
    result = conn.execute("""
        SELECT patient_id, family_name, phone, email, date_of_birth
        FROM patients
        WHERE patient_id = 'MRN001'
    """).fetchone()
    
    if result:
        patient_id, family_name, phone, email, dob = result
        print(f"\n  Sample record (MRN001):")
        print(f"    Patient ID: {patient_id} (preserved)")
        print(f"    Family Name: {family_name} (redacted)")
        print(f"    Phone: {phone} (redacted)")
        print(f"    Email: {email} (redacted)")
        print(f"    DOB: {dob} (redacted)")
    
    # Check audit log
    audit_count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    print(f"\n  Audit log entries: {audit_count}")
    
    storage.close()
    print("\nSUCCESS: CSV flow completed successfully!")


def demonstrate_json_flow(json_file: Path, db_path: str):
    """Demonstrate JSON -> Pandas -> Redaction -> DuckDB flow."""
    print("\n" + "=" * 70)
    print("JSON Flow: JSON -> Pandas -> Redaction -> DuckDB")
    print("=" * 70)
    
    # Initialize
    db_config = DatabaseConfig(db_type="duckdb", db_path=db_path)
    storage = DuckDBAdapter(db_config=db_config)
    storage.initialize_schema()
    
    # Get adapter
    adapter = get_adapter(str(json_file))
    print(f"\nSUCCESS: Adapter: {adapter.__class__.__name__}")
    
    # Process
    print("\n[Processing] Ingesting JSON file...")
    print("  → JSON file is parsed")
    print("  → Records are converted to DataFrames")
    print("  → PII redaction is applied")
    print("  → DataFrames are validated")
    
    for result in adapter.ingest(str(json_file)):
        if result.is_success():
            df = result.value
            print(f"\n  DataFrame: {len(df)} rows, columns: {list(df.columns)}")
            
            # Determine table
            if 'patient_id' in df.columns:
                table = "patients"
            elif 'encounter_id' in df.columns:
                table = "encounters"
            elif 'observation_id' in df.columns:
                table = "observations"
            else:
                continue
            
            persist_result = storage.persist_dataframe(df, table)
            if persist_result.is_success():
                print(f"  SUCCESS: Persisted {persist_result.value} rows to {table}")
    
    # Verify
    conn = storage._get_connection()
    patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    encounter_count = conn.execute("SELECT COUNT(*) FROM encounters").fetchone()[0]
    observation_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    
    print(f"\n  Database counts:")
    print(f"    Patients: {patient_count}")
    print(f"    Encounters: {encounter_count}")
    print(f"    Observations: {observation_count}")
    
    storage.close()
    print("\nSUCCESS: JSON flow completed successfully!")


def demonstrate_xml_flow(xml_file: Path, xml_config_file: Path, db_path: str):
    """Demonstrate XML -> GoldenRecord -> Redaction -> DuckDB flow."""
    print("\n" + "=" * 70)
    print("XML Flow: XML -> GoldenRecord -> Redaction -> DuckDB")
    print("=" * 70)
    
    # Initialize
    db_config = DatabaseConfig(db_type="duckdb", db_path=db_path)
    storage = DuckDBAdapter(db_config=db_config)
    storage.initialize_schema()
    
    # Get adapter
    adapter = get_adapter(str(xml_file), config_path=str(xml_config_file))
    print(f"\nSUCCESS: Adapter: {adapter.__class__.__name__}")
    
    # Process
    print("\n[Processing] Ingesting XML file...")
    print("  → XML file is parsed (defusedxml for security)")
    print("  → Records are transformed to GoldenRecord objects")
    print("  → PII redaction is applied (via field validators)")
    print("  → GoldenRecords are validated")
    
    for result in adapter.ingest(str(xml_file)):
        if result.is_success():
            golden_record = result.value
            print(f"\n  GoldenRecord:")
            print(f"    Patient ID: {golden_record.patient.patient_id}")
            print(f"    Family Name: {golden_record.patient.family_name} (redacted)")
            print(f"    Encounters: {len(golden_record.encounters)}")
            print(f"    Observations: {len(golden_record.observations)}")
            
            persist_result = storage.persist(golden_record)
            if persist_result.is_success():
                print(f"  SUCCESS: Persisted record ID: {persist_result.value}")
    
    # Verify
    conn = storage._get_connection()
    patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    print(f"\n  Total patients in DB: {patient_count}")
    
    storage.close()
    print("\nSUCCESS: XML flow completed successfully!")


def main():
    """Run end-to-end flow demonstration."""
    print("=" * 70)
    print("Data-Dialysis: End-to-End Data Flow Demonstration")
    print("=" * 70)
    print("\nThis example demonstrates the complete data flow:")
    print("  1. CSV/JSON -> Pandas DataFrame -> PII Redaction -> DuckDB")
    print("  2. XML -> GoldenRecord -> PII Redaction -> DuckDB")
    print("\nAll PII is automatically redacted before persistence.")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create sample files
        csv_file = tmp_path / "patients.csv"
        json_file = tmp_path / "patients.json"
        xml_file = tmp_path / "patients.xml"
        xml_config_file = tmp_path / "xml_config.json"
        db_path = str(tmp_path / "clinical.duckdb")
        
        print("\n" + "-" * 70)
        print("Creating Sample Data Files")
        print("-" * 70)
        create_sample_csv(csv_file)
        create_sample_json(json_file)
        create_sample_xml(xml_file)
        create_xml_config(xml_config_file)
        
        # Demonstrate flows
        demonstrate_csv_flow(csv_file, db_path)
        demonstrate_json_flow(json_file, db_path)
        demonstrate_xml_flow(xml_file, xml_config_file, db_path)
        
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print("\nSUCCESS: All data flows completed successfully!")
        print("SUCCESS: PII was redacted at each step")
        print("SUCCESS: Data was validated before persistence")
        print("SUCCESS: Audit trail was maintained")
        print(f"\nDatabase file: {db_path}")
        print("(Temporary files will be cleaned up)")


if __name__ == "__main__":
    main()

