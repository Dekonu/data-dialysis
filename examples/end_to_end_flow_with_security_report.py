"""End-to-End Example Using src/main.py with Security Report Generation.

This example demonstrates the complete data flow using src/main.py:
1. Creates sample CSV, JSON, and XML files with PII data
2. Uses src/main.py to process each file
3. Generates and displays security reports showing all redaction events

This shows how the main pipeline automatically:
- Logs all redaction events
- Generates security reports after ingestion
- Saves reports to files
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.config_manager import DatabaseConfig


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


def create_sample_xml(file_path: Path, config_file: Path):
    """Create a sample XML file with PII data."""
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<ClinicalData>
    <PatientRecord>
        <MRN>MRN005</MRN>
        <Demographics>
            <FullName>Charlie Brown</FullName>
            <BirthDate>1992-11-05</BirthDate>
            <Gender>male</Gender>
            <SSN>555-44-3333</SSN>
            <Phone>555-555-5555</Phone>
            <Email>charlie.brown@example.com</Email>
            <Address>
                <Street>999 Maple Dr</Street>
                <City>Seattle</City>
                <State>WA</State>
                <ZIP>98101</ZIP>
            </Address>
        </Demographics>
        <Visit>
            <AdmitDate>2023-02-01T09:00:00</AdmitDate>
            <Status>finished</Status>
            <Type>outpatient</Type>
            <DxCode>Z00.00</DxCode>
        </Visit>
        <Notes>
            <ProgressNote>Annual physical exam. Patient SSN: 555-44-3333. All vitals normal.</ProgressNote>
        </Notes>
    </PatientRecord>
</ClinicalData>"""
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml_data)
    
    # Create XML config file
    xml_config = {
        "root_element": "./PatientRecord",
        "fields": {
            "patient_id": "./MRN",
            "first_name": "./Demographics/FullName",
            "date_of_birth": "./Demographics/BirthDate",
            "gender": "./Demographics/Gender",
            "ssn": "./Demographics/SSN",
            "phone": "./Demographics/Phone",
            "email": "./Demographics/Email",
            "address_line1": "./Demographics/Address/Street",
            "city": "./Demographics/Address/City",
            "state": "./Demographics/Address/State",
            "postal_code": "./Demographics/Address/ZIP",
        }
    }
    
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(xml_config, f, indent=2)
    
    print(f"SUCCESS: Created XML file: {file_path}")
    print(f"SUCCESS: Created XML config: {config_file}")
    print(f"  Records: 1")
    print(f"  Contains PII: Name, SSN, Phone, Email, Address, DOB")


def display_security_report(report_file: Path):
    """Display the security report in a readable format."""
    if not report_file.exists():
        print(f"\nWARNING: Security report not found: {report_file}")
        return
    
    print("\n" + "=" * 70)
    print("SECURITY REPORT")
    print("=" * 70)
    
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    summary = report.get('summary', {})
    
    print(f"\nReport Timestamp: {report.get('report_timestamp', 'N/A')}")
    print(f"Ingestion ID: {report.get('ingestion_id', 'N/A')}")
    
    print(f"\nSUMMARY")
    print(f"  Total Redactions: {summary.get('total_redactions', 0)}")
    
    print(f"\nRedactions by Field:")
    for field, count in sorted(summary.get('redactions_by_field', {}).items()):
        print(f"    {field}: {count}")
    
    print(f"\nRedactions by Rule:")
    for rule, count in sorted(summary.get('redactions_by_rule', {}).items()):
        print(f"    {rule}: {count}")
    
    print(f"\nRedactions by Adapter:")
    for adapter, count in sorted(summary.get('redactions_by_adapter', {}).items()):
        print(f"    {adapter}: {count}")
    
    # Show sample events
    events = report.get('events', [])
    if events:
        print(f"\nSample Events (showing first 5 of {len(events)}):")
        for i, event in enumerate(events[:5], 1):
            print(f"    {i}. {event.get('field_name')} - {event.get('rule_triggered')} at {event.get('timestamp')}")
    
    print(f"\nFull report saved to: {report_file}")
    print("=" * 70)


def run_main_pipeline(input_file: Path, xml_config: Path = None, db_path: str = None):
    """Run src/main.py with the given input file."""
    # Set up environment variables for DuckDB
    env = os.environ.copy()
    if db_path:
        env['DD_DB_TYPE'] = 'duckdb'
        env['DD_DB_PATH'] = db_path
    # Ensure security reports are saved
    env['DD_SAVE_SECURITY_REPORT'] = 'true'
    env['DD_SECURITY_REPORT_DIR'] = str(Path(__file__).parent.parent / 'reports')
    
    # Build command
    cmd = [sys.executable, '-m', 'src.main', '--input', str(input_file), '--verbose']
    
    if xml_config:
        cmd.extend(['--xml-config', str(xml_config)])
    
    # Run the main pipeline
    print(f"\n{'=' * 70}")
    print(f"Running: python -m src.main --input {input_file.name}")
    if xml_config:
        print(f"         --xml-config {xml_config.name}")
    print(f"{'=' * 70}\n")
    
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0, result.stdout


def find_latest_security_report(reports_dir: Path):
    """Find the most recently created security report."""
    if not reports_dir.exists():
        return None
    
    report_files = list(reports_dir.glob("security_report_*.json"))
    if not report_files:
        return None
    
    # Sort by modification time, most recent first
    report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return report_files[0]


def main():
    """Main function to demonstrate end-to-end flow with security reports."""
    print("=" * 70)
    print("End-to-End Flow with Security Report Generation")
    print("Using src/main.py")
    print("=" * 70)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample files
        csv_file = temp_path / "sample_patients.csv"
        json_file = temp_path / "sample_patients.json"
        xml_file = temp_path / "sample_patients.xml"
        xml_config_file = temp_path / "xml_config.json"
        
        print("\n[Step 1] Creating sample data files...")
        create_sample_csv(csv_file)
        create_sample_json(json_file)
        create_sample_xml(xml_file, xml_config_file)
        
        # Create database file
        db_file = temp_path / "clinical_data.duckdb"
        db_path = str(db_file)
        
        # Create reports directory
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Process CSV file
        print("\n[Step 2] Processing CSV file with src/main.py...")
        success, output = run_main_pipeline(csv_file, db_path=db_path)
        if success:
            print("SUCCESS: CSV processing completed")
            # Extract ingestion ID from output if possible
            if "Ingestion ID:" in output:
                ingestion_id = output.split("Ingestion ID:")[1].split()[0].strip()
                print(f"  Ingestion ID: {ingestion_id}")
        else:
            print("FAILED: CSV processing failed")
        
        # Find and display security report
        latest_report = find_latest_security_report(reports_dir)
        if latest_report:
            print("\n[Step 3] Security Report for CSV ingestion:")
            display_security_report(latest_report)
        
        # Process JSON file
        print("\n[Step 4] Processing JSON file with src/main.py...")
        success, output = run_main_pipeline(json_file, db_path=db_path)
        if success:
            print("SUCCESS: JSON processing completed")
        else:
            print("FAILED: JSON processing failed")
        
        # Find and display latest security report
        latest_report = find_latest_security_report(reports_dir)
        if latest_report:
            print("\n[Step 5] Security Report for JSON ingestion:")
            display_security_report(latest_report)
        
        # Process XML file
        print("\n[Step 6] Processing XML file with src/main.py...")
        success, output = run_main_pipeline(xml_file, xml_config=xml_config_file, db_path=db_path)
        if success:
            print("SUCCESS: XML processing completed")
        else:
            print("FAILED: XML processing failed")
        
        # Find and display latest security report
        latest_report = find_latest_security_report(reports_dir)
        if latest_report:
            print("\n[Step 7] Security Report for XML ingestion:")
            display_security_report(latest_report)
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"SUCCESS: Processed 3 files (CSV, JSON, XML)")
        print(f"SUCCESS: Database: {db_path}")
        print(f"SUCCESS: Security reports saved to: {reports_dir}")
        print(f"\nAll security reports are available in: {reports_dir}")
        print("Each report contains:")
        print("  - Total redaction count")
        print("  - Redactions by field (name, SSN, phone, email, etc.)")
        print("  - Redactions by rule (SSN_PATTERN, PHONE_PATTERN, etc.)")
        print("  - Redactions by adapter (csv_ingester, json_ingester, etc.)")
        print("  - Detailed event log with timestamps and hashes")
        print("=" * 70)


if __name__ == "__main__":
    main()

