"""Generate XML test files of various sizes for benchmarking.

This script creates XML files matching the ClinicalData schema used in the
DataDialysis project for performance testing and benchmarking.

Security Impact:
    - Generates synthetic test data (not real PII)
    - Files are for testing ingestion performance only
"""

import random
import string
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple


# Sample data pools for realistic test data generation
FIRST_NAMES = [
    "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Jessica",
    "William", "Ashley", "James", "Amanda", "Christopher", "Melissa", "Daniel",
    "Michelle", "Matthew", "Kimberly", "Anthony", "Amy", "Mark", "Angela",
    "Donald", "Stephanie", "Steven", "Nicole", "Paul", "Elizabeth", "Andrew",
    "Helen", "Joshua", "Sandra", "Kenneth", "Donna", "Kevin", "Carol", "Brian",
    "Ruth", "George", "Sharon", "Edward", "Michelle", "Ronald", "Laura"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris",
    "Clark", "Lewis", "Robinson", "Walker", "Young", "King", "Wright", "Scott",
    "Torres", "Nguyen", "Gonzalez", "Hill", "Flores", "Green", "Adams", "Nelson",
    "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts"
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis",
    "Seattle", "Denver", "Washington", "Boston", "El Paso", "Detroit", "Nashville",
    "Portland", "Oklahoma City", "Las Vegas", "Memphis", "Louisville", "Baltimore"
]

STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
          "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
          "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
          "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
          "WI", "WY"]

GENDERS = ["male", "female", "other", "unknown"]

ENCOUNTER_STATUSES = ["planned", "arrived", "triaged", "in-progress", "onleave",
                      "finished", "cancelled", "entered-in-error", "unknown"]

ENCOUNTER_TYPES = ["emergency", "inpatient", "outpatient", "ambulatory", "virtual"]

DIAGNOSIS_CODES = [
    "I10", "E11.9", "I50.9", "J44.1", "N18.6", "E78.5", "I25.10", "M79.3",
    "K21.9", "G89.29", "F32.9", "E66.01", "I25.2", "N39.0", "R06.02"
]

CLINICAL_NOTES = [
    "Patient presents with normal vital signs. Blood pressure 120/80, heart rate 72 bpm.",
    "Follow-up visit. Patient reports improvement in symptoms. Continue current medication regimen.",
    "Annual physical examination. All systems reviewed. No acute concerns identified.",
    "Patient reports chest pain. EKG performed, results normal. Referred for further evaluation.",
    "Routine check-up. Patient is compliant with medications. No adverse effects reported.",
    "Post-operative follow-up. Wound healing well. Patient advised to continue wound care.",
    "Chronic condition management. Patient stable on current treatment plan.",
    "Preventive care visit. Vaccinations up to date. Health maintenance discussed.",
    "Patient presents with upper respiratory symptoms. Prescribed symptomatic treatment.",
    "Diabetes management visit. Blood glucose levels well controlled. Continue monitoring."
]


def generate_ssn() -> str:
    """Generate a synthetic SSN for testing."""
    return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"


def generate_phone() -> str:
    """Generate a synthetic phone number."""
    return f"{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


def generate_email(first_name: str, last_name: str) -> str:
    """Generate a synthetic email address."""
    domains = ["example.com", "test.org", "demo.net", "sample.gov"]
    return f"{first_name.lower()}.{last_name.lower()}@{random.choice(domains)}"


def generate_street_address() -> str:
    """Generate a synthetic street address."""
    numbers = random.randint(1, 9999)
    streets = ["Main St", "Oak Ave", "Park Blvd", "Elm St", "Maple Dr", "Cedar Ln",
               "Pine Rd", "First St", "Second Ave", "Washington St", "Lincoln Ave"]
    return f"{numbers} {random.choice(streets)}"


def generate_postal_code() -> str:
    """Generate a synthetic ZIP code."""
    return f"{random.randint(10000, 99999)}"


def generate_birth_date() -> str:
    """Generate a valid birth date (not in future)."""
    start_date = datetime(1950, 1, 1)
    end_date = datetime(2010, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    birth_date = start_date + timedelta(days=random_days)
    return birth_date.strftime("%Y-%m-%d")


def generate_admit_date() -> str:
    """Generate a valid admit date."""
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    admit_date = start_date + timedelta(days=random_days)
    return admit_date.strftime("%Y-%m-%dT%H:%M:%S")


def generate_mrn(record_num: int) -> str:
    """Generate a valid MRN (at least 3 characters)."""
    return f"MRN{record_num:08d}"


def generate_patient_record(record_num: int) -> str:
    """Generate a single PatientRecord XML element."""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    full_name = f"{first_name} {last_name}"
    mrn = generate_mrn(record_num)
    birth_date = generate_birth_date()
    gender = random.choice(GENDERS)
    ssn = generate_ssn()
    phone = generate_phone()
    email = generate_email(first_name, last_name)
    street = generate_street_address()
    city = random.choice(CITIES)
    state = random.choice(STATES)
    zip_code = generate_postal_code()
    admit_date = generate_admit_date()
    status = random.choice(ENCOUNTER_STATUSES)
    encounter_type = random.choice(ENCOUNTER_TYPES)
    dx_code = random.choice(DIAGNOSIS_CODES)
    note = random.choice(CLINICAL_NOTES)
    
    record = f"""    <PatientRecord>
        <MRN>{mrn}</MRN>
        <Demographics>
            <FullName>{full_name}</FullName>
            <BirthDate>{birth_date}</BirthDate>
            <Gender>{gender}</Gender>
            <SSN>{ssn}</SSN>
            <Phone>{phone}</Phone>
            <Email>{email}</Email>
            <Address>
                <Street>{street}</Street>
                <City>{city}</City>
                <State>{state}</State>
                <ZIP>{zip_code}</ZIP>
            </Address>
        </Demographics>
        <Visit>
            <AdmitDate>{admit_date}</AdmitDate>
            <Status>{status}</Status>
            <Type>{encounter_type}</Type>
            <DxCode>{dx_code}</DxCode>
        </Visit>
        <Notes>
            <ProgressNote>{note}</ProgressNote>
        </Notes>
    </PatientRecord>
"""
    return record


def estimate_record_size() -> int:
    """Estimate the size of a single PatientRecord in bytes."""
    sample_record = generate_patient_record(1)
    return len(sample_record.encode('utf-8'))


def calculate_records_needed(target_size_mb: int, record_size: int) -> int:
    """Calculate number of records needed to reach target size."""
    target_bytes = target_size_mb * 1024 * 1024
    # Account for XML header and root element
    header_size = len('<?xml version="1.0" encoding="UTF-8"?>\n<ClinicalData>\n'.encode('utf-8'))
    footer_size = len('</ClinicalData>\n'.encode('utf-8'))
    available_size = target_bytes - header_size - footer_size
    records_needed = int(available_size / record_size)
    return max(1, records_needed)


def generate_xml_file(target_size_mb: int, output_path: Path) -> Tuple[int, int]:
    """Generate an XML file of approximately target_size_mb.
    
    Returns:
        Tuple of (actual_size_mb, record_count)
    """
    print(f"Generating {target_size_mb}MB XML file...")
    
    # Estimate record size
    record_size = estimate_record_size()
    records_needed = calculate_records_needed(target_size_mb, record_size)
    
    print(f"  Estimated record size: {record_size} bytes")
    print(f"  Generating {records_needed:,} records...")
    
    # Write XML file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<ClinicalData>\n')
        
        # Write records
        for i in range(1, records_needed + 1):
            if i % 10000 == 0:
                print(f"  Progress: {i:,}/{records_needed:,} records...")
            f.write(generate_patient_record(i))
        
        # Write footer
        f.write('</ClinicalData>\n')
    
    # Get actual file size
    actual_size = output_path.stat().st_size
    actual_size_mb = actual_size / (1024 * 1024)
    
    print(f"  [OK] Generated: {output_path.name}")
    print(f"  Actual size: {actual_size_mb:.2f} MB ({actual_size:,} bytes)")
    print(f"  Records: {records_needed:,}")
    print()
    
    return actual_size_mb, records_needed


def main():
    """Generate XML test files of various sizes."""
    print("=" * 60)
    print("XML Test File Generator for DataDialysis Benchmarking")
    print("=" * 60)
    print()
    
    # Create test_data directory if it doesn't exist
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Target sizes in MB
    target_sizes = [1, 5, 10, 25, 50, 75, 100]
    
    results = []
    
    for size_mb in target_sizes:
        output_file = test_data_dir / f"test_data_{size_mb}mb.xml"
        actual_size, record_count = generate_xml_file(size_mb, output_file)
        results.append({
            'target_mb': size_mb,
            'actual_mb': actual_size,
            'records': record_count,
            'file': output_file
        })
    
    # Summary
    print("=" * 60)
    print("Generation Summary")
    print("=" * 60)
    print(f"{'Target (MB)':<12} {'Actual (MB)':<12} {'Records':<12} {'File'}")
    print("-" * 60)
    for result in results:
        print(f"{result['target_mb']:<12} {result['actual_mb']:<12.2f} "
              f"{result['records']:<12,} {result['file'].name}")
    
    print()
    print(f"All files saved to: {test_data_dir.absolute()}")
    print()
    print("You can now use these files for benchmarking XML ingestion performance.")


if __name__ == "__main__":
    main()

