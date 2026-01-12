"""Generate bad data files for circuit breaker benchmarking.

This script creates test files with configurable failure rates to test
circuit breaker behavior under various data quality scenarios.

Security Impact:
    - Generates synthetic test data (not real PII)
    - Files are for testing circuit breaker and validation logic only
"""

import random
import string
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


# Sample data pools for realistic test data generation
FIRST_NAMES = [
    "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Jessica",
    "William", "Ashley", "James", "Amanda", "Christopher", "Melissa", "Daniel",
    "Michelle", "Matthew", "Kimberly", "Anthony", "Amy", "Mark", "Angela",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas",
]


def generate_bad_xml_file(
    output_path: Path,
    num_records: int,
    failure_rate_percent: float,
    target_size_mb: Optional[float] = None
) -> Tuple[float, int]:
    """Generate XML file with configurable failure rate.
    
    Args:
        output_path: Path to output XML file
        num_records: Number of records to generate
        failure_rate_percent: Percentage of records that should fail (0-100)
        target_size_mb: Optional target file size (will adjust num_records if provided)
    
    Returns:
        Tuple of (actual_size_mb, actual_record_count)
    
    Bad record types:
        - MRN too short (< 3 characters)
        - Future birth date
        - Invalid date format
        - Missing required fields
        - Invalid gender values
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_bad = int(num_records * (failure_rate_percent / 100.0))
    num_good = num_records - num_bad
    
    # Create list of record types (True = good, False = bad)
    record_types = [True] * num_good + [False] * num_bad
    random.shuffle(record_types)
    
    xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<ClinicalData>']
    
    record_count = 0
    for i, is_good in enumerate(record_types):
        if is_good:
            # Valid record
            mrn = f"MRN{i:06d}"
            first_name = random.choice(FIRST_NAMES)
            last_name = random.choice(LAST_NAMES)
            birth_date = (datetime.now() - timedelta(days=random.randint(18*365, 80*365))).strftime("%Y-%m-%d")
            gender = random.choice(["male", "female", "other"])
            
            xml_lines.append(f'    <PatientRecord>')
            xml_lines.append(f'        <MRN>{mrn}</MRN>')
            xml_lines.append(f'        <Demographics>')
            xml_lines.append(f'            <FullName>{first_name} {last_name}</FullName>')
            xml_lines.append(f'            <BirthDate>{birth_date}</BirthDate>')
            xml_lines.append(f'            <Gender>{gender}</Gender>')
            xml_lines.append(f'        </Demographics>')
            xml_lines.append(f'    </PatientRecord>')
        else:
            # Bad record - choose random failure type
            failure_type = random.choice(["short_mrn", "future_dob", "invalid_date", "missing_field", "invalid_gender"])
            
            mrn = f"MRN{i:06d}"
            first_name = random.choice(FIRST_NAMES)
            last_name = random.choice(LAST_NAMES)
            
            xml_lines.append(f'    <PatientRecord>')
            
            if failure_type == "short_mrn":
                # MRN too short
                xml_lines.append(f'        <MRN>{random.choice(["A", "AB", "X"])}</MRN>')
                xml_lines.append(f'        <Demographics>')
                xml_lines.append(f'            <FullName>{first_name} {last_name}</FullName>')
                birth_date = (datetime.now() - timedelta(days=random.randint(18*365, 80*365))).strftime("%Y-%m-%d")
                xml_lines.append(f'            <BirthDate>{birth_date}</BirthDate>')
                xml_lines.append(f'        </Demographics>')
            elif failure_type == "future_dob":
                # Future birth date
                xml_lines.append(f'        <MRN>{mrn}</MRN>')
                xml_lines.append(f'        <Demographics>')
                xml_lines.append(f'            <FullName>{first_name} {last_name}</FullName>')
                future_date = (datetime.now() + timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
                xml_lines.append(f'            <BirthDate>{future_date}</BirthDate>')
                xml_lines.append(f'        </Demographics>')
            elif failure_type == "invalid_date":
                # Invalid date format
                xml_lines.append(f'        <MRN>{mrn}</MRN>')
                xml_lines.append(f'        <Demographics>')
                xml_lines.append(f'            <FullName>{first_name} {last_name}</FullName>')
                xml_lines.append(f'            <BirthDate>INVALID-DATE</BirthDate>')
                xml_lines.append(f'        </Demographics>')
            elif failure_type == "missing_field":
                # Missing required field (no BirthDate)
                xml_lines.append(f'        <MRN>{mrn}</MRN>')
                xml_lines.append(f'        <Demographics>')
                xml_lines.append(f'            <FullName>{first_name} {last_name}</FullName>')
                xml_lines.append(f'        </Demographics>')
            elif failure_type == "invalid_gender":
                # Invalid gender value
                xml_lines.append(f'        <MRN>{mrn}</MRN>')
                xml_lines.append(f'        <Demographics>')
                xml_lines.append(f'            <FullName>{first_name} {last_name}</FullName>')
                birth_date = (datetime.now() - timedelta(days=random.randint(18*365, 80*365))).strftime("%Y-%m-%d")
                xml_lines.append(f'            <BirthDate>{birth_date}</BirthDate>')
                xml_lines.append(f'            <Gender>INVALID_GENDER</Gender>')
                xml_lines.append(f'        </Demographics>')
            
            xml_lines.append(f'    </PatientRecord>')
        
        record_count += 1
    
    xml_lines.append('</ClinicalData>')
    
    # Write to file
    xml_content = '\n'.join(xml_lines)
    output_path.write_text(xml_content, encoding='utf-8')
    
    # Calculate actual size
    actual_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    return actual_size_mb, record_count


def generate_bad_json_file(
    output_path: Path,
    num_records: int,
    failure_rate_percent: float
) -> float:
    """Generate JSON file with configurable failure rate.
    
    Args:
        output_path: Path to output JSON file
        num_records: Number of records to generate
        failure_rate_percent: Percentage of records that should fail (0-100)
    
    Returns:
        Actual file size in MB
    
    Bad record types:
        - Invalid date format
        - Missing required fields
        - Invalid data types
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_bad = int(num_records * (failure_rate_percent / 100.0))
    num_good = num_records - num_bad
    
    # Create list of record types (True = good, False = bad)
    record_types = [True] * num_good + [False] * num_bad
    random.shuffle(record_types)
    
    records = []
    
    for i, is_good in enumerate(record_types):
        if is_good:
            # Valid record
            patient_id = f"MRN{i:06d}"
            first_name = random.choice(FIRST_NAMES)
            last_name = random.choice(LAST_NAMES)
            birth_date = (datetime.now() - timedelta(days=random.randint(18*365, 80*365))).strftime("%Y-%m-%d")
            
            record = {
                "patient_id": patient_id,
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": birth_date,
                "gender": random.choice(["M", "F", "O"]),
            }
        else:
            # Bad record - choose random failure type
            failure_type = random.choice(["invalid_date", "missing_field", "invalid_type"])
            
            patient_id = f"MRN{i:06d}"
            first_name = random.choice(FIRST_NAMES)
            last_name = random.choice(LAST_NAMES)
            
            if failure_type == "invalid_date":
                record = {
                    "patient_id": patient_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "date_of_birth": "INVALID-DATE",
                    "gender": random.choice(["M", "F", "O"]),
                }
            elif failure_type == "missing_field":
                record = {
                    "patient_id": patient_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    # Missing date_of_birth
                    "gender": random.choice(["M", "F", "O"]),
                }
            elif failure_type == "invalid_type":
                record = {
                    "patient_id": patient_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "date_of_birth": 12345,  # Invalid type (should be string)
                    "gender": random.choice(["M", "F", "O"]),
                }
        
        records.append(record)
    
    # Write JSON file
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    
    # Calculate actual size
    actual_size_mb = output_path.stat().st_size / (1024 * 1024)
    return actual_size_mb


def generate_bad_csv_file(
    output_path: Path,
    num_records: int,
    failure_rate_percent: float
) -> float:
    """Generate CSV file with configurable failure rate.
    
    Args:
        output_path: Path to output CSV file
        num_records: Number of records to generate
        failure_rate_percent: Percentage of records that should fail (0-100)
    
    Returns:
        Actual file size in MB
    
    Bad record types:
        - Invalid date format
        - Missing required fields (empty values)
        - Invalid data types
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_bad = int(num_records * (failure_rate_percent / 100.0))
    num_good = num_records - num_bad
    
    # Create list of record types (True = good, False = bad)
    record_types = [True] * num_good + [False] * num_bad
    random.shuffle(record_types)
    
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "patient_id", "first_name", "last_name", "date_of_birth", "gender"
        ])
        
        for i, is_good in enumerate(record_types):
            if is_good:
                # Valid record
                patient_id = f"MRN{i:06d}"
                first_name = random.choice(FIRST_NAMES)
                last_name = random.choice(LAST_NAMES)
                birth_date = (datetime.now() - timedelta(days=random.randint(18*365, 80*365))).strftime("%Y-%m-%d")
                gender = random.choice(["M", "F", "O"])
                
                writer.writerow([patient_id, first_name, last_name, birth_date, gender])
            else:
                # Bad record - choose random failure type
                failure_type = random.choice(["invalid_date", "missing_field", "empty_value"])
                
                patient_id = f"MRN{i:06d}"
                first_name = random.choice(FIRST_NAMES)
                last_name = random.choice(LAST_NAMES)
                
                if failure_type == "invalid_date":
                    writer.writerow([patient_id, first_name, last_name, "INVALID-DATE", random.choice(["M", "F", "O"])])
                elif failure_type == "missing_field":
                    writer.writerow([patient_id, first_name, last_name, "", random.choice(["M", "F", "O"])])
                elif failure_type == "empty_value":
                    writer.writerow([patient_id, "", last_name, (datetime.now() - timedelta(days=random.randint(18*365, 80*365))).strftime("%Y-%m-%d"), random.choice(["M", "F", "O"])])
    
    # Calculate actual size
    actual_size_mb = output_path.stat().st_size / (1024 * 1024)
    return actual_size_mb


if __name__ == "__main__":
    # Example usage
    test_dir = Path("test_data/bad_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test files with different failure rates
    for failure_rate in [25, 50, 75, 90]:
        print(f"Generating {failure_rate}% failure rate files...")
        
        # XML
        xml_path = test_dir / f"bad_xml_{failure_rate}pct.xml"
        size, count = generate_bad_xml_file(xml_path, num_records=100, failure_rate_percent=failure_rate)
        print(f"  XML: {xml_path.name} - {size:.2f}MB, {count} records")
        
        # JSON
        json_path = test_dir / f"bad_json_{failure_rate}pct.json"
        size = generate_bad_json_file(json_path, num_records=100, failure_rate_percent=failure_rate)
        print(f"  JSON: {json_path.name} - {size:.2f}MB")
        
        # CSV
        csv_path = test_dir / f"bad_csv_{failure_rate}pct.csv"
        size = generate_bad_csv_file(csv_path, num_records=100, failure_rate_percent=failure_rate)
        print(f"  CSV: {csv_path.name} - {size:.2f}MB")
