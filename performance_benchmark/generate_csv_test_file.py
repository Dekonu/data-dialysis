"""Generate large CSV test file with patient records and observations.

This script creates CSV files with patient, encounter, and observation data
for testing the ingestion pipeline with large datasets.

Security Impact:
    - Generates synthetic test data (not real PII)
    - Files are for testing ingestion performance only
"""

import csv
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


# Sample data pools for realistic test data generation
FIRST_NAMES = [
    "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Jessica",
    "William", "Ashley", "James", "Amanda", "Christopher", "Melissa", "Daniel",
    "Michelle", "Matthew", "Kimberly", "Anthony", "Amy", "Mark", "Angela",
    "Donald", "Stephanie", "Steven", "Nicole", "Paul", "Elizabeth", "Andrew",
    "Helen", "Joshua", "Sandra", "Kenneth", "Donna", "Kevin", "Carol", "Brian",
    "Ruth", "George", "Sharon", "Edward", "Ronald", "Laura"
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

# Observation categories and codes
OBSERVATION_CATEGORIES = ["vital-signs", "laboratory", "imaging", "procedure", "exam"]
OBSERVATION_CODES = {
    "vital-signs": ["8480-6", "8867-4", "9279-1", "8310-5", "29463-7"],
    "laboratory": ["718-7", "777-3", "789-8", "786-4", "785-6", "787-2"],
    "imaging": ["24627-2", "24628-0", "24629-8", "24630-6"],
    "procedure": ["47519-4", "47520-2", "47521-0"],
    "exam": ["8716-3", "8717-1", "8718-9"]
}

OBSERVATION_VALUES = {
    "vital-signs": {
        "8480-6": lambda: f"{random.randint(90, 140)}/{random.randint(60, 90)}",  # BP
        "8867-4": lambda: str(random.randint(60, 100)),  # Heart rate
        "9279-1": lambda: str(random.randint(12, 20)),  # Respiration
        "8310-5": lambda: f"{random.randint(96, 100)}.{random.randint(0, 9)}",  # Temperature
        "29463-7": lambda: str(random.randint(90, 100)),  # O2 saturation
    },
    "laboratory": {
        "718-7": lambda: str(random.randint(4, 6)),  # Hemoglobin
        "777-3": lambda: str(random.randint(35, 50)),  # Hematocrit
        "789-8": lambda: str(random.randint(150, 450)),  # Platelet count
        "786-4": lambda: str(random.randint(4, 11)),  # WBC
        "785-6": lambda: str(random.randint(70, 100)),  # Glucose
        "787-2": lambda: str(random.randint(0, 1)),  # Glucose (fasting)
    }
}

OBSERVATION_UNITS = {
    "vital-signs": {"8480-6": "mmHg", "8867-4": "bpm", "9279-1": "/min", 
                    "8310-5": "F", "29463-7": "%"},
    "laboratory": {"718-7": "g/dL", "777-3": "%", "789-8": "K/uL", 
                   "786-4": "K/uL", "785-6": "mg/dL", "787-2": "mg/dL"}
}

OBSERVATION_NOTES = [
    "Patient presents with normal vital signs.",
    "Follow-up observation. Results within normal range.",
    "Routine monitoring. No acute concerns.",
    "Patient reports improvement. Continue monitoring.",
    "Baseline measurement established.",
    "Results consistent with previous observations.",
    "Patient advised to continue current regimen.",
    "No significant changes from baseline."
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


def generate_date_of_birth() -> str:
    """Generate a date of birth between 1920 and 2010."""
    start_date = datetime(1920, 1, 1)
    end_date = datetime(2010, 12, 31)
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime("%Y-%m-%d")


def generate_postal_code() -> str:
    """Generate a 5-digit postal code."""
    return f"{random.randint(10000, 99999)}"


def generate_observation(
    patient_id: str,
    observation_num: int,
    base_date: datetime
) -> Dict[str, Any]:
    """Generate a single observation record.
    
    Parameters:
        patient_id: Patient identifier
        observation_num: Observation number for this patient
        base_date: Base date for observation timing
        
    Returns:
        Dictionary with observation data
    """
    category = random.choice(OBSERVATION_CATEGORIES)
    codes = OBSERVATION_CODES.get(category, OBSERVATION_CODES["vital-signs"])
    code = random.choice(codes)
    
    # Generate value based on code
    value = None
    unit = None
    if category in OBSERVATION_VALUES and code in OBSERVATION_VALUES[category]:
        value = OBSERVATION_VALUES[category][code]()
        unit = OBSERVATION_UNITS.get(category, {}).get(code)
    
    # Generate effective date (within last 2 years)
    days_ago = random.randint(0, 730)
    effective_date = (base_date - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")
    
    observation = {
        "observation_id": f"OBS{patient_id}{observation_num:03d}",
        "patient_id": patient_id,
        "status": random.choice(["final", "preliminary", "registered", "amended"]),
        "category": category,
        "code": code,
        "effective_date": effective_date,
        "value": value,
        "unit": unit,
        "notes": random.choice(OBSERVATION_NOTES)
    }
    
    # Add optional fields randomly
    if random.random() > 0.5:
        observation["issued"] = effective_date
    if random.random() > 0.7:
        observation["performer_name"] = f"Dr. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    if random.random() > 0.8:
        observation["interpretation"] = random.choice(["normal", "abnormal", "high", "low", "critical"])
    if random.random() > 0.9:
        observation["reference_range"] = "See reference values"
    
    return observation


def generate_patient_record(record_num: int, base_date: datetime) -> Dict[str, Any]:
    """Generate a single patient record with observations.
    
    Parameters:
        record_num: Record number
        base_date: Base date for generating dates
        
    Returns:
        Dictionary with patient, encounters, and observations
    """
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    patient_id = f"MRN{record_num:06d}"
    
    # Generate patient data
    patient = {
        "patient_id": patient_id,
        "first_name": first_name,
        "last_name": last_name,
        "date_of_birth": generate_date_of_birth(),
        "gender": random.choice(GENDERS),
        "ssn": generate_ssn(),
        "phone": generate_phone(),
        "email": generate_email(first_name, last_name),
        "address_line1": generate_street_address(),
        "city": random.choice(CITIES),
        "state": random.choice(STATES),
        "postal_code": generate_postal_code()
    }
    
    # Generate 1-5 observations per patient
    num_observations = random.randint(1, 5)
    observations = [
        generate_observation(patient_id, i + 1, base_date)
        for i in range(num_observations)
    ]
    
    # Generate 0-2 encounters (optional)
    encounters = []
    if random.random() > 0.3:  # 70% chance of having an encounter
        num_encounters = random.randint(1, 2)
        for i in range(num_encounters):
            encounter_date = base_date - timedelta(days=random.randint(0, 365))
            encounter = {
                "encounter_id": f"ENC{patient_id}{i+1:02d}",
                "patient_id": patient_id,
                "status": random.choice(["finished", "in-progress", "planned", "arrived"]),
                "class_code": random.choice(["outpatient", "inpatient", "emergency", "ambulatory"]),
                "period_start": encounter_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "period_end": (encounter_date + timedelta(hours=random.randint(1, 4))).strftime("%Y-%m-%dT%H:%M:%S"),
                "diagnosis_codes": random.sample(
                    ["I10", "E11.9", "I50.9", "J44.1", "N18.6", "E78.5"],
                    k=random.randint(1, 3)
                )
            }
            encounters.append(encounter)
    
    return {
        "patient": patient,
        "encounters": encounters,
        "observations": observations
    }


def generate_csv_file(
    output_path: Path,
    num_records: int = 200000,
    format_type: str = "flat",
    delimiter: str = ","
):
    """Generate CSV file(s) with patient records and observations.
    
    Parameters:
        output_path: Base path for CSV files (will append _patients.csv, _encounters.csv, _observations.csv)
        num_records: Number of patient records to generate (default: 200,000)
        format_type: CSV format - "flat" (separate files) or "denormalized" (single file with all data)
        delimiter: CSV delimiter character (default: ",")
    """
    print(f"Generating CSV file(s) with {num_records:,} patient records...")
    
    if format_type == "flat":
        # Generate separate CSV files for patients, encounters, and observations
        patients_path = output_path.parent / f"{output_path.stem}_patients.csv"
        encounters_path = output_path.parent / f"{output_path.stem}_encounters.csv"
        observations_path = output_path.parent / f"{output_path.stem}_observations.csv"
    else:
        # Single denormalized file (one row per observation, with patient data repeated)
        patients_path = output_path
        encounters_path = None
        observations_path = output_path  # Same file for denormalized format
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_date = datetime.now()
    
    # Patient CSV columns
    patient_columns = [
        "patient_id", "first_name", "last_name", "date_of_birth", "gender",
        "ssn", "phone", "email", "address_line1", "city", "state", "postal_code"
    ]
    
    # Encounter CSV columns
    encounter_columns = [
        "encounter_id", "patient_id", "status", "class_code",
        "period_start", "period_end", "diagnosis_codes"
    ]
    
    # Observation CSV columns
    observation_columns = [
        "observation_id", "patient_id", "status", "category", "code",
        "effective_date", "value", "unit", "notes", "issued",
        "performer_name", "interpretation", "reference_range"
    ]
    
    # Generate records in batches to show progress
    batch_size = 10000
    
    # Open CSV files for writing with proper exception handling
    patients_file = None
    encounters_file = None
    observations_file = None
    patients_writer = None
    encounters_writer = None
    observations_writer = None
    
    try:
        # Open CSV files for writing
        patients_file = open(patients_path, 'w', newline='', encoding='utf-8')
        patients_writer = csv.DictWriter(patients_file, fieldnames=patient_columns, delimiter=delimiter)
        patients_writer.writeheader()
        
        if format_type == "flat":
            encounters_file = open(encounters_path, 'w', newline='', encoding='utf-8')
            encounters_writer = csv.DictWriter(encounters_file, fieldnames=encounter_columns, delimiter=delimiter)
            encounters_writer.writeheader()
            
            observations_file = open(observations_path, 'w', newline='', encoding='utf-8')
            observations_writer = csv.DictWriter(observations_file, fieldnames=observation_columns, delimiter=delimiter)
            observations_writer.writeheader()
        else:
            # Denormalized format: combine all columns
            denormalized_columns = patient_columns + observation_columns[1:]  # Skip observation_id, include patient_id
            # Close patients_file first (not used in denormalized format)
            patients_file.close()
            patients_file = None
            patients_writer = None
            
            observations_file = open(patients_path, 'w', newline='', encoding='utf-8')  # Use patients_path for denormalized
            observations_writer = csv.DictWriter(observations_file, fieldnames=denormalized_columns, delimiter=delimiter)
            observations_writer.writeheader()
            encounters_file = None
            encounters_writer = None
        
        # Main processing loop
        for batch_start in range(0, num_records, batch_size):
            batch_end = min(batch_start + batch_size, num_records)
            print(f"Generating records {batch_start:,} to {batch_end:,}...")
            
            for record_num in range(batch_start + 1, batch_end + 1):
                record = generate_patient_record(record_num, base_date)
                patient = record["patient"]
                encounters = record["encounters"]
                observations = record["observations"]
                
                if format_type == "flat":
                    # Write patient row
                    patient_row = {col: patient.get(col, "") for col in patient_columns}
                    patients_writer.writerow(patient_row)
                    
                    # Write encounter rows
                    # Write encounter rows
                    for encounter in encounters:
                        encounter_row = {
                            "encounter_id": encounter["encounter_id"],
                            "patient_id": encounter["patient_id"],
                            "status": encounter["status"],
                            "class_code": encounter["class_code"],
                            "period_start": encounter["period_start"],
                            "period_end": encounter["period_end"],
                            "diagnosis_codes": ",".join(encounter["diagnosis_codes"])  # Join array as comma-separated
                        }
                        encounters_writer.writerow(encounter_row)
                    
                    # Write observation rows
                    for observation in observations:
                        observation_row = {
                            "observation_id": observation["observation_id"],
                            "patient_id": observation["patient_id"],
                            "status": observation.get("status", ""),
                            "category": observation.get("category", ""),
                            "code": observation.get("code", ""),
                            "effective_date": observation.get("effective_date", ""),
                            "value": observation.get("value", ""),
                            "unit": observation.get("unit", ""),
                            "notes": observation.get("notes", ""),
                            "issued": observation.get("issued", ""),
                            "performer_name": observation.get("performer_name", ""),
                            "interpretation": observation.get("interpretation", ""),
                            "reference_range": observation.get("reference_range", "")
                        }
                        observations_writer.writerow(observation_row)
                else:
                    # Denormalized format: one row per observation with patient data
                    for observation in observations:
                        # Combine patient and observation data
                        combined_row = {col: patient.get(col, "") for col in patient_columns}
                        combined_row.update({
                            "status": observation.get("status", ""),
                            "category": observation.get("category", ""),
                            "code": observation.get("code", ""),
                            "effective_date": observation.get("effective_date", ""),
                            "value": observation.get("value", ""),
                            "unit": observation.get("unit", ""),
                            "notes": observation.get("notes", ""),
                            "issued": observation.get("issued", ""),
                            "performer_name": observation.get("performer_name", ""),
                            "interpretation": observation.get("interpretation", ""),
                            "reference_range": observation.get("reference_range", "")
                        })
                        observations_writer.writerow(combined_row)
    
    finally:
        # Ensure all files are closed, even if an exception occurs
        if patients_file is not None:
            try:
                patients_file.close()
            except Exception:
                pass
        if encounters_file is not None:
            try:
                encounters_file.close()
            except Exception:
                pass
        if observations_file is not None:
            try:
                observations_file.close()
            except Exception:
                pass
    
    # Calculate totals after all files are written
    total_observations = 0
    total_encounters = 0
    
    # Count observations
    if format_type == "flat":
        with open(observations_path, 'r', encoding='utf-8') as f:
            observations_reader = csv.reader(f, delimiter=delimiter)
            next(observations_reader)  # Skip header
            total_observations = sum(1 for _ in observations_reader)
        
        # Count encounters
        with open(encounters_path, 'r', encoding='utf-8') as f:
            encounters_reader = csv.reader(f, delimiter=delimiter)
            next(encounters_reader)  # Skip header
            total_encounters = sum(1 for _ in encounters_reader)
    else:
        # For denormalized format, count rows in the single file
        with open(patients_path, 'r', encoding='utf-8') as f:
            observations_reader = csv.reader(f, delimiter=delimiter)
            next(observations_reader)  # Skip header
            total_observations = sum(1 for _ in observations_reader)
    
    # Calculate file sizes
    patients_size = patients_path.stat().st_size
    patients_size_mb = patients_size / (1024 * 1024)
    
    print(f"\nSUCCESS: CSV file(s) generated successfully!")
    print(f"  Patients file: {patients_path}")
    print(f"  Size: {patients_size_mb:.2f} MB ({patients_size:,} bytes)")
    print(f"  Records: {num_records:,}")
    
    if format_type == "flat":
        encounters_size = encounters_path.stat().st_size
        encounters_size_mb = encounters_size / (1024 * 1024)
        observations_size = observations_path.stat().st_size
        observations_size_mb = observations_size / (1024 * 1024)
        
        print(f"  Encounters file: {encounters_path}")
        print(f"  Size: {encounters_size_mb:.2f} MB ({encounters_size:,} bytes)")
        print(f"  Records: {total_encounters:,}")
        print(f"  Observations file: {observations_path}")
        print(f"  Size: {observations_size_mb:.2f} MB ({observations_size:,} bytes)")
        print(f"  Records: {total_observations:,}")
    else:
        observations_size = observations_path.stat().st_size
        observations_size_mb = observations_size / (1024 * 1024)
        print(f"  Observations file: {observations_path}")
        print(f"  Size: {observations_size_mb:.2f} MB ({observations_size:,} bytes)")
        print(f"  Records: {total_observations:,}")
    
    print(f"  Average observations per patient: {total_observations / num_records:.2f}")
    if format_type == "flat":
        print(f"  Average encounters per patient: {total_encounters / num_records:.2f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate large CSV test file(s) with patient records and observations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: test_data/test_data_{num_records}_patients.csv for flat format)"
    )
    parser.add_argument(
        "--records",
        type=int,
        default=200000,
        help="Number of patient records to generate (default: 200000)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["flat", "denormalized"],
        default="flat",
        help="CSV format: 'flat' (separate files) or 'denormalized' (single file) (default: flat)"
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter character (default: ',')"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        if args.format == "flat":
            args.output = Path(f"test_data/test_data_{args.records}_patients.csv")
        else:
            args.output = Path(f"test_data/test_data_{args.records}_denormalized.csv")
    
    generate_csv_file(args.output, args.records, args.format, args.delimiter)


if __name__ == "__main__":
    main()

