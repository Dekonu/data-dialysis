"""Generate large JSON test file with patient records and observations.

This script creates a JSON file with 200,000 patient records, each containing
observations for testing the ingestion pipeline with large datasets.

Security Impact:
    - Generates synthetic test data (not real PII)
    - Files are for testing ingestion performance only
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any


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


def generate_json_file(output_path: Path, num_records: int = 200000):
    """Generate a large JSON file with patient records and observations.
    
    Parameters:
        output_path: Path where JSON file should be written
        num_records: Number of patient records to generate (default: 200,000)
    """
    print(f"Generating JSON file with {num_records:,} records...")
    if output_path is None:
        output_path = Path(f"test_data/test_data_{num_records}.json")

    print(f"Output: {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_date = datetime.now()
    records = []
    
    # Generate records in batches to show progress
    batch_size = 10000
    for batch_start in range(0, num_records, batch_size):
        batch_end = min(batch_start + batch_size, num_records)
        print(f"Generating records {batch_start:,} to {batch_end:,}...")
        
        batch_records = [
            generate_patient_record(i + 1, base_date)
            for i in range(batch_start, batch_end)
        ]
        records.extend(batch_records)
    
    # Write JSON file
    print(f"Writing {len(records):,} records to JSON file...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    
    # Calculate file size
    file_size = output_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    # Calculate statistics
    total_observations = sum(len(r["observations"]) for r in records)
    total_encounters = sum(len(r["encounters"]) for r in records)
    
    print(f"\n✓ JSON file generated successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"  Records: {len(records):,}")
    print(f"  Total observations: {total_observations:,}")
    print(f"  Total encounters: {total_encounters:,}")
    print(f"  Average observations per patient: {total_observations / len(records):.2f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate large JSON test file with patient records and observations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: test_data/test_data_{num_records}.json)"
    )
    parser.add_argument(
        "--records",
        type=int,
        default=200000,
        help="Number of patient records to generate (default: 200000)"
    )
    
    args = parser.parse_args()
    
    generate_json_file(args.output, args.records)


if __name__ == "__main__":
    main()

