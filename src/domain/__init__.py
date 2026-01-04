"""Domain layer for Data-Dialysis.

This module contains the core business logic and golden record schemas.
All domain models are pure Python with no external dependencies beyond Pydantic.
"""

from .golden_record import (
    PatientRecord,
    ClinicalObservation,
    EncounterRecord,
    GoldenRecord,
)

__all__ = [
    "PatientRecord",
    "ClinicalObservation",
    "EncounterRecord",
    "GoldenRecord",
]

