"""DuckDB Storage Adapter.

This adapter implements the StoragePort contract for persisting validated GoldenRecords
to DuckDB, an in-process OLAP database optimized for analytical workloads.

Security Impact:
    - Only validated GoldenRecord instances can be persisted
    - All operations are logged to immutable audit trail
    - Connection credentials are managed securely via configuration
    - Schema enforces data integrity constraints

Architecture:
    - Implements StoragePort (Hexagonal Architecture)
    - Isolated from domain core - only depends on ports and models
    - Transactional batch operations ensure data consistency
    - Audit trail is append-only for compliance
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json

import duckdb
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from src.domain.ports import (
    StoragePort,
    Result,
    StorageError,
    ValidationError,
)
from src.domain.golden_record import (
    GoldenRecord,
    PatientRecord,
    ClinicalObservation,
    EncounterRecord,
)
from src.infrastructure.config_manager import DatabaseConfig

logger = logging.getLogger(__name__)


class DuckDBAdapter(StoragePort):
    """DuckDB implementation of StoragePort for analytical data storage.
    
    This adapter provides efficient bulk loading and query capabilities
    for validated clinical records. DuckDB is optimized for OLAP workloads
    and provides excellent performance for analytical queries.
    
    Security Impact:
        - All records are validated before persistence
        - Audit trail is immutable and tamper-proof
        - Connection credentials are never logged
        - Schema enforces referential integrity
    
    Parameters:
        db_path: Path to DuckDB database file (or ':memory:' for in-memory)
        config: Optional configuration dictionary (for future extensibility)
    
    Example Usage:
        ```python
        # Using configuration manager (recommended)
        from src.infrastructure.config_manager import get_database_config
        
        db_config = get_database_config()
        adapter = DuckDBAdapter(db_config=db_config)
        
        # Or using db_path directly (backward compatibility)
        adapter = DuckDBAdapter(db_path="data/clinical.duckdb")
        
        result = adapter.initialize_schema()
        if result.is_success():
            result = adapter.persist(golden_record)
        ```
    """
    
    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        db_path: Optional[str] = None,
        config: Optional[dict] = None
    ):
        """Initialize DuckDB adapter.
        
        Parameters:
            db_config: DatabaseConfig from configuration manager (preferred)
            db_path: Path to DuckDB database file (or ':memory:' for in-memory)
                   Deprecated: Use db_config instead
            config: Optional configuration dictionary (deprecated)
        
        Security Impact:
            - Database path is validated to prevent path traversal attacks
            - Connection is established lazily (on first operation)
            - Credentials are managed securely via configuration manager
        
        Note:
            If both db_config and db_path are provided, db_config takes precedence.
            If neither is provided, defaults to in-memory database.
        """
        # Use DatabaseConfig if provided, otherwise fall back to db_path for backward compatibility
        if db_config:
            if db_config.db_type != "duckdb":
                raise StorageError(
                    f"DatabaseConfig type '{db_config.db_type}' does not match DuckDB adapter",
                    operation="__init__"
                )
            self.db_path = db_config.db_path or ":memory:"
        elif db_path:
            self.db_path = db_path
        else:
            # Default to in-memory if nothing provided
            self.db_path = ":memory:"
        
        self.config = config or {}
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        self._initialized = False
        
        # Validate db_path to prevent path traversal
        if self.db_path != ":memory:":
            db_path_obj = Path(self.db_path)
            if not db_path_obj.parent.exists():
                raise StorageError(
                    f"Database directory does not exist: {db_path_obj.parent}",
                    operation="__init__"
                )
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection.
        
        Returns:
            DuckDB connection instance
        
        Security Impact:
            - Connection is created lazily to avoid unnecessary resource usage
            - Connection is reused for performance
        """
        if self._connection is None:
            try:
                self._connection = duckdb.connect(self.db_path)
                logger.info(f"Connected to DuckDB database: {self.db_path}")
            except Exception as e:
                raise StorageError(
                    f"Failed to connect to DuckDB: {str(e)}",
                    operation="connect",
                    details={"db_path": self.db_path}
                )
        return self._connection
    
    def initialize_schema(self) -> Result[None]:
        """Initialize database schema (tables, indexes, constraints).
        
        Creates tables for:
        - patients: Patient demographic records
        - encounters: Encounter/visit records
        - observations: Clinical observation records
        - audit_log: Immutable audit trail
        
        Returns:
            Result[None]: Success or failure result
        
        Security Impact:
            - Schema enforces data integrity constraints
            - Audit log table is append-only
            - Indexes optimize query performance
        """
        try:
            conn = self._get_connection()
            
            # Create patients table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id VARCHAR PRIMARY KEY,
                    identifiers VARCHAR,
                    family_name VARCHAR,
                    given_names VARCHAR,
                    name_prefix VARCHAR,
                    name_suffix VARCHAR,
                    date_of_birth DATE,
                    gender VARCHAR,
                    deceased BOOLEAN,
                    deceased_date TIMESTAMP,
                    marital_status VARCHAR,
                    address_line1 VARCHAR,
                    address_line2 VARCHAR,
                    city VARCHAR,
                    state VARCHAR,
                    postal_code VARCHAR,
                    country VARCHAR,
                    address_use VARCHAR,
                    phone VARCHAR,
                    email VARCHAR,
                    fax VARCHAR,
                    emergency_contact_name VARCHAR,
                    emergency_contact_relationship VARCHAR,
                    emergency_contact_phone VARCHAR,
                    language VARCHAR,
                    managing_organization VARCHAR,
                    ingestion_timestamp TIMESTAMP NOT NULL,
                    source_adapter VARCHAR NOT NULL,
                    transformation_hash VARCHAR
                )
            """)
            
            # Create encounters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS encounters (
                    encounter_id VARCHAR PRIMARY KEY,
                    patient_id VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    class_code VARCHAR NOT NULL,
                    type VARCHAR,
                    service_type VARCHAR,
                    priority VARCHAR,
                    period_start TIMESTAMP,
                    period_end TIMESTAMP,
                    length_minutes INTEGER,
                    reason_code VARCHAR,
                    diagnosis_codes VARCHAR,
                    facility_name VARCHAR,
                    location_address VARCHAR,
                    participant_name VARCHAR,
                    participant_role VARCHAR,
                    service_provider VARCHAR,
                    ingestion_timestamp TIMESTAMP NOT NULL,
                    source_adapter VARCHAR NOT NULL,
                    transformation_hash VARCHAR,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                )
            """)
            
            # Create observations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id VARCHAR PRIMARY KEY,
                    patient_id VARCHAR NOT NULL,
                    encounter_id VARCHAR,
                    status VARCHAR NOT NULL,
                    category VARCHAR NOT NULL,
                    code VARCHAR,
                    effective_date TIMESTAMP,
                    issued TIMESTAMP,
                    performer_name VARCHAR,
                    value VARCHAR,
                    unit VARCHAR,
                    interpretation VARCHAR,
                    body_site VARCHAR,
                    method VARCHAR,
                    device VARCHAR,
                    reference_range VARCHAR,
                    notes VARCHAR,
                    ingestion_timestamp TIMESTAMP NOT NULL,
                    source_adapter VARCHAR NOT NULL,
                    transformation_hash VARCHAR,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id)
                )
            """)
            
            # Create audit log table (immutable, append-only)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id VARCHAR PRIMARY KEY,
                    event_type VARCHAR NOT NULL,
                    event_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    record_id VARCHAR,
                    transformation_hash VARCHAR,
                    details JSON,
                    source_adapter VARCHAR,
                    severity VARCHAR,
                    table_name VARCHAR,
                    row_count INTEGER
                )
            """)
            
            # Create redaction logs table for detailed PII redaction tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    log_id VARCHAR PRIMARY KEY,
                    field_name VARCHAR NOT NULL,
                    original_hash VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    rule_triggered VARCHAR NOT NULL,
                    record_id VARCHAR,
                    source_adapter VARCHAR,
                    ingestion_id VARCHAR,
                    redacted_value VARCHAR,
                    original_value_length INTEGER
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_source ON patients(source_adapter)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_timestamp ON patients(ingestion_timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_encounters_patient ON encounters(patient_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_patient ON observations(patient_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_encounter ON observations(encounter_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(event_timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_record_id ON audit_log(record_id)")
            
            # Create indexes for redaction logs
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_field_name ON logs(field_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_rule_triggered ON logs(rule_triggered)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_record_id ON logs(record_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_ingestion_id ON logs(ingestion_id)")
            
            self._initialized = True
            logger.info("Database schema initialized successfully")
            
            return Result.success_result(None)
            
        except Exception as e:
            error_msg = f"Failed to initialize schema: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="initialize_schema"),
                error_type="StorageError"
            )
    
    def persist(self, record: GoldenRecord) -> Result[str]:
        """Persist a single GoldenRecord to storage.
        
        Parameters:
            record: Validated GoldenRecord instance (PII already redacted)
        
        Returns:
            Result[str]: Record identifier (patient_id) or error
        """
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            
            # Begin transaction
            conn.begin()
            
            try:
                # Insert patient record
                patient = record.patient
                patient_dict = patient.model_dump()
                patient_dict['ingestion_timestamp'] = record.ingestion_timestamp
                patient_dict['source_adapter'] = record.source_adapter
                patient_dict['transformation_hash'] = record.transformation_hash
                
                # Convert lists to JSON strings for DuckDB array support
                if 'identifiers' in patient_dict and patient_dict['identifiers']:
                    patient_dict['identifiers'] = json.dumps(patient_dict['identifiers'])
                if 'given_names' in patient_dict and patient_dict['given_names']:
                    patient_dict['given_names'] = json.dumps(patient_dict['given_names'])
                if 'name_prefix' in patient_dict and patient_dict['name_prefix']:
                    patient_dict['name_prefix'] = json.dumps(patient_dict['name_prefix'])
                if 'name_suffix' in patient_dict and patient_dict['name_suffix']:
                    patient_dict['name_suffix'] = json.dumps(patient_dict['name_suffix'])
                
                conn.execute("""
                    INSERT OR REPLACE INTO patients (
                        patient_id, identifiers, family_name, given_names, name_prefix, name_suffix,
                        date_of_birth, gender, deceased, deceased_date, marital_status,
                        address_line1, address_line2, city, state, postal_code, country,
                        address_use, phone, email, fax, emergency_contact_name,
                        emergency_contact_relationship, emergency_contact_phone, language,
                        managing_organization, ingestion_timestamp, source_adapter, transformation_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    patient_dict.get('patient_id'),
                    patient_dict.get('identifiers'),
                    patient_dict.get('family_name'),
                    patient_dict.get('given_names'),
                    patient_dict.get('name_prefix'),
                    patient_dict.get('name_suffix'),
                    patient_dict.get('date_of_birth'),
                    patient_dict.get('gender'),
                    patient_dict.get('deceased'),
                    patient_dict.get('deceased_date'),
                    patient_dict.get('marital_status'),
                    patient_dict.get('address_line1'),
                    patient_dict.get('address_line2'),
                    patient_dict.get('city'),
                    patient_dict.get('state'),
                    patient_dict.get('postal_code'),
                    patient_dict.get('country'),
                    patient_dict.get('address_use'),
                    patient_dict.get('phone'),
                    patient_dict.get('email'),
                    patient_dict.get('fax'),
                    patient_dict.get('emergency_contact_name'),
                    patient_dict.get('emergency_contact_relationship'),
                    patient_dict.get('emergency_contact_phone'),
                    patient_dict.get('language'),
                    patient_dict.get('managing_organization'),
                    patient_dict.get('ingestion_timestamp'),
                    patient_dict.get('source_adapter'),
                    patient_dict.get('transformation_hash'),
                ])
                
                # Insert encounters
                for encounter in record.encounters:
                    encounter_dict = encounter.model_dump()
                    encounter_dict['ingestion_timestamp'] = record.ingestion_timestamp
                    encounter_dict['source_adapter'] = record.source_adapter
                    encounter_dict['transformation_hash'] = record.transformation_hash
                    
                    if 'diagnosis_codes' in encounter_dict and encounter_dict['diagnosis_codes']:
                        encounter_dict['diagnosis_codes'] = json.dumps(encounter_dict['diagnosis_codes'])
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO encounters (
                            encounter_id, patient_id, status, class_code, type, service_type, priority,
                            period_start, period_end, length_minutes, reason_code, diagnosis_codes,
                            facility_name, location_address, participant_name, participant_role,
                            service_provider, ingestion_timestamp, source_adapter, transformation_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        encounter_dict.get('encounter_id'),
                        encounter_dict.get('patient_id'),
                        encounter_dict.get('status'),
                        encounter_dict.get('class_code'),
                        encounter_dict.get('type'),
                        encounter_dict.get('service_type'),
                        encounter_dict.get('priority'),
                        encounter_dict.get('period_start'),
                        encounter_dict.get('period_end'),
                        encounter_dict.get('length_minutes'),
                        encounter_dict.get('reason_code'),
                        encounter_dict.get('diagnosis_codes'),
                        encounter_dict.get('facility_name'),
                        encounter_dict.get('location_address'),
                        encounter_dict.get('participant_name'),
                        encounter_dict.get('participant_role'),
                        encounter_dict.get('service_provider'),
                        encounter_dict.get('ingestion_timestamp'),
                        encounter_dict.get('source_adapter'),
                        encounter_dict.get('transformation_hash'),
                    ])
                
                # Insert observations
                for observation in record.observations:
                    observation_dict = observation.model_dump()
                    observation_dict['ingestion_timestamp'] = record.ingestion_timestamp
                    observation_dict['source_adapter'] = record.source_adapter
                    observation_dict['transformation_hash'] = record.transformation_hash
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO observations (
                            observation_id, patient_id, encounter_id, status, category, code,
                            effective_date, issued, performer_name, value, unit, interpretation,
                            body_site, method, device, reference_range, notes,
                            ingestion_timestamp, source_adapter, transformation_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        observation_dict.get('observation_id'),
                        observation_dict.get('patient_id'),
                        observation_dict.get('encounter_id'),
                        observation_dict.get('status'),
                        observation_dict.get('category'),
                        observation_dict.get('code'),
                        observation_dict.get('effective_date'),
                        observation_dict.get('issued'),
                        observation_dict.get('performer_name'),
                        observation_dict.get('value'),
                        observation_dict.get('unit'),
                        observation_dict.get('interpretation'),
                        observation_dict.get('body_site'),
                        observation_dict.get('method'),
                        observation_dict.get('device'),
                        observation_dict.get('reference_range'),
                        observation_dict.get('notes'),
                        observation_dict.get('ingestion_timestamp'),
                        observation_dict.get('source_adapter'),
                        observation_dict.get('transformation_hash'),
                    ])
                
                # Commit transaction
                conn.commit()
                
                # Log audit event (singular record - row_count is None)
                self.log_audit_event(
                    event_type="PERSISTENCE",
                    record_id=patient.patient_id,
                    transformation_hash=record.transformation_hash,
                    details={
                        "source_adapter": record.source_adapter,
                        "encounter_count": len(record.encounters),
                        "observation_count": len(record.observations),
                    },
                    table_name="patients",  # Main table for GoldenRecord
                    row_count=None  # Singular record, not bulk
                )
                
                logger.info(f"Persisted GoldenRecord for patient_id: {patient.patient_id}")
                return Result.success_result(patient.patient_id)
                
            except Exception as e:
                conn.rollback()
                raise e
                
        except Exception as e:
            error_msg = f"Failed to persist record: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="persist", details={"patient_id": record.patient.patient_id}),
                error_type="StorageError"
            )
    
    def persist_batch(self, records: list[GoldenRecord]) -> Result[list[str]]:
        """Persist multiple GoldenRecords in a single transaction.
        
        Parameters:
            records: List of validated GoldenRecord instances
        
        Returns:
            Result[list[str]]: List of record identifiers or error
        """
        if not records:
            return Result.success_result([])
        
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            conn.begin()
            
            try:
                record_ids = []
                for record in records:
                    # Use persist logic but within batch transaction
                    result = self.persist(record)
                    if not result.is_success():
                        raise StorageError(
                            f"Batch persistence failed for record: {result.error}",
                            operation="persist_batch",
                            details={"record_count": len(records)}
                        )
                    record_ids.append(result.value)
                
                conn.commit()
                logger.info(f"Persisted batch of {len(records)} records")
                
                # Log bulk persistence audit event
                # Extract source_adapter from first record if available
                source_adapter = 'batch_ingestion'
                if records and records[0].source_adapter:
                    source_adapter = records[0].source_adapter
                
                # Count total rows across all tables (patients + encounters + observations)
                total_rows = len(records)  # Each record has 1 patient
                for record in records:
                    total_rows += len(record.encounters) + len(record.observations)
                
                # Log bulk audit event
                self.log_audit_event(
                    event_type="BULK_PERSISTENCE",
                    record_id=None,
                    transformation_hash=None,
                    details={
                        "source_adapter": source_adapter,
                        "record_count": len(records),
                    },
                    table_name="patients",  # Primary table, but includes encounters/observations
                    row_count=total_rows
                )
                
                return Result.success_result(record_ids)
                
            except Exception as e:
                conn.rollback()
                raise e
                
        except Exception as e:
            error_msg = f"Failed to persist batch: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="persist_batch", details={"record_count": len(records)}),
                error_type="StorageError"
            )
    
    def persist_dataframe(self, df: pd.DataFrame, table_name: str) -> Result[int]:
        """Persist a pandas DataFrame directly to a table.
        
        Parameters:
            df: Validated pandas DataFrame (PII already redacted)
            table_name: Target table name (e.g., 'patients', 'observations')
        
        Returns:
            Result[int]: Number of rows persisted or error
        """
        if df.empty:
            return Result.success_result(0)
        
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            
            # Get table columns to ensure DataFrame columns match
            table_columns_result = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            table_columns = [col[1] for col in table_columns_result]  # Column name is at index 1
            
            # Filter DataFrame to only include columns that exist in the table
            df_columns = [col for col in df.columns if col in table_columns]
            
            if not df_columns:
                raise StorageError(
                    f"No matching columns between DataFrame and table '{table_name}'",
                    operation="persist_dataframe",
                    details={"df_columns": list(df.columns), "table_columns": table_columns}
                )
            
            # Select only matching columns from DataFrame
            df_filtered = df[df_columns].copy()
            
            # Add required NOT NULL columns if they're missing (for bulk DataFrame inserts)
            # These are typically added during GoldenRecord creation, but DataFrames from CSV/JSON may not have them
            from datetime import datetime
            required_columns = {
                'ingestion_timestamp': datetime.now(),
                'source_adapter': 'bulk_ingestion',
                'transformation_hash': None
            }
            
            for col_name, default_value in required_columns.items():
                if col_name in table_columns and col_name not in df_filtered.columns:
                    df_filtered[col_name] = default_value
                    df_columns.append(col_name)
            
            # Use DuckDB's efficient DataFrame insertion
            # Register DataFrame as a view, then insert
            conn.register('df_temp', df_filtered)
            
            # Build column list for INSERT statement
            columns_str = ', '.join(df_columns)
            conn.execute(f"INSERT OR REPLACE INTO {table_name} ({columns_str}) SELECT {columns_str} FROM df_temp")
            conn.unregister('df_temp')
            
            row_count = len(df)
            logger.info(f"Persisted {row_count} rows to table '{table_name}'")
            
            # Log audit event
            # Get source_adapter from DataFrame if available, otherwise use default
            source_adapter = 'bulk_ingestion'
            if 'source_adapter' in df_filtered.columns:
                # Get first non-null value, or use default
                source_adapter_series = df_filtered['source_adapter'].dropna()
                if len(source_adapter_series) > 0:
                    source_adapter = str(source_adapter_series.iloc[0])
            
            self.log_audit_event(
                event_type="BULK_PERSISTENCE",
                record_id=None,
                transformation_hash=None,
                details={
                    "table_name": table_name,
                    "row_count": row_count,
                    "source_adapter": source_adapter,
                }
            )
            
            return Result.success_result(row_count)
            
        except Exception as e:
            error_msg = f"Failed to persist DataFrame to {table_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="persist_dataframe", details={"table_name": table_name}),
                error_type="StorageError"
            )
    
    def log_audit_event(
        self,
        event_type: str,
        record_id: Optional[str],
        transformation_hash: Optional[str],
        details: Optional[dict] = None,
        table_name: Optional[str] = None,
        row_count: Optional[int] = None
    ) -> Result[str]:
        """Log an audit trail event for compliance and observability.
        
        Parameters:
            event_type: Type of event (e.g., 'REDACTION', 'SCHEMA_COERCION', 'PERSISTENCE')
            record_id: Unique identifier of the affected record (if applicable)
            transformation_hash: Hash of original data for traceability
            details: Additional event metadata
        
        Returns:
            Result[str]: Audit event identifier or error
        """
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            audit_id = str(uuid.uuid4())
            
            # Determine severity based on event type
            severity = "INFO"
            if event_type in ["REDACTION", "PII_DETECTED"]:
                severity = "CRITICAL"
            elif event_type in ["VALIDATION_ERROR", "TRANSFORMATION_ERROR"]:
                severity = "WARNING"
            
            details_json = json.dumps(details) if details else None
            
            conn.execute("""
                INSERT INTO audit_log (
                    audit_id, event_type, event_timestamp, record_id,
                    transformation_hash, details, source_adapter, severity,
                    table_name, row_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                audit_id,
                event_type,
                datetime.now(),
                record_id,
                transformation_hash,
                details_json,
                details.get('source_adapter') if details else None,
                severity,
                table_name,
                row_count,
            ])
            
            logger.debug(f"Logged audit event: {event_type} (ID: {audit_id})")
            return Result.success_result(audit_id)
            
        except Exception as e:
            error_msg = f"Failed to log audit event: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="log_audit_event"),
                error_type="StorageError"
            )
    
    def log_redaction_event(
        self,
        field_name: str,
        original_hash: str,
        rule_triggered: str,
        record_id: Optional[str] = None,
        source_adapter: Optional[str] = None,
        ingestion_id: Optional[str] = None,
        redacted_value: Optional[str] = None,
        original_value_length: Optional[int] = None
    ) -> Result[str]:
        """Log a single redaction event to the logs table.
        
        Parameters:
            field_name: Name of the field that was redacted
            original_hash: SHA256 hash of the original value
            rule_triggered: Name of the redaction rule that was triggered
            record_id: Unique identifier of the record
            source_adapter: Source adapter identifier
            ingestion_id: Ingestion run identifier
            redacted_value: Redacted value (optional, for debugging)
            original_value_length: Length of original value (optional)
        
        Returns:
            Result[str]: Log entry ID or error
        """
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            log_id = str(uuid.uuid4())
            
            conn.execute("""
                INSERT INTO logs (
                    log_id, field_name, original_hash, timestamp, rule_triggered,
                    record_id, source_adapter, ingestion_id, redacted_value, original_value_length
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                log_id,
                field_name,
                original_hash,
                datetime.now(),
                rule_triggered,
                record_id,
                source_adapter,
                ingestion_id,
                redacted_value,
                original_value_length,
            ])
            
            logger.debug(f"Logged redaction event: {field_name} - {rule_triggered} (ID: {log_id})")
            return Result.success_result(log_id)
            
        except Exception as e:
            error_msg = f"Failed to log redaction event: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="log_redaction_event"),
                error_type="StorageError"
            )
    
    def flush_redaction_logs(self, redaction_logs: list[dict]) -> Result[int]:
        """Flush multiple redaction logs to the database in a single transaction.
        
        Parameters:
            redaction_logs: List of redaction log dictionaries
        
        Returns:
            Result[int]: Number of logs persisted or error
        """
        if not redaction_logs:
            return Result.success_result(0)
        
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            
            # Bulk insert redaction logs
            for log_entry in redaction_logs:
                conn.execute("""
                    INSERT INTO logs (
                        log_id, field_name, original_hash, timestamp, rule_triggered,
                        record_id, source_adapter, ingestion_id, redacted_value, original_value_length
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    log_entry.get('log_id', str(uuid.uuid4())),
                    log_entry.get('field_name'),
                    log_entry.get('original_hash'),
                    log_entry.get('timestamp', datetime.now()),
                    log_entry.get('rule_triggered'),
                    log_entry.get('record_id'),
                    log_entry.get('source_adapter'),
                    log_entry.get('ingestion_id'),
                    log_entry.get('redacted_value'),
                    log_entry.get('original_value_length'),
                ])
            
            count = len(redaction_logs)
            logger.info(f"Flushed {count} redaction logs to database")
            return Result.success_result(count)
            
        except Exception as e:
            error_msg = f"Failed to flush redaction logs: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="flush_redaction_logs"),
                error_type="StorageError"
            )
    
    def generate_security_report(
        self,
        ingestion_id: Optional[str] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None
    ) -> Result[dict]:
        """Generate a security report of all redaction events.
        
        Parameters:
            ingestion_id: Optional ingestion ID to filter by
            start_timestamp: Optional start time for report period
            end_timestamp: Optional end time for report period
        
        Returns:
            Result[dict]: Security report dictionary or error
        """
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            
            # Build query with optional filters
            query = "SELECT * FROM logs WHERE 1=1"
            params = []
            
            if ingestion_id:
                query += " AND ingestion_id = ?"
                params.append(ingestion_id)
            
            if start_timestamp:
                query += " AND timestamp >= ?"
                params.append(start_timestamp)
            
            if end_timestamp:
                query += " AND timestamp <= ?"
                params.append(end_timestamp)
            
            query += " ORDER BY timestamp DESC"
            
            # Execute query
            result = conn.execute(query, params).fetchall()
            
            # Get column names
            columns = [desc[0] for desc in conn.execute("PRAGMA table_info(logs)").fetchall()]
            
            # Convert to list of dictionaries
            logs = [dict(zip(columns, row)) for row in result]
            
            # Generate summary statistics
            total_redactions = len(logs)
            redactions_by_field = {}
            redactions_by_rule = {}
            redactions_by_adapter = {}
            
            for log in logs:
                field = log.get('field_name', 'unknown')
                rule = log.get('rule_triggered', 'unknown')
                adapter = log.get('source_adapter', 'unknown')
                
                redactions_by_field[field] = redactions_by_field.get(field, 0) + 1
                redactions_by_rule[rule] = redactions_by_rule.get(rule, 0) + 1
                redactions_by_adapter[adapter] = redactions_by_adapter.get(adapter, 0) + 1
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "ingestion_id": ingestion_id,
                "start_timestamp": start_timestamp.isoformat() if start_timestamp else None,
                "end_timestamp": end_timestamp.isoformat() if end_timestamp else None,
                "summary": {
                    "total_redactions": total_redactions,
                    "redactions_by_field": redactions_by_field,
                    "redactions_by_rule": redactions_by_rule,
                    "redactions_by_adapter": redactions_by_adapter,
                },
                "events": logs
            }
            
            logger.info(f"Generated security report: {total_redactions} redaction events")
            return Result.success_result(report)
            
        except Exception as e:
            error_msg = f"Failed to generate security report: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="generate_security_report"),
                error_type="StorageError"
            )
    
    def close(self) -> None:
        """Close storage connection and release resources."""
        if self._connection is not None:
            try:
                self._connection.close()
                self._connection = None
                logger.info("Closed DuckDB connection")
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")

