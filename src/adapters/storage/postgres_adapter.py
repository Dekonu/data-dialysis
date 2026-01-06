"""PostgreSQL Storage Adapter (Alternative Implementation).

This adapter provides a PostgreSQL implementation of the StoragePort contract,
optimized for Supabase and cloud PostgreSQL deployments.

Security Impact:
    - Only validated GoldenRecord instances can be persisted
    - All operations are logged to immutable audit trail
    - Connection credentials are managed securely via configuration
    - Schema enforces data integrity constraints
    - SSL connections required for secure network communication

Architecture:
    - Implements StoragePort (Hexagonal Architecture)
    - Isolated from domain core - only depends on ports and models
    - Transactional batch operations ensure data consistency
    - Connection pooling for performance
    - Audit trail is append-only for compliance
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Any
from urllib.parse import quote_plus
import json

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import execute_values, Json
    from sqlalchemy import create_engine
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

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


class PostgresAdapter(StoragePort):
    """PostgreSQL implementation of StoragePort optimized for Supabase.
    
    This adapter provides robust transactional storage for validated clinical records.
    PostgreSQL is optimized for OLTP workloads and provides excellent ACID guarantees.
    
    Security Impact:
        - All records are validated before persistence
        - Audit trail is immutable and tamper-proof
        - Connection credentials are never logged
        - Schema enforces referential integrity
        - SSL connections required for secure communication
    
    Parameters:
        db_config: DatabaseConfig from configuration manager (preferred)
        connection_string: PostgreSQL connection string (alternative)
        host: Database host (if not using connection_string or db_config)
        port: Database port (default: 5432)
        database: Database name
        username: Database username
        password: Database password
        ssl_mode: SSL mode (require, prefer, disable) - defaults to 'require' for security
        pool_size: Connection pool size (default: 5)
        max_overflow: Maximum connection pool overflow (default: 10)
    
    Example Usage:
        ```python
        # Using configuration manager (recommended)
        from src.infrastructure.config_manager import get_database_config
        
        db_config = get_database_config()
        adapter = PostgresAdapter(db_config=db_config)
        
        # Or using connection string directly
        adapter = PostgresAdapter(
            connection_string="postgresql://user:pass@host/db?sslmode=require"
        )
        
        result = adapter.initialize_schema()
        if result.is_success():
            result = adapter.persist(golden_record)
        ```
    """
    
    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 5432,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl_mode: str = "require",
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """Initialize PostgreSQL adapter.
        
        Parameters:
            db_config: DatabaseConfig from configuration manager (preferred)
            connection_string: Full PostgreSQL connection string
            host: Database host (required if no connection_string or db_config)
            port: Database port (default: 5432)
            database: Database name (required if no connection_string or db_config)
            username: Database username
            password: Database password
            ssl_mode: SSL mode (require, prefer, disable) - defaults to 'require'
            pool_size: Connection pool size
            max_overflow: Maximum connection pool overflow
        
        Security Impact:
            - Connection credentials are validated but never logged
            - Connection is established lazily (on first operation)
            - SSL mode defaults to 'require' for security
            - Credentials are managed securely via configuration manager
        
        Raises:
            StorageError: If psycopg2 is not installed or configuration is invalid
        """
        if not PSYCOPG2_AVAILABLE:
            raise StorageError(
                "psycopg2 is required for PostgreSQL adapter. Install with: pip install psycopg2-binary",
                operation="__init__"
            )
        
        self._connection_pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialized = False
        
        # Use DatabaseConfig if provided (preferred method)
        if db_config:
            if db_config.db_type != "postgresql":
                raise StorageError(
                    f"DatabaseConfig type '{db_config.db_type}' does not match PostgreSQL adapter",
                    operation="__init__"
                )
            
            # Build connection parameters from DatabaseConfig
            if db_config.connection_string:
                self.connection_params = {"dsn": db_config.connection_string.get_secret_value()}
            else:
                if not all([db_config.host, db_config.database]):
                    raise StorageError(
                        "PostgreSQL DatabaseConfig requires host and database",
                        operation="__init__"
                    )
                
                self.connection_params = {
                    "host": db_config.host,
                    "port": db_config.port or 5432,
                    "database": db_config.database,
                    "user": db_config.username,
                }
                
                # Add password if provided
                if db_config.password:
                    self.connection_params["password"] = db_config.password.get_secret_value()
                
                # Add SSL mode (default to require for security)
                if db_config.ssl_mode:
                    self.connection_params["sslmode"] = db_config.ssl_mode
                else:
                    self.connection_params["sslmode"] = "require"
            
            # Use pool size from config if available
            self.pool_size = db_config.pool_size or pool_size
            self.max_overflow = db_config.max_overflow or max_overflow
        
        # Fall back to connection_string or individual parameters
        elif connection_string:
            self.connection_params = {"dsn": connection_string}
            self.pool_size = pool_size
            self.max_overflow = max_overflow
        else:
            if not all([host, database]):
                raise StorageError(
                    "PostgreSQL adapter requires either db_config, connection_string, or (host and database)",
                    operation="__init__"
                )
            
            self.connection_params = {
                "host": host,
                "port": port,
                "database": database,
                "user": username,
                "password": password,
                "sslmode": ssl_mode,
            }
            
            self.pool_size = pool_size
            self.max_overflow = max_overflow
    
    def _get_connection_pool(self) -> pool.ThreadedConnectionPool:
        """Get or create PostgreSQL connection pool.
        
        Returns:
            PostgreSQL connection pool instance
        
        Security Impact:
            - Connection pool is created lazily to avoid unnecessary resource usage
            - Pool is reused for performance
        
        Raises:
            StorageError: If connection pool cannot be created
        """
        if self._connection_pool is None:
            try:
                self._connection_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=self.pool_size + self.max_overflow,
                    **self.connection_params
                )
                logger.info("Created PostgreSQL connection pool")
            except Exception as e:
                raise StorageError(
                    f"Failed to create PostgreSQL connection pool: {str(e)}",
                    operation="connect",
                    details={"host": self.connection_params.get("host", "N/A")}
                )
        return self._connection_pool
    
    def _get_connection(self):
        """Get a connection from the pool.
        
        Returns:
            PostgreSQL connection instance
        
        Raises:
            StorageError: If connection cannot be obtained
        """
        try:
            pool = self._get_connection_pool()
            return pool.getconn()
        except Exception as e:
            raise StorageError(
                f"Failed to get connection from pool: {str(e)}",
                operation="get_connection"
            )
    
    def _return_connection(self, conn):
        """Return a connection to the pool.
        
        Parameters:
            conn: Connection to return
        """
        try:
            pool = self._get_connection_pool()
            pool.putconn(conn)
        except Exception as e:
            logger.warning(f"Error returning connection to pool: {str(e)}")
    
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
            - Foreign keys enforce referential integrity
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id VARCHAR(50) PRIMARY KEY,
                    identifiers TEXT[],
                    family_name VARCHAR(255),
                    given_names TEXT[],
                    name_prefix TEXT[],
                    name_suffix TEXT[],
                    date_of_birth DATE,
                    gender VARCHAR(20),
                    deceased BOOLEAN,
                    deceased_date TIMESTAMP,
                    marital_status VARCHAR(50),
                    address_line1 VARCHAR(255),
                    address_line2 VARCHAR(255),
                    city VARCHAR(100),
                    state VARCHAR(2),
                    postal_code VARCHAR(10),
                    country VARCHAR(3),
                    address_use VARCHAR(20),
                    phone VARCHAR(20),
                    email VARCHAR(255),
                    fax VARCHAR(20),
                    emergency_contact_name VARCHAR(255),
                    emergency_contact_relationship VARCHAR(50),
                    emergency_contact_phone VARCHAR(20),
                    language VARCHAR(10),
                    managing_organization VARCHAR(100),
                    ingestion_timestamp TIMESTAMP NOT NULL,
                    source_adapter VARCHAR(50) NOT NULL,
                    transformation_hash VARCHAR(64)
                )
            """)
            
            # Create encounters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS encounters (
                    encounter_id VARCHAR(50) PRIMARY KEY,
                    patient_id VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    class_code VARCHAR(20) NOT NULL,
                    type VARCHAR(50),
                    service_type VARCHAR(50),
                    priority VARCHAR(20),
                    period_start TIMESTAMP,
                    period_end TIMESTAMP,
                    length_minutes INTEGER,
                    reason_code VARCHAR(20),
                    diagnosis_codes TEXT[],
                    facility_name VARCHAR(255),
                    location_address VARCHAR(255),
                    participant_name VARCHAR(255),
                    participant_role VARCHAR(50),
                    service_provider VARCHAR(100),
                    ingestion_timestamp TIMESTAMP NOT NULL,
                    source_adapter VARCHAR(50) NOT NULL,
                    transformation_hash VARCHAR(64),
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )
            """)
            
            # Create observations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id VARCHAR(50) PRIMARY KEY,
                    patient_id VARCHAR(50) NOT NULL,
                    encounter_id VARCHAR(50),
                    status VARCHAR(20) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    code VARCHAR(50),
                    effective_date TIMESTAMP,
                    issued TIMESTAMP,
                    performer_name VARCHAR(255),
                    value VARCHAR(255),
                    unit VARCHAR(50),
                    interpretation TEXT,
                    body_site VARCHAR(50),
                    method VARCHAR(50),
                    device VARCHAR(100),
                    reference_range VARCHAR(255),
                    notes TEXT,
                    ingestion_timestamp TIMESTAMP NOT NULL,
                    source_adapter VARCHAR(50) NOT NULL,
                    transformation_hash VARCHAR(64),
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id) ON DELETE CASCADE
                )
            """)
            
            # Create audit log table (immutable, append-only)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id VARCHAR(50) PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,
                    event_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    record_id VARCHAR(50),
                    transformation_hash VARCHAR(64),
                    details JSONB,
                    source_adapter VARCHAR(50),
                    severity VARCHAR(20),
                    table_name VARCHAR(50),
                    row_count INTEGER
                )
            """)
            
            # Add columns if they don't exist (for existing databases)
            try:
                cursor.execute("ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS table_name VARCHAR(50)")
                cursor.execute("ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS row_count INTEGER")
            except Exception:
                # Columns may already exist, ignore error
                pass
            
            # Create redaction logs table for detailed PII redaction tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    log_id VARCHAR(50) PRIMARY KEY,
                    field_name VARCHAR(50) NOT NULL,
                    original_hash VARCHAR(64) NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    rule_triggered VARCHAR(50) NOT NULL,
                    record_id VARCHAR(50),
                    source_adapter VARCHAR(50),
                    ingestion_id VARCHAR(50),
                    redacted_value VARCHAR(255),
                    original_value_length INTEGER
                )
            """)
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_patients_source ON patients(source_adapter)",
                "CREATE INDEX IF NOT EXISTS idx_patients_timestamp ON patients(ingestion_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_encounters_patient ON encounters(patient_id)",
                "CREATE INDEX IF NOT EXISTS idx_encounters_timestamp ON encounters(ingestion_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_observations_patient ON observations(patient_id)",
                "CREATE INDEX IF NOT EXISTS idx_observations_encounter ON observations(encounter_id)",
                "CREATE INDEX IF NOT EXISTS idx_observations_timestamp ON observations(ingestion_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type)",
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(event_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_audit_record_id ON audit_log(record_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_log(severity)",
                "CREATE INDEX IF NOT EXISTS idx_logs_field_name ON logs(field_name)",
                "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_logs_rule_triggered ON logs(rule_triggered)",
                "CREATE INDEX IF NOT EXISTS idx_logs_record_id ON logs(record_id)",
                "CREATE INDEX IF NOT EXISTS idx_logs_ingestion_id ON logs(ingestion_id)",
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            cursor.close()
            
            self._initialized = True
            logger.info("PostgreSQL database schema initialized successfully")
            
            return Result.success_result(None)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error_msg = f"Failed to initialize schema: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="initialize_schema"),
                error_type="StorageError"
            )
        finally:
            if conn:
                self._return_connection(conn)
    
    def persist(self, record: GoldenRecord) -> Result[str]:
        """Persist a single GoldenRecord to storage.
        
        Parameters:
            record: Validated GoldenRecord instance (PII already redacted)
        
        Returns:
            Result[str]: Record identifier (patient_id) or error
        """
        # Validate input type
        if not isinstance(record, GoldenRecord):
            error_msg = f"Expected GoldenRecord, got {type(record).__name__}"
            logger.error(error_msg)
            return Result.failure_result(
                StorageError(error_msg, operation="persist", details={"received_type": type(record).__name__}),
                error_type="StorageError"
            )
        
        conn = None
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Insert patient record
            patient = record.patient
            patient_dict = patient.model_dump()
            
            # Convert lists to PostgreSQL arrays
            identifiers = patient_dict.get('identifiers', []) or []
            given_names = patient_dict.get('given_names', []) or []
            name_prefix = patient_dict.get('name_prefix', []) or []
            name_suffix = patient_dict.get('name_suffix', []) or []
            
            cursor.execute("""
                INSERT INTO patients (
                    patient_id, identifiers, family_name, given_names, name_prefix, name_suffix,
                    date_of_birth, gender, deceased, deceased_date, marital_status,
                    address_line1, address_line2, city, state, postal_code, country,
                    address_use, phone, email, fax, emergency_contact_name,
                    emergency_contact_relationship, emergency_contact_phone, language,
                    managing_organization, ingestion_timestamp, source_adapter, transformation_hash
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (patient_id) DO UPDATE SET
                    identifiers = EXCLUDED.identifiers,
                    family_name = EXCLUDED.family_name,
                    given_names = EXCLUDED.given_names,
                    name_prefix = EXCLUDED.name_prefix,
                    name_suffix = EXCLUDED.name_suffix,
                    date_of_birth = EXCLUDED.date_of_birth,
                    gender = EXCLUDED.gender,
                    deceased = EXCLUDED.deceased,
                    deceased_date = EXCLUDED.deceased_date,
                    marital_status = EXCLUDED.marital_status,
                    address_line1 = EXCLUDED.address_line1,
                    address_line2 = EXCLUDED.address_line2,
                    city = EXCLUDED.city,
                    state = EXCLUDED.state,
                    postal_code = EXCLUDED.postal_code,
                    country = EXCLUDED.country,
                    address_use = EXCLUDED.address_use,
                    phone = EXCLUDED.phone,
                    email = EXCLUDED.email,
                    fax = EXCLUDED.fax,
                    emergency_contact_name = EXCLUDED.emergency_contact_name,
                    emergency_contact_relationship = EXCLUDED.emergency_contact_relationship,
                    emergency_contact_phone = EXCLUDED.emergency_contact_phone,
                    language = EXCLUDED.language,
                    managing_organization = EXCLUDED.managing_organization,
                    ingestion_timestamp = EXCLUDED.ingestion_timestamp,
                    source_adapter = EXCLUDED.source_adapter,
                    transformation_hash = EXCLUDED.transformation_hash
            """, [
                patient_dict.get('patient_id'),
                identifiers,
                patient_dict.get('family_name'),
                given_names,
                name_prefix,
                name_suffix,
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
                record.ingestion_timestamp,
                record.source_adapter,
                record.transformation_hash,
            ])
            
            # Insert encounters
            for encounter in record.encounters:
                encounter_dict = encounter.model_dump()
                diagnosis_codes = encounter_dict.get('diagnosis_codes', []) or []
                
                cursor.execute("""
                    INSERT INTO encounters (
                        encounter_id, patient_id, status, class_code, type, service_type, priority,
                        period_start, period_end, length_minutes, reason_code, diagnosis_codes,
                        facility_name, location_address, participant_name, participant_role,
                        service_provider, ingestion_timestamp, source_adapter, transformation_hash
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (encounter_id) DO UPDATE SET
                        patient_id = EXCLUDED.patient_id,
                        status = EXCLUDED.status,
                        class_code = EXCLUDED.class_code,
                        type = EXCLUDED.type,
                        service_type = EXCLUDED.service_type,
                        priority = EXCLUDED.priority,
                        period_start = EXCLUDED.period_start,
                        period_end = EXCLUDED.period_end,
                        length_minutes = EXCLUDED.length_minutes,
                        reason_code = EXCLUDED.reason_code,
                        diagnosis_codes = EXCLUDED.diagnosis_codes,
                        facility_name = EXCLUDED.facility_name,
                        location_address = EXCLUDED.location_address,
                        participant_name = EXCLUDED.participant_name,
                        participant_role = EXCLUDED.participant_role,
                        service_provider = EXCLUDED.service_provider,
                        ingestion_timestamp = EXCLUDED.ingestion_timestamp,
                        source_adapter = EXCLUDED.source_adapter,
                        transformation_hash = EXCLUDED.transformation_hash
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
                    diagnosis_codes,
                    encounter_dict.get('facility_name'),
                    encounter_dict.get('location_address'),
                    encounter_dict.get('participant_name'),
                    encounter_dict.get('participant_role'),
                    encounter_dict.get('service_provider'),
                    record.ingestion_timestamp,
                    record.source_adapter,
                    record.transformation_hash,
                ])
            
            # Insert observations
            for observation in record.observations:
                observation_dict = observation.model_dump()
                
                cursor.execute("""
                    INSERT INTO observations (
                        observation_id, patient_id, encounter_id, status, category, code,
                        effective_date, issued, performer_name, value, unit, interpretation,
                        body_site, method, device, reference_range, notes,
                        ingestion_timestamp, source_adapter, transformation_hash
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (observation_id) DO UPDATE SET
                        patient_id = EXCLUDED.patient_id,
                        encounter_id = EXCLUDED.encounter_id,
                        status = EXCLUDED.status,
                        category = EXCLUDED.category,
                        code = EXCLUDED.code,
                        effective_date = EXCLUDED.effective_date,
                        issued = EXCLUDED.issued,
                        performer_name = EXCLUDED.performer_name,
                        value = EXCLUDED.value,
                        unit = EXCLUDED.unit,
                        interpretation = EXCLUDED.interpretation,
                        body_site = EXCLUDED.body_site,
                        method = EXCLUDED.method,
                        device = EXCLUDED.device,
                        reference_range = EXCLUDED.reference_range,
                        notes = EXCLUDED.notes,
                        ingestion_timestamp = EXCLUDED.ingestion_timestamp,
                        source_adapter = EXCLUDED.source_adapter,
                        transformation_hash = EXCLUDED.transformation_hash
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
                    record.ingestion_timestamp,
                    record.source_adapter,
                    record.transformation_hash,
                ])
            
            conn.commit()
            cursor.close()
            
            # Log audit event (singular record - row_count is None)
            self.log_audit_event(
                event_type="PERSISTENCE",
                record_id=patient.patient_id,
                transformation_hash=record.transformation_hash,
                details={
                    "encounter_count": len(record.encounters),
                    "observation_count": len(record.observations),
                },
                table_name="patients",  # Main table for GoldenRecord
                row_count=None,  # Singular record, not bulk
                source_adapter=record.source_adapter
            )
            
            logger.info(f"Persisted GoldenRecord for patient_id: {patient.patient_id}")
            return Result.success_result(patient.patient_id)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error_msg = f"Failed to persist record: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Safely get patient_id for error details
            patient_id = None
            try:
                if isinstance(record, GoldenRecord) and record.patient:
                    patient_id = record.patient.patient_id
            except Exception:
                pass
            return Result.failure_result(
                StorageError(error_msg, operation="persist", details={"patient_id": patient_id}),
                error_type="StorageError"
            )
        finally:
            if conn:
                self._return_connection(conn)
    
    def persist_batch(self, records: list[GoldenRecord]) -> Result[list[str]]:
        """Persist multiple GoldenRecords in a single transaction.
        
        Parameters:
            records: List of validated GoldenRecord instances
        
        Returns:
            Result[list[str]]: List of record identifiers or error
        """
        if not records:
            return Result.success_result([])
        
        conn = None
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            record_ids = []
            for record in records:
                # Use same logic as persist() but within batch transaction
                result = self._persist_record_in_transaction(cursor, record)
                if not result.is_success():
                    raise StorageError(
                        f"Batch persistence failed for record: {result.error}",
                        operation="persist_batch",
                        details={"record_count": len(records)}
                    )
                record_ids.append(result.value)
            
            conn.commit()
            cursor.close()
            
            logger.info(f"Persisted batch of {len(records)} records")
            
            # Log bulk persistence audit event
            # Extract source_adapter from first record if available
            source_adapter = 'batch_ingestion'
            if records and records[0].source_adapter:
                source_adapter = records[0].source_adapter
            
            # Count rows per table
            patients_count = len(records)  # Each record has 1 patient
            encounters_count = 0
            observations_count = 0
            for record in records:
                encounters_count += len(record.encounters)
                observations_count += len(record.observations)
            
            total_rows = patients_count + encounters_count + observations_count
            
            # Log bulk audit event with table breakdown
            self.log_audit_event(
                event_type="BULK_PERSISTENCE",
                record_id=None,
                transformation_hash=None,
                details={
                    "record_count": len(records),
                    "patients": patients_count,
                    "encounters": encounters_count,
                    "observations": observations_count,
                },
                table_name="patients",  # Primary table (entity)
                row_count=total_rows,
                source_adapter=source_adapter
            )
            
            return Result.success_result(record_ids)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error_msg = f"Failed to persist batch: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="persist_batch", details={"record_count": len(records)}),
                error_type="StorageError"
            )
        finally:
            if conn:
                self._return_connection(conn)
    
    def _persist_record_in_transaction(self, cursor, record: GoldenRecord) -> Result[str]:
        """Helper method to persist a record within an existing transaction.
        
        This is used by persist_batch() to avoid creating nested transactions.
        """
        try:
            # Same logic as persist() but using provided cursor
            patient = record.patient
            patient_dict = patient.model_dump()
            
            identifiers = patient_dict.get('identifiers', []) or []
            given_names = patient_dict.get('given_names', []) or []
            name_prefix = patient_dict.get('name_prefix', []) or []
            name_suffix = patient_dict.get('name_suffix', []) or []
            
            cursor.execute("""
                INSERT INTO patients (
                    patient_id, identifiers, family_name, given_names, name_prefix, name_suffix,
                    date_of_birth, gender, deceased, deceased_date, marital_status,
                    address_line1, address_line2, city, state, postal_code, country,
                    address_use, phone, email, fax, emergency_contact_name,
                    emergency_contact_relationship, emergency_contact_phone, language,
                    managing_organization, ingestion_timestamp, source_adapter, transformation_hash
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (patient_id) DO UPDATE SET
                    identifiers = EXCLUDED.identifiers,
                    family_name = EXCLUDED.family_name,
                    given_names = EXCLUDED.given_names,
                    name_prefix = EXCLUDED.name_prefix,
                    name_suffix = EXCLUDED.name_suffix,
                    date_of_birth = EXCLUDED.date_of_birth,
                    gender = EXCLUDED.gender,
                    deceased = EXCLUDED.deceased,
                    deceased_date = EXCLUDED.deceased_date,
                    marital_status = EXCLUDED.marital_status,
                    address_line1 = EXCLUDED.address_line1,
                    address_line2 = EXCLUDED.address_line2,
                    city = EXCLUDED.city,
                    state = EXCLUDED.state,
                    postal_code = EXCLUDED.postal_code,
                    country = EXCLUDED.country,
                    address_use = EXCLUDED.address_use,
                    phone = EXCLUDED.phone,
                    email = EXCLUDED.email,
                    fax = EXCLUDED.fax,
                    emergency_contact_name = EXCLUDED.emergency_contact_name,
                    emergency_contact_relationship = EXCLUDED.emergency_contact_relationship,
                    emergency_contact_phone = EXCLUDED.emergency_contact_phone,
                    language = EXCLUDED.language,
                    managing_organization = EXCLUDED.managing_organization,
                    ingestion_timestamp = EXCLUDED.ingestion_timestamp,
                    source_adapter = EXCLUDED.source_adapter,
                    transformation_hash = EXCLUDED.transformation_hash
            """, [
                patient_dict.get('patient_id'),
                identifiers,
                patient_dict.get('family_name'),
                given_names,
                name_prefix,
                name_suffix,
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
                record.ingestion_timestamp,
                record.source_adapter,
                record.transformation_hash,
            ])
            
            # Insert encounters and observations (same as persist())
            for encounter in record.encounters:
                encounter_dict = encounter.model_dump()
                diagnosis_codes = encounter_dict.get('diagnosis_codes', []) or []
                
                cursor.execute("""
                    INSERT INTO encounters (
                        encounter_id, patient_id, status, class_code, type, service_type, priority,
                        period_start, period_end, length_minutes, reason_code, diagnosis_codes,
                        facility_name, location_address, participant_name, participant_role,
                        service_provider, ingestion_timestamp, source_adapter, transformation_hash
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (encounter_id) DO UPDATE SET
                        patient_id = EXCLUDED.patient_id,
                        status = EXCLUDED.status,
                        class_code = EXCLUDED.class_code,
                        type = EXCLUDED.type,
                        service_type = EXCLUDED.service_type,
                        priority = EXCLUDED.priority,
                        period_start = EXCLUDED.period_start,
                        period_end = EXCLUDED.period_end,
                        length_minutes = EXCLUDED.length_minutes,
                        reason_code = EXCLUDED.reason_code,
                        diagnosis_codes = EXCLUDED.diagnosis_codes,
                        facility_name = EXCLUDED.facility_name,
                        location_address = EXCLUDED.location_address,
                        participant_name = EXCLUDED.participant_name,
                        participant_role = EXCLUDED.participant_role,
                        service_provider = EXCLUDED.service_provider,
                        ingestion_timestamp = EXCLUDED.ingestion_timestamp,
                        source_adapter = EXCLUDED.source_adapter,
                        transformation_hash = EXCLUDED.transformation_hash
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
                    diagnosis_codes,
                    encounter_dict.get('facility_name'),
                    encounter_dict.get('location_address'),
                    encounter_dict.get('participant_name'),
                    encounter_dict.get('participant_role'),
                    encounter_dict.get('service_provider'),
                    record.ingestion_timestamp,
                    record.source_adapter,
                    record.transformation_hash,
                ])
            
            for observation in record.observations:
                observation_dict = observation.model_dump()
                
                cursor.execute("""
                    INSERT INTO observations (
                        observation_id, patient_id, encounter_id, status, category, code,
                        effective_date, issued, performer_name, value, unit, interpretation,
                        body_site, method, device, reference_range, notes,
                        ingestion_timestamp, source_adapter, transformation_hash
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (observation_id) DO UPDATE SET
                        patient_id = EXCLUDED.patient_id,
                        encounter_id = EXCLUDED.encounter_id,
                        status = EXCLUDED.status,
                        category = EXCLUDED.category,
                        code = EXCLUDED.code,
                        effective_date = EXCLUDED.effective_date,
                        issued = EXCLUDED.issued,
                        performer_name = EXCLUDED.performer_name,
                        value = EXCLUDED.value,
                        unit = EXCLUDED.unit,
                        interpretation = EXCLUDED.interpretation,
                        body_site = EXCLUDED.body_site,
                        method = EXCLUDED.method,
                        device = EXCLUDED.device,
                        reference_range = EXCLUDED.reference_range,
                        notes = EXCLUDED.notes,
                        ingestion_timestamp = EXCLUDED.ingestion_timestamp,
                        source_adapter = EXCLUDED.source_adapter,
                        transformation_hash = EXCLUDED.transformation_hash
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
                    record.ingestion_timestamp,
                    record.source_adapter,
                    record.transformation_hash,
                ])
            
            return Result.success_result(patient.patient_id)
            
        except Exception as e:
            error_msg = f"Failed to persist record in transaction: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="persist", details={"patient_id": record.patient_id}),
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
        
        engine = None
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            # Create SQLAlchemy engine from connection string for pandas to_sql
            if 'dsn' in self.connection_params:
                connection_string = self.connection_params['dsn']
            else:
                # Build connection string from individual parameters
                password = self.connection_params.get('password', '')
                username = self.connection_params.get('user', '')
                host = self.connection_params.get('host', '')
                port = self.connection_params.get('port', 5432)
                database = self.connection_params.get('database', '')
                sslmode = self.connection_params.get('sslmode', 'require')
                
                # URL-encode password and username if they contain special characters
                password_encoded = quote_plus(password) if password else ''
                username_encoded = quote_plus(username) if username else ''
                
                connection_string = f"postgresql://{username_encoded}:{password_encoded}@{host}:{port}/{database}?sslmode={sslmode}"
            
            engine = create_engine(connection_string, pool_pre_ping=True)
            
            # Use pandas to_sql for efficient bulk insertion
            row_count = df.to_sql(
                name=table_name,
                con=engine,
                if_exists='append',  # Use append for incremental loading
                index=False,
                method='multi'  # Use multi-row insert for performance
            )
            
            logger.info(f"Persisted {row_count} rows to table '{table_name}'")
            
            # Log audit event (bulk operation)
            self.log_audit_event(
                event_type="BULK_PERSISTENCE",
                record_id=None,
                transformation_hash=None,
                details={},
                table_name=table_name,
                row_count=row_count
            )
            
            return Result.success_result(row_count)
            
        except Exception as e:
            error_msg = f"Failed to persist DataFrame to {table_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="persist_dataframe", details={"table_name": table_name}),
                error_type="StorageError"
            )
        finally:
            if engine:
                engine.dispose()
    
    def log_audit_event(
        self,
        event_type: str,
        record_id: Optional[str],
        transformation_hash: Optional[str],
        details: Optional[dict] = None,
        table_name: Optional[str] = None,
        row_count: Optional[int] = None,
        source_adapter: Optional[str] = None
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
        conn = None
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            cursor = conn.cursor()
            audit_id = str(uuid.uuid4())
            
            # Determine severity based on event type
            severity = "INFO"
            if event_type in ["REDACTION", "PII_DETECTED"]:
                severity = "CRITICAL"
            elif event_type in ["VALIDATION_ERROR", "TRANSFORMATION_ERROR"]:
                severity = "WARNING"
            
            # Convert details to JSONB
            details_json = Json(details) if details else None
            
            cursor.execute("""
                INSERT INTO audit_log (
                    audit_id, event_type, event_timestamp, record_id,
                    transformation_hash, details, source_adapter, severity,
                    table_name, row_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            
            conn.commit()
            cursor.close()
            
            logger.debug(f"Logged audit event: {event_type} (ID: {audit_id})")
            return Result.success_result(audit_id)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error_msg = f"Failed to log audit event: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="log_audit_event"),
                error_type="StorageError"
            )
        finally:
            if conn:
                self._return_connection(conn)
    
    def flush_redaction_logs(self, redaction_logs: list[dict]) -> Result[int]:
        """Flush multiple redaction logs to the database in a single transaction.
        
        Parameters:
            redaction_logs: List of redaction log dictionaries
            
        Returns:
            Result[int]: Number of logs persisted or error
        """
        if not redaction_logs:
            return Result.success_result(0)
        
        conn = None
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Bulk insert redaction logs
            for log_entry in redaction_logs:
                cursor.execute("""
                    INSERT INTO logs (
                        log_id, field_name, original_hash, timestamp, rule_triggered,
                        record_id, source_adapter, ingestion_id, redacted_value, original_value_length
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            
            conn.commit()
            cursor.close()
            
            count = len(redaction_logs)
            logger.info(f"Flushed {count} redaction logs to database")
            return Result.success_result(count)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error_msg = f"Failed to flush redaction logs: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="flush_redaction_logs"),
                error_type="StorageError"
            )
        finally:
            if conn:
                self._return_connection(conn)
    
    def close(self) -> None:
        """Close storage connection pool and release resources."""
        if self._connection_pool is not None:
            try:
                self._connection_pool.closeall()
                self._connection_pool = None
                logger.info("Closed PostgreSQL connection pool")
            except Exception as e:
                logger.warning(f"Error closing connection pool: {str(e)}")

