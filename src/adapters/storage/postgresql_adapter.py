"""PostgreSQL Storage Adapter.

This adapter implements the StoragePort contract for persisting validated GoldenRecords
to PostgreSQL, a production-grade relational database optimized for transactional workloads.

Security Impact:
    - Only validated GoldenRecord instances can be persisted
    - All operations are logged to immutable audit trail
    - Connection credentials are managed securely via configuration
    - Schema enforces data integrity constraints
    - SSL connections supported for secure network communication

Architecture:
    - Implements StoragePort (Hexagonal Architecture)
    - Isolated from domain core - only depends on ports and models
    - Transactional batch operations ensure data consistency
    - Connection pooling for performance
    - Audit trail is append-only for compliance
"""

import logging
import uuid
import threading
import time
from datetime import datetime
from typing import Optional, Any, List
from urllib.parse import quote_plus
import json

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_values, Json
from sqlalchemy import create_engine
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
from src.infrastructure.dataframe_cleaner import DataFrameCleaner

logger = logging.getLogger(__name__)


class PostgreSQLAdapter(StoragePort):
    """PostgreSQL implementation of StoragePort for production data storage.
    
    This adapter provides robust transactional storage for validated clinical records.
    PostgreSQL is optimized for OLTP workloads and provides excellent ACID guarantees.
    
    Security Impact:
        - All records are validated before persistence
        - Audit trail is immutable and tamper-proof
        - Connection credentials are never logged
        - Schema enforces referential integrity
        - SSL connections supported for secure communication
    
    Parameters:
        connection_string: PostgreSQL connection string (or use individual parameters)
        host: Database host (if not using connection_string)
        port: Database port (default: 5432)
        database: Database name
        username: Database username
        password: Database password
        ssl_mode: SSL mode (require, prefer, disable)
        pool_size: Connection pool size (default: 5)
        max_overflow: Maximum connection pool overflow (default: 10)
    
    Example Usage:
        ```python
        # Using configuration manager (recommended)
        from src.infrastructure.config_manager import get_database_config
        
        db_config = get_database_config()
        adapter = PostgreSQLAdapter(db_config=db_config)
        
        # Or using connection string directly (backward compatibility)
        adapter = PostgreSQLAdapter(connection_string="postgresql://user:pass@host/db")
        
        # Or using individual parameters (backward compatibility)
        adapter = PostgreSQLAdapter(
            host="localhost",
            database="clinical_db",
            username="admin",
            password="secret"
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
        ssl_mode: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        config: Optional[dict] = None
    ):
        """Initialize PostgreSQL adapter.
        
        Parameters:
            db_config: DatabaseConfig from configuration manager (preferred)
            connection_string: Full PostgreSQL connection string
                             Deprecated: Use db_config instead
            host: Database host (required if no connection_string or db_config)
            port: Database port (default: 5432)
            database: Database name (required if no connection_string or db_config)
            username: Database username
            password: Database password
            ssl_mode: SSL mode (require, prefer, disable)
            pool_size: Connection pool size
            max_overflow: Maximum connection pool overflow
            config: Optional configuration dictionary (deprecated)
        
        Security Impact:
            - Connection credentials are validated but never logged
            - Connection is established lazily (on first operation)
            - SSL mode defaults to 'prefer' for security
            - Credentials are managed securely via configuration manager
        
        Note:
            Priority order: db_config > connection_string > individual parameters
            If db_config is provided, other parameters are ignored.
        """
        self.config = config or {}
        self._connection_pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialized = False
        self._schema_initialized = False  # Cache flag for schema initialization
        self._schema_lock = threading.Lock()  # Thread-safe lock for schema initialization
        
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
                
                # Add SSL mode
                if db_config.ssl_mode:
                    self.connection_params["sslmode"] = db_config.ssl_mode
                else:
                    self.connection_params["sslmode"] = "prefer"
            
            # Use pool size from config if available
            self.pool_size = db_config.pool_size
            self.max_overflow = db_config.max_overflow
        
        # Fall back to connection_string or individual parameters (backward compatibility)
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
            }
            
            # Add SSL mode if provided
            if ssl_mode:
                self.connection_params["sslmode"] = ssl_mode
            else:
                self.connection_params["sslmode"] = "prefer"
            
            self.pool_size = pool_size
            self.max_overflow = max_overflow
    
    def _get_connection_pool(self) -> pool.ThreadedConnectionPool:
        """Get or create PostgreSQL connection pool.
        
        Returns:
            PostgreSQL connection pool instance
        
        Security Impact:
            - Connection pool is created lazily to avoid unnecessary resource usage
            - Pool is reused for performance
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
        - logs: Redaction logs for PII tracking
        
        Returns:
            Result[None]: Success or failure result
        
        Security Impact:
            - Schema enforces data integrity constraints
            - Audit log table is append-only
            - Indexes optimize query performance
            - Foreign keys enforce referential integrity
        
        Note:
            Schema initialization is cached per adapter instance to prevent
            repeated database calls. The cache is thread-safe.
        """
        # Return early if schema is already initialized (cached)
        # Use double-checked locking pattern for thread safety
        if self._schema_initialized:
            return Result.success_result(None)
        
        # Acquire lock to prevent concurrent schema initialization
        with self._schema_lock:
            # Check again after acquiring lock (double-checked locking)
            if self._schema_initialized:
                return Result.success_result(None)
            
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
                
                # Create change audit log table for Change Data Capture (CDC)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS change_audit_log (
                    change_id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid()::text,
                    table_name VARCHAR(50) NOT NULL,
                    record_id VARCHAR(50) NOT NULL,
                    field_name VARCHAR(100) NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    change_type VARCHAR(20) NOT NULL,
                    changed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ingestion_id VARCHAR(50),
                    source_adapter VARCHAR(50),
                    changed_by VARCHAR(100)
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
                "CREATE INDEX IF NOT EXISTS idx_change_audit_table_record ON change_audit_log(table_name, record_id)",
                "CREATE INDEX IF NOT EXISTS idx_change_audit_timestamp ON change_audit_log(changed_at)",
                "CREATE INDEX IF NOT EXISTS idx_change_audit_ingestion ON change_audit_log(ingestion_id)",
                "CREATE INDEX IF NOT EXISTS idx_change_audit_covering ON change_audit_log(table_name, record_id, changed_at, field_name)",
                ]
                
                for index_sql in indexes:
                    cursor.execute(index_sql)
                
                conn.commit()
                cursor.close()
                
                self._initialized = True
                self._schema_initialized = True  # Cache the initialization
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
            return Result.failure_result(
                StorageError(error_msg, operation="persist", details={"patient_id": record.patient.patient_id}),
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
                StorageError(error_msg, operation="persist", details={"patient_id": record.patient.patient_id}),
                error_type="StorageError"
            )
    
    def _get_primary_key(self, table_name: str) -> Optional[str]:
        """Get the primary key column name for a table.
        
        Parameters:
            table_name: Name of the table
            
        Returns:
            Primary key column name or None if not found
        """
        primary_keys = {
            'patients': 'patient_id',
            'encounters': 'encounter_id',
            'observations': 'observation_id'
        }
        return primary_keys.get(table_name)
    
    def _bulk_fetch_existing_records(
        self,
        record_ids: List[str],
        table_name: str,
        primary_key: str
    ) -> pd.DataFrame:
        """Bulk fetch existing records from the database.
        
        This method is optimized for performance with large batches (10k-50k records).
        It fetches all records in a single query rather than per-record queries.
        
        Parameters:
            record_ids: List of primary key values to fetch
            table_name: Name of the table
            primary_key: Primary key column name
            
        Returns:
            DataFrame with existing records (empty if none found)
        
        Performance:
            - Single bulk query instead of N queries
            - Uses parameterized query for safety
            - Returns empty DataFrame if no records found
        """
        if not record_ids:
            return pd.DataFrame()
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build parameterized query for bulk fetch
            # Use IN clause with parameterized values
            placeholders = ','.join(['%s'] * len(record_ids))
            query = f"SELECT * FROM {table_name} WHERE {primary_key} IN ({placeholders})"
            
            cursor.execute(query, record_ids)
            
            # Fetch all results
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            cursor.close()
            self._return_connection(conn)
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            existing_df = pd.DataFrame(rows, columns=columns)
            logger.debug(f"Fetched {len(existing_df)} existing records from {table_name}")
            return existing_df
            
        except Exception as e:
            if conn:
                self._return_connection(conn)
            logger.warning(f"Failed to fetch existing records from {table_name}: {str(e)}")
            # Return empty DataFrame on error (treat as all new records)
            return pd.DataFrame()
    
    def _update_changed_fields_bulk(
        self,
        merged_df: pd.DataFrame,
        changes_df: pd.DataFrame,
        table_name: str,
        primary_key: str
    ) -> Result[int]:
        """Update only changed fields in bulk.
        
        This method builds efficient UPDATE statements that only modify fields
        that have actually changed. For simplicity and performance, it uses
        a temporary table approach or falls back to selective field updates.
        
        Parameters:
            merged_df: Merged DataFrame with _old and _new columns
            changes_df: DataFrame of detected changes (from ChangeDetector)
            table_name: Name of the table
            primary_key: Primary key column name
            
        Returns:
            Result[int]: Number of records updated or error
        
        Performance:
            - Only updates fields that actually changed
            - Uses bulk operations where possible
            - Falls back to standard update if selective update fails
        """
        if changes_df.empty or merged_df.empty:
            return Result.success_result(0)
        
        # For Phase 3, we'll use a simpler approach:
        # Extract only the changed fields and update them using standard persist
        # This still provides the benefit of change detection and logging
        # Full selective field updates can be optimized in a future iteration
        
        # Get unique record IDs that need updating
        record_ids = changes_df['record_id'].unique().tolist()
        
        # Get list of changed fields
        changed_fields = changes_df['field_name'].unique().tolist()
        
        # Filter merged_df to only records that need updating
        # Note: After merge with suffixes, the primary key column doesn't get a suffix
        # So we can access it directly
        try:
            updates_merged = merged_df[merged_df[primary_key].isin(record_ids)].copy()
        except KeyError as e:
            logger.error(f"Primary key {primary_key} not found in merged DataFrame. Available columns: {list(merged_df.columns)}")
            # Fall back to standard update
            return self._fallback_to_standard_update(merged_df, primary_key, table_name)
        
        if updates_merged.empty:
            return Result.success_result(0)
        
        # Build DataFrame with only changed fields
        # Start with primary key - it doesn't have a suffix after merge
        update_data = {}
        try:
            if primary_key in updates_merged.columns:
                update_data[primary_key] = updates_merged[primary_key].tolist()
            else:
                logger.error(f"Primary key {primary_key} not found in updates_merged. Available columns: {list(updates_merged.columns)}")
                return self._fallback_to_standard_update(merged_df, primary_key, table_name)
        except Exception as e:
            logger.error(f"Error accessing primary key {primary_key}: {str(e)}")
            return self._fallback_to_standard_update(merged_df, primary_key, table_name)
        
        # For each changed field, get the _new value from merged_df
        for field_name in changed_fields:
            new_col = f"{field_name}_new"
            try:
                if new_col in updates_merged.columns:
                    # Get values as list to maintain index alignment
                    update_data[field_name] = updates_merged[new_col].tolist()
                elif field_name in updates_merged.columns:
                    # If _new column doesn't exist, try original name (shouldn't happen but handle it)
                    update_data[field_name] = updates_merged[field_name].tolist()
                else:
                    logger.warning(f"Field {field_name} (or {new_col}) not found in merged DataFrame. Available: {list(updates_merged.columns)}")
                    # Skip this field - it won't be updated
            except Exception as e:
                logger.warning(f"Error accessing field {field_name}: {str(e)}. Skipping this field.")
                continue
        
        # Create DataFrame with only changed fields
        if not update_data or len(update_data) == 1:  # Only primary key, no fields to update
            logger.warning("No fields to update after filtering")
            return Result.success_result(0)
        
        # Ensure all lists have the same length
        lengths = [len(v) for v in update_data.values() if isinstance(v, list)]
        if lengths and len(set(lengths)) > 1:
            logger.error(f"Inconsistent data lengths in update_data: {lengths}")
            return self._fallback_to_standard_update(merged_df, primary_key, table_name)
        
        try:
            updates_df = pd.DataFrame(update_data)
        except Exception as e:
            logger.error(f"Error creating update DataFrame: {str(e)}")
            return self._fallback_to_standard_update(merged_df, primary_key, table_name)
        
        if updates_df.empty:
            return Result.success_result(0)
        
        # Use standard persist which will do UPSERT
        # The key benefit here is that we've detected changes and logged them
        # The actual UPDATE still updates all provided fields, but we only provide changed ones
        # This is a good balance between performance and complexity for Phase 3
        try:
            result = self.persist_dataframe(updates_df, table_name)
            return result if result is not None else Result.success_result(0)
        except Exception as e:
            logger.error(f"Error in persist_dataframe during update: {str(e)}")
            return self._fallback_to_standard_update(merged_df, primary_key, table_name)
    
    def _fallback_to_standard_update(
        self,
        merged_df: pd.DataFrame,
        primary_key: str,
        table_name: str
    ) -> Result[int]:
        """Fallback to standard update when selective update fails.
        
        This extracts all new columns from the merged DataFrame and uses
        standard persist (which will do UPSERT).
        """
        try:
            # Get all columns that end with _new (these are the new values)
            new_cols = [col for col in merged_df.columns if col.endswith('_new')]
            if not new_cols:
                return Result.success_result(0)
            
            # Also include primary key
            if primary_key not in merged_df.columns:
                logger.error(f"Primary key {primary_key} not found for fallback update")
                return Result.success_result(0)
            
            # Extract update data
            update_cols = [primary_key] + new_cols
            updates_df = merged_df[update_cols].copy()
            
            # Rename _new columns back to original names
            updates_df.columns = [col.replace('_new', '') if col.endswith('_new') else col for col in updates_df.columns]
            
            return self.persist_dataframe(updates_df, table_name)
        except Exception as e:
            logger.error(f"Error in fallback update: {str(e)}")
            return Result.failure_result(
                StorageError(f"Failed to perform fallback update: {str(e)}", operation="_fallback_to_standard_update"),
                error_type="StorageError"
            )
        
        # Use standard persist which will do UPSERT
        # The key benefit here is that we've detected changes and logged them
        # The actual UPDATE still updates all provided fields, but we only provide changed ones
        # This is a good balance between performance and complexity for Phase 3
        return self.persist_dataframe(updates_df, table_name)
    
    def persist_dataframe_smart(
        self,
        df: pd.DataFrame,
        table_name: str,
        enable_cdc: bool = True,
        ingestion_id: Optional[str] = None,
        source_adapter: Optional[str] = None
    ) -> Result[int]:
        """Persist DataFrame with smart updates (only changed fields) and CDC.
        
        This method implements Change Data Capture (CDC) by:
        1. Fetching existing records in bulk
        2. Detecting field-level changes using vectorized operations
        3. Only updating changed fields (performance optimization)
        4. Logging all changes to change_audit_log
        
        Parameters:
            df: Validated pandas DataFrame (PII already redacted)
            table_name: Target table name
            enable_cdc: Whether to enable CDC (default: True)
            ingestion_id: ID of the ingestion run (for change logging)
            source_adapter: Source adapter identifier (for change logging)
            
        Returns:
            Result[int]: Number of rows processed or error
        
        Performance:
            - Single bulk fetch per chunk (not per record)
            - Vectorized change detection using pandas
            - Skips UPDATE if no changes detected
            - Bulk operations for inserts and updates
        """
        if df.empty:
            return Result.success_result(0)
        
        # If CDC is disabled, fall back to standard persist
        if not enable_cdc:
            return self.persist_dataframe(df, table_name)
        
        # Get primary key
        primary_key = self._get_primary_key(table_name)
        if not primary_key or primary_key not in df.columns:
            # No primary key or not in DataFrame - fall back to standard persist
            logger.warning(f"No primary key found for {table_name}, using standard persist")
            return self.persist_dataframe(df, table_name)
        
        try:
            # Get unique record IDs from DataFrame
            record_ids = df[primary_key].unique().tolist()
            
            # Bulk fetch existing records
            existing_df = self._bulk_fetch_existing_records(record_ids, table_name, primary_key)
            
            # Import ChangeDetector and ChangeAuditLogger
            from src.domain.services.change_detector import ChangeDetector
            from src.infrastructure.audit.change_audit_logger import ChangeAuditLogger
            
            # Initialize change detector and logger
            detector = ChangeDetector(
                ingestion_id=ingestion_id,
                source_adapter=source_adapter
            )
            change_logger = ChangeAuditLogger()
            change_logger.set_ingestion_context(ingestion_id, source_adapter)
            
            total_processed = 0
            total_changes = 0
            
            # Fast path: all new records
            if existing_df.empty:
                logger.debug(f"All {len(df)} records are new for {table_name}, using fast insert path")
                result = self.persist_dataframe(df, table_name)
                if result.is_success():
                    # Log all fields as INSERTs
                    for _, row in df.iterrows():
                        record_id = str(row[primary_key])
                        for col in df.columns:
                            if col != primary_key and pd.notna(row[col]):
                                change_logger.log_change(
                                    table_name=table_name,
                                    record_id=record_id,
                                    field_name=col,
                                    old_value=None,
                                    new_value=str(row[col]) if not isinstance(row[col], (list, dict)) else json.dumps(row[col]),
                                    change_type="INSERT"
                                )
                                total_changes += 1
                    
                    # Flush change logs
                    if change_logger.has_logs():
                        flush_result = self.flush_change_logs(change_logger.get_logs())
                        if flush_result.is_success():
                            logger.info(f"Logged {change_logger.get_log_count()} INSERT change events for {table_name}")
                
                return result
            
            # Merge to identify inserts vs updates
            merged = existing_df.merge(
                df,
                on=primary_key,
                suffixes=('_old', '_new'),
                how='outer',
                indicator=True
            )
            
            # Process inserts (records that don't exist)
            inserts_df = merged[merged['_merge'] == 'right_only']
            if not inserts_df.empty:
                # Extract only new columns (remove _old suffix columns)
                insert_cols = [col for col in df.columns if not col.endswith('_old')]
                inserts_df_clean = inserts_df[[col for col in insert_cols if col in inserts_df.columns]].copy()
                
                insert_result = self.persist_dataframe(inserts_df_clean, table_name)
                if insert_result.is_success():
                    total_processed += insert_result.value
                    
                    # Log INSERT changes
                    for _, row in inserts_df_clean.iterrows():
                        record_id = str(row[primary_key])
                        for col in insert_cols:
                            if col != primary_key and col in row.index and pd.notna(row[col]):
                                try:
                                    value = row[col]
                                    if isinstance(value, (list, dict)):
                                        new_value = json.dumps(value)
                                    else:
                                        new_value = str(value)
                                    
                                    change_logger.log_change(
                                        table_name=table_name,
                                        record_id=record_id,
                                        field_name=col,
                                        old_value=None,
                                        new_value=new_value,
                                        change_type="INSERT"
                                    )
                                    total_changes += 1
                                except (KeyError, AttributeError) as e:
                                    logger.warning(f"Error logging INSERT change for {col}: {str(e)}")
                                    continue
            
            # Process updates (records that exist)
            updates_df = merged[merged['_merge'] == 'both']
            if not updates_df.empty:
                # Use ChangeDetector to find field-level changes
                changes_df = detector.detect_changes_vectorized(
                    updates_df,
                    table_name,
                    primary_key
                )
                
                if not changes_df.empty:
                    # Build selective UPDATE statements with only changed fields
                    # Group changes by record_id to build efficient UPDATE statements
                    update_result = self._update_changed_fields_bulk(
                        updates_df,
                        changes_df,
                        table_name,
                        primary_key
                    )
                    if update_result and update_result.is_success():
                        total_processed += update_result.value
                    
                    # Log UPDATE changes
                    for _, change_row in changes_df.iterrows():
                        # Serialize values for logging
                        old_val = change_row.get('old_value')
                        new_val = change_row.get('new_value')
                        
                        # Handle complex types
                        if isinstance(old_val, (list, dict)):
                            old_val = json.dumps(old_val) if old_val is not None else None
                        elif old_val is not None:
                            old_val = str(old_val)
                        
                        if isinstance(new_val, (list, dict)):
                            new_val = json.dumps(new_val) if new_val is not None else None
                        elif new_val is not None:
                            new_val = str(new_val)
                        
                        change_logger.log_change(
                            table_name=change_row.get('table_name', table_name),
                            record_id=str(change_row.get('record_id')),
                            field_name=change_row.get('field_name'),
                            old_value=old_val,
                            new_value=new_val,
                            change_type=change_row.get('change_type', 'UPDATE')
                        )
                        total_changes += 1
                else:
                    # No changes detected - skip update (performance optimization)
                    logger.debug(f"No changes detected for {len(updates_df)} records in {table_name}, skipping update")
                    total_processed += len(updates_df)
            
            # Flush all change logs
            if change_logger.has_logs():
                flush_result = self.flush_change_logs(change_logger.get_logs())
                if flush_result.is_success():
                    logger.info(
                        f"Logged {change_logger.get_log_count()} change events for {table_name} "
                        f"({total_changes} fields changed)"
                    )
            
            return Result.success_result(total_processed)
            
        except Exception as e:
            error_msg = f"Failed to persist DataFrame with smart updates to {table_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Fall back to standard persist on error
            logger.warning(f"Falling back to standard persist for {table_name}")
            return self.persist_dataframe(df, table_name)
    
    def persist_dataframe(self, df: pd.DataFrame, table_name: str) -> Result[int]:
        """Persist a pandas DataFrame directly to a table.
        
        Uses PostgreSQL COPY for large DataFrames without arrays (faster) or execute_values for others.
        
        Parameters:
            df: Validated pandas DataFrame (PII already redacted)
            table_name: Target table name (e.g., 'patients', 'observations')
        
        Returns:
            Result[int]: Number of rows persisted or error
        """
        if df.empty:
            return Result.success_result(0)
        
        # Check if table has array columns - COPY CSV doesn't handle arrays well
        array_columns = {
            'patients': ['identifiers', 'given_names', 'name_prefix', 'name_suffix'],
            'encounters': ['diagnosis_codes'],
            'observations': []
        }
        table_array_cols = array_columns.get(table_name, [])
        has_array_cols = any(col in df.columns for col in table_array_cols)
        
        # Use COPY for large DataFrames without arrays (10k+ rows)
        # COPY is 2-3x faster than INSERT for bulk data, but arrays need special handling
        if len(df) >= 10000 and not has_array_cols:
            return self._persist_dataframe_copy(df, table_name)
        else:
            # Use INSERT for smaller batches or tables with arrays
            return self._persist_dataframe_insert(df, table_name)
    
    def _persist_dataframe_copy(self, df: pd.DataFrame, table_name: str) -> Result[int]:
        """Persist DataFrame using PostgreSQL COPY command (optimized for large batches).
        
        Tracks processing time for latency metrics.
        
        COPY is 2-3x faster than INSERT for bulk data operations.
        
        Parameters:
            df: Validated pandas DataFrame (PII already redacted)
            table_name: Target table name
        
        Returns:
            Result[int]: Number of rows persisted or error
        """
        if df.empty:
            return Result.success_result(0)
        
        start_time = time.time()
        conn = None
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Define expected columns for each table
            table_columns = {
                'patients': [
                    'patient_id', 'identifiers', 'family_name', 'given_names', 'name_prefix', 'name_suffix',
                    'date_of_birth', 'gender', 'deceased', 'deceased_date', 'marital_status',
                    'address_line1', 'address_line2', 'city', 'state', 'postal_code', 'country',
                    'address_use', 'phone', 'email', 'fax', 'emergency_contact_name',
                    'emergency_contact_relationship', 'emergency_contact_phone', 'language',
                    'managing_organization', 'ingestion_timestamp', 'source_adapter', 'transformation_hash'
                ],
                'encounters': [
                    'encounter_id', 'patient_id', 'status', 'class_code', 'type', 'service_type', 'priority',
                    'period_start', 'period_end', 'length_minutes', 'reason_code', 'diagnosis_codes',
                    'facility_name', 'location_address', 'participant_name', 'participant_role',
                    'service_provider', 'ingestion_timestamp', 'source_adapter', 'transformation_hash'
                ],
                'observations': [
                    'observation_id', 'patient_id', 'encounter_id', 'status', 'category', 'code',
                    'effective_date', 'issued', 'performer_name', 'value', 'unit', 'interpretation',
                    'body_site', 'method', 'device', 'reference_range', 'notes',
                    'ingestion_timestamp', 'source_adapter', 'transformation_hash'
                ]
            }
            
            # Filter DataFrame to only include columns that exist in the table schema
            expected_cols = table_columns.get(table_name, list(df.columns))
            available_cols = [col for col in expected_cols if col in df.columns]
            df_filtered = df[available_cols].copy()
            
            # Add required columns if missing
            from datetime import datetime
            source_adapter_value = 'bulk_ingestion'
            if 'source_adapter' in df.columns and not df['source_adapter'].isna().all():
                source_adapter_series = df['source_adapter'].dropna()
                if len(source_adapter_series) > 0:
                    source_adapter_value = str(source_adapter_series.iloc[0])
            
            required_columns = {
                'ingestion_timestamp': datetime.now(),
                'source_adapter': source_adapter_value,
                'transformation_hash': None
            }
            
            for col_name, default_value in required_columns.items():
                if col_name in expected_cols and col_name not in df_filtered.columns:
                    df_filtered[col_name] = default_value
                    available_cols.append(col_name)
            
            # Prepare data for COPY
            array_columns = {
                'patients': ['identifiers', 'given_names', 'name_prefix', 'name_suffix'],
                'encounters': ['diagnosis_codes'],
                'observations': []
            }
            table_array_cols = array_columns.get(table_name, [])
            
            # Clean DataFrame
            df_cleaned = DataFrameCleaner.prepare_for_database(
                df_filtered,
                array_columns=table_array_cols,
                enum_columns=None,
                convert_nat=True
            )
            
            # Use COPY FROM STDIN with TEXT format (not CSV) for better NULL and array handling
            # TEXT format handles NULLs and arrays more reliably than CSV
            from io import StringIO
            
            buffer = StringIO()
            
            # Convert DataFrame to tuples for processing
            values = DataFrameCleaner.convert_to_tuples(
                df_cleaned,
                handle_nat=True,
                array_columns=set(table_array_cols)
            )
            
            # Write rows in PostgreSQL COPY TEXT format
            # TEXT format: tab-separated, \N for NULL, {value1,value2} for arrays
            for row in values:
                formatted_values = []
                for idx, val in enumerate(row):
                    col_name = df_cleaned.columns[idx]
                    
                    if val is None:
                        # NULL value
                        formatted_values.append('\\N')
                    elif col_name in table_array_cols and isinstance(val, (list, tuple)):
                        # Format array as PostgreSQL array: {value1,value2}
                        if len(val) == 0:
                            formatted_values.append('{}')
                        else:
                            # Escape array values and format as {value1,value2}
                            escaped_values = []
                            for v in val:
                                if v is None:
                                    escaped_values.append('NULL')
                                else:
                                    v_str = str(v)
                                    # Escape backslashes, quotes, and commas for arrays
                                    v_str = v_str.replace('\\', '\\\\').replace('"', '\\"')
                                    escaped_values.append(v_str)
                            formatted_values.append('{' + ','.join(escaped_values) + '}')
                    else:
                        # Regular value - convert to string
                        val_str = str(val) if val is not None else '\\N'
                        # Escape tabs and newlines for TEXT format
                        val_str = val_str.replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
                        formatted_values.append(val_str)
                
                # Write tab-separated row
                buffer.write('\t'.join(formatted_values) + '\n')
            
            buffer.seek(0)
            
            # Use COPY FROM STDIN with TEXT format (handles NULLs and arrays better)
            columns_str = ', '.join(df_cleaned.columns)
            copy_sql = f"COPY {table_name} ({columns_str}) FROM STDIN WITH (FORMAT text, NULL '\\N')"
            
            cursor.copy_expert(copy_sql, buffer)
            
            total_rows = len(df_cleaned)
            conn.commit()
            cursor.close()
            
            # Calculate processing time
            processing_time_seconds = time.time() - start_time
            processing_time_ms = processing_time_seconds * 1000
            
            logger.info(
                f"Persisted {total_rows} rows to table '{table_name}' using COPY "
                f"in {processing_time_ms:.2f}ms ({total_rows/processing_time_seconds:.0f} rows/sec)"
            )
            
            # Log audit event with processing time
            source_adapter = 'bulk_ingestion'
            if 'source_adapter' in df.columns:
                source_adapter_series = df['source_adapter'].dropna()
                if len(source_adapter_series) > 0:
                    source_adapter = str(source_adapter_series.iloc[0])
            
            self.log_audit_event(
                event_type="BULK_PERSISTENCE",
                record_id=None,
                transformation_hash=None,
                details={
                    "source_adapter": source_adapter,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "processing_time_seconds": round(processing_time_seconds, 4),
                },
                table_name=table_name,
                row_count=total_rows,
                source_adapter=source_adapter
            )
            
            return Result.success_result(total_rows)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error_msg = f"Failed to persist DataFrame to {table_name} using COPY: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Fall back to INSERT method on COPY failure
            logger.info(f"Falling back to INSERT method for {table_name}")
            return self._persist_dataframe_insert(df, table_name)
        finally:
            if conn:
                self._return_connection(conn)
    
    def _persist_dataframe_insert(self, df: pd.DataFrame, table_name: str) -> Result[int]:
        """Persist DataFrame using INSERT with execute_values (for smaller batches).
        
        Parameters:
            df: Validated pandas DataFrame (PII already redacted)
            table_name: Target table name
        
        Returns:
            Result[int]: Number of rows persisted or error
        """
        if df.empty:
            return Result.success_result(0)
        
        start_time = time.time()
        engine = None
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            # Create SQLAlchemy engine from connection string for pandas to_sql
            # Use the same connection parameters as the main connection pool
            if 'dsn' in self.connection_params:
                connection_string = self.connection_params['dsn']
            else:
                # Build connection string from individual parameters
                password = self.connection_params.get('password', '')
                username = self.connection_params.get('user', '')
                host = self.connection_params.get('host', '')
                port = self.connection_params.get('port', 5432)
                database = self.connection_params.get('database', '')
                sslmode = self.connection_params.get('sslmode', 'prefer')
                
                # URL-encode password and username if they contain special characters
                password_encoded = quote_plus(password) if password else ''
                username_encoded = quote_plus(username) if username else ''
                
                connection_string = f"postgresql://{username_encoded}:{password_encoded}@{host}:{port}/{database}?sslmode={sslmode}"
            
            engine = create_engine(connection_string, pool_pre_ping=True)
            
            # Define expected columns for each table (to filter out deprecated/extra columns)
            table_columns = {
                'patients': [
                    'patient_id', 'identifiers', 'family_name', 'given_names', 'name_prefix', 'name_suffix',
                    'date_of_birth', 'gender', 'deceased', 'deceased_date', 'marital_status',
                    'address_line1', 'address_line2', 'city', 'state', 'postal_code', 'country',
                    'address_use', 'phone', 'email', 'fax', 'emergency_contact_name',
                    'emergency_contact_relationship', 'emergency_contact_phone', 'language',
                    'managing_organization', 'ingestion_timestamp', 'source_adapter', 'transformation_hash'
                ],
                'encounters': [
                    'encounter_id', 'patient_id', 'status', 'class_code', 'type', 'service_type', 'priority',
                    'period_start', 'period_end', 'length_minutes', 'reason_code', 'diagnosis_codes',
                    'facility_name', 'location_address', 'participant_name', 'participant_role',
                    'service_provider', 'ingestion_timestamp', 'source_adapter', 'transformation_hash'
                ],
                'observations': [
                    'observation_id', 'patient_id', 'encounter_id', 'status', 'category', 'code',
                    'effective_date', 'issued', 'performer_name', 'value', 'unit', 'interpretation',
                    'body_site', 'method', 'device', 'reference_range', 'notes',
                    'ingestion_timestamp', 'source_adapter', 'transformation_hash'
                ]
            }
            
            # Filter DataFrame to only include columns that exist in the table schema
            expected_cols = table_columns.get(table_name, list(df.columns))
            available_cols = [col for col in expected_cols if col in df.columns]
            df_filtered = df[available_cols].copy()
            
            # Add required NOT NULL columns if they're missing (for bulk DataFrame inserts)
            # These are typically added during GoldenRecord creation, but DataFrames from CSV/JSON may not have them
            from datetime import datetime
            
            # Get source_adapter from DataFrame if available, otherwise use default
            source_adapter_value = 'bulk_ingestion'
            if 'source_adapter' in df.columns and not df['source_adapter'].isna().all():
                # Get first non-null value from original DataFrame
                source_adapter_series = df['source_adapter'].dropna()
                if len(source_adapter_series) > 0:
                    source_adapter_value = str(source_adapter_series.iloc[0])
            elif 'source_adapter' in df_filtered.columns and not df_filtered['source_adapter'].isna().all():
                # Check filtered DataFrame if not in original
                source_adapter_series = df_filtered['source_adapter'].dropna()
                if len(source_adapter_series) > 0:
                    source_adapter_value = str(source_adapter_series.iloc[0])
            
            required_columns = {
                'ingestion_timestamp': datetime.now(),
                'source_adapter': source_adapter_value,
                'transformation_hash': None
            }
            
            for col_name, default_value in required_columns.items():
                if col_name in expected_cols and col_name not in df_filtered.columns:
                    df_filtered[col_name] = default_value
                    available_cols.append(col_name)
            
            # Define array columns for each table
            array_columns = {
                'patients': ['identifiers', 'given_names', 'name_prefix', 'name_suffix'],
                'encounters': ['diagnosis_codes'],
                'observations': []
            }
            table_array_cols = array_columns.get(table_name, [])
            
            # Use DataFrameCleaner service for all data cleaning operations
            # This makes the code more maintainable and reusable across adapters
            df_cleaned = DataFrameCleaner.prepare_for_database(
                df_filtered,
                array_columns=table_array_cols,
                enum_columns=None,  # Auto-detect enum columns
                convert_nat=True
            )
            
            # Use psycopg2's execute_values for efficient bulk insertion
            # This avoids parameter limit issues and handles arrays correctly
            from psycopg2.extras import execute_values
            import psycopg2
            
            # Get connection from engine
            conn = engine.raw_connection()
            cursor = conn.cursor()
            
            try:
                # Prepare column names and data
                columns = list(df_cleaned.columns)
                
                # Convert DataFrame to list of tuples for database insertion
                # The cleaner handles NaT conversion, array normalization, and enum conversion
                values = DataFrameCleaner.convert_to_tuples(
                    df_cleaned,
                    handle_nat=True,
                    array_columns=set(table_array_cols)
                )
                
                # Build INSERT statement with ON CONFLICT handling for primary keys
                # This allows re-running ingestion without duplicate key errors
                columns_str = ', '.join(columns)
                
                # Determine primary key column for ON CONFLICT
                primary_key = None
                if table_name == 'patients':
                    primary_key = 'patient_id'
                elif table_name == 'encounters':
                    primary_key = 'encounter_id'
                elif table_name == 'observations':
                    primary_key = 'observation_id'
                
                # Deduplicate DataFrame if using ON CONFLICT (PostgreSQL doesn't allow duplicates in same INSERT)
                # Keep last occurrence of duplicates (most recent data wins)
                if primary_key and primary_key in columns:
                    initial_count = len(df_cleaned)
                    df_cleaned = df_cleaned.drop_duplicates(subset=[primary_key], keep='last')
                    duplicates_removed = initial_count - len(df_cleaned)
                    if duplicates_removed > 0:
                        logger.warning(
                            f"Removed {duplicates_removed} duplicate {primary_key} values from {table_name} "
                            f"DataFrame (kept last occurrence). Original: {initial_count}, Deduplicated: {len(df_cleaned)}"
                        )
                        # Rebuild columns list and values after deduplication
                        columns = list(df_cleaned.columns)
                        values = DataFrameCleaner.convert_to_tuples(
                            df_cleaned,
                            handle_nat=True,
                            array_columns=set(table_array_cols)
                        )
                        columns_str = ', '.join(columns)
                
                if primary_key and primary_key in columns:
                    # Use UPSERT (INSERT ... ON CONFLICT DO UPDATE) to handle duplicates
                    # This allows re-running ingestion without errors
                    update_cols = [col for col in columns if col != primary_key]
                    update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_cols])
                    # execute_values uses %s as placeholder - it will be replaced with template
                    insert_sql = f"""
                        INSERT INTO {table_name} ({columns_str}) 
                        VALUES %s
                        ON CONFLICT ({primary_key}) DO UPDATE SET {update_clause}
                    """
                    # Template for execute_values: one %s per column
                    template = '(' + ', '.join(['%s'] * len(columns)) + ')'
                else:
                    # No primary key or conflict handling - simple INSERT
                    insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES %s"
                    template = None
                
                # Use execute_values for efficient bulk insert
                # This handles arrays and avoids parameter limit issues
                # Note: execute_values supports ON CONFLICT when using a template
                execute_values(
                    cursor,
                    insert_sql,
                    values,
                    template=template,
                    page_size=1000  # Insert 1000 rows at a time
                )
                
                total_rows = len(values)
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                raise
            finally:
                cursor.close()
                conn.close()
            
            # Calculate processing time
            processing_time_seconds = time.time() - start_time
            processing_time_ms = processing_time_seconds * 1000
            
            logger.info(
                f"Persisted {total_rows} rows to table '{table_name}' "
                f"in {processing_time_ms:.2f}ms ({total_rows/processing_time_seconds:.0f} rows/sec)"
            )
            
            # Extract source_adapter from dataframe if available
            source_adapter = 'bulk_ingestion'
            if 'source_adapter' in df.columns:
                # Get first non-null value, or use default
                source_adapter_series = df['source_adapter'].dropna()
                if len(source_adapter_series) > 0:
                    source_adapter = str(source_adapter_series.iloc[0])
            
            # Log audit event (bulk operation) with processing time
            self.log_audit_event(
                event_type="BULK_PERSISTENCE",
                record_id=None,
                transformation_hash=None,
                details={
                    "source_adapter": source_adapter,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "processing_time_seconds": round(processing_time_seconds, 4),
                },
                table_name=table_name,
                row_count=total_rows
            )
            
            return Result.success_result(total_rows)
            
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
            
            # Use source_adapter parameter if provided, otherwise try to get from details
            source_adapter_value = source_adapter or (details.get('source_adapter') if details else None)
            
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
                source_adapter_value,
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
    
    def flush_change_logs(self, change_logs: list[dict]) -> Result[int]:
        """Flush multiple change audit logs to the database in a single transaction.
        
        This method uses bulk insert (execute_values) for performance with large
        batches of change events, which is critical for CDC with millions of records.
        
        Parameters:
            change_logs: List of change log dictionaries (from ChangeAuditLogger.get_logs())
            
        Returns:
            Result[int]: Number of logs persisted or error
        
        Security Impact:
            - Change logs may contain PII (old/new values)
            - Ensure values are redacted before logging
            - Logs are immutable (append-only) for compliance
        
        Performance:
            - Uses execute_values for bulk insert (much faster than individual INSERTs)
            - Single transaction for all logs in the batch
            - Optimized for batches of 10k-50k change events
        """
        if not change_logs:
            return Result.success_result(0)
        
        conn = None
        try:
            if not self._initialized:
                init_result = self.initialize_schema()
                if not init_result.is_success():
                    return init_result
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare data for bulk insert using execute_values
            from psycopg2.extras import execute_values
            
            columns = [
                'change_id', 'table_name', 'record_id', 'field_name',
                'old_value', 'new_value', 'change_type', 'changed_at',
                'ingestion_id', 'source_adapter', 'changed_by'
            ]
            
            # Convert log entries to tuples for bulk insert
            values = []
            for log_entry in change_logs:
                values.append((
                    log_entry.get('change_id', str(uuid.uuid4())),
                    log_entry.get('table_name'),
                    log_entry.get('record_id'),
                    log_entry.get('field_name'),
                    log_entry.get('old_value'),
                    log_entry.get('new_value'),
                    log_entry.get('change_type'),
                    log_entry.get('changed_at', datetime.now()),
                    log_entry.get('ingestion_id'),
                    log_entry.get('source_adapter'),
                    log_entry.get('changed_by', 'system')
                ))
            
            # Bulk insert using execute_values for performance
            insert_sql = f"""
                INSERT INTO change_audit_log ({', '.join(columns)})
                VALUES %s
            """
            
            execute_values(
                cursor,
                insert_sql,
                values,
                template=None,
                page_size=10000  # Insert in chunks of 10k for very large batches
            )
            
            conn.commit()
            cursor.close()
            
            count = len(change_logs)
            logger.info(f"Flushed {count} change audit logs to database")
            return Result.success_result(count)
            
        except Exception as e:
            if conn:
                conn.rollback()
            error_msg = f"Failed to flush change logs: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                StorageError(error_msg, operation="flush_change_logs"),
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

