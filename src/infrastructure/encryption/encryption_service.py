"""Encryption service for PII data.

This service provides encryption/decryption capabilities for storing
original unredacted data in the raw data vault.

Security Impact:
    - Uses AES-256 encryption via Fernet (symmetric encryption)
    - Keys derived from environment variables or key management service
    - All PII in raw vault is encrypted at rest
    - Enables CDC to track real changes, not redacted masks

Architecture:
    - Infrastructure layer component
    - Used by storage adapters for raw vault persistence
    - Follows Hexagonal Architecture: isolated from domain core
"""

import base64
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class EncryptionService:
    """Service for encrypting/decrypting PII data.
    
    Uses AES-256 encryption via Fernet (symmetric encryption).
    Keys are derived from environment variable or key management service.
    
    Security Impact:
        - All PII data is encrypted before storage in raw vault
        - Keys should be stored securely (env vars, key management service)
        - Supports key rotation via key_id tracking
    """
    
    def __init__(self, key: Optional[bytes] = None, key_id: Optional[str] = None):
        """Initialize encryption service.
        
        Parameters:
            key: Encryption key bytes (if None, derived from env var)
            key_id: Identifier for the encryption key (for rotation support)
        """
        if key is None:
            key = self._derive_key_from_env()
        
        try:
            self.cipher = Fernet(key)
        except ValueError as e:
            logger.error(f"Invalid encryption key: {str(e)}")
            raise ValueError("Invalid encryption key format. Key must be base64-encoded 32-byte key.") from e
        
        self.key_id = key_id or os.getenv('ENCRYPTION_KEY_ID', 'default')
        logger.debug(f"EncryptionService initialized with key_id: {self.key_id}")
    
    @staticmethod
    def _derive_key_from_env() -> bytes:
        """Derive encryption key from environment variable.
        
        Looks for ENCRYPTION_KEY environment variable. If not found,
        derives from ENCRYPTION_KEY_PASSWORD using PBKDF2.
        
        Returns:
            bytes: 32-byte encryption key
            
        Raises:
            ValueError: If no encryption key can be derived
        """
        # Try direct key from env (base64-encoded)
        key_str = os.getenv('ENCRYPTION_KEY')
        if key_str:
            try:
                return base64.urlsafe_b64decode(key_str.encode('utf-8'))
            except Exception as e:
                logger.warning(f"Failed to decode ENCRYPTION_KEY: {str(e)}")
        
        # Try password-based derivation
        password = os.getenv('ENCRYPTION_KEY_PASSWORD')
        if password:
            salt = os.getenv('ENCRYPTION_KEY_SALT', 'datadialysis-salt').encode('utf-8')
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
            return key
        
        # Fallback: generate a key (NOT SECURE for production!)
        logger.warning(
            "No ENCRYPTION_KEY or ENCRYPTION_KEY_PASSWORD found. "
            "Using generated key (NOT SECURE for production!). "
            "Set ENCRYPTION_KEY environment variable."
        )
        return Fernet.generate_key()
    
    def encrypt_value(self, value: Any) -> Optional[bytes]:
        """Encrypt a single value.
        
        Parameters:
            value: Value to encrypt (will be converted to string)
            
        Returns:
            bytes: Encrypted value, or None if input is None
        """
        if value is None:
            return None
        
        try:
            value_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, default=str)
            return self.cipher.encrypt(value_str.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to encrypt value: {str(e)}")
            raise
    
    def decrypt_value(self, encrypted: Optional[bytes]) -> Optional[str]:
        """Decrypt a single value.
        
        Parameters:
            encrypted: Encrypted bytes
            
        Returns:
            str: Decrypted value, or None if input is None
        """
        if encrypted is None:
            return None
        
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decrypt value: {str(e)}")
            raise
    
    def encrypt_record(self, record_dict: Dict[str, Any]) -> bytes:
        """Encrypt entire record as JSON.
        
        Parameters:
            record_dict: Dictionary representing the record
            
        Returns:
            bytes: Encrypted JSON bytes
        """
        try:
            json_str = json.dumps(record_dict, default=str, sort_keys=True)
            return self.cipher.encrypt(json_str.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to encrypt record: {str(e)}")
            raise
    
    def decrypt_record(self, encrypted: bytes) -> Dict[str, Any]:
        """Decrypt entire record from JSON.
        
        Parameters:
            encrypted: Encrypted JSON bytes
            
        Returns:
            dict: Decrypted record dictionary
        """
        try:
            json_str = self.cipher.decrypt(encrypted).decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to decrypt record: {str(e)}")
            raise
    
    @staticmethod
    def hash_value(value: Any) -> Optional[str]:
        """Generate SHA256 hash of a value (for verification without decryption).
        
        Parameters:
            value: Value to hash
            
        Returns:
            str: Hex digest of SHA256 hash, or None if input is None
        """
        if value is None:
            return None
        
        value_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, default=str, sort_keys=True)
        return hashlib.sha256(value_str.encode('utf-8')).hexdigest()
    
    def get_key_id(self) -> str:
        """Get the current encryption key ID."""
        return self.key_id
