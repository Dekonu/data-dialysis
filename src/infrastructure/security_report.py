"""Security Report Generator.

This module provides functionality to generate security reports from redaction logs.
Reports summarize all PII redaction events for compliance and auditing purposes.

Security Impact:
    - Provides comprehensive audit trail of all redactions
    - Enables compliance reporting for HIPAA/GDPR
    - Supports forensic analysis of data transformations
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.domain.ports import Result, StoragePort


def generate_security_report(
    storage: StoragePort,
    output_path: Optional[str] = None,
    ingestion_id: Optional[str] = None,
    start_timestamp: Optional[datetime] = None,
    end_timestamp: Optional[datetime] = None
) -> Result[dict]:
    """Generate a security report from redaction logs.
    
    Parameters:
        storage: Storage adapter instance (must support generate_security_report)
        output_path: Optional path to save report as JSON file
        ingestion_id: Optional ingestion ID to filter by
        start_timestamp: Optional start time for report period
        end_timestamp: Optional end time for report period
    
    Returns:
        Result[dict]: Security report dictionary or error
    
    Security Impact:
        - Report contains summary of all PII redactions
        - Can be saved to file for compliance documentation
        - Includes statistics and detailed event log
    """
    if not hasattr(storage, 'generate_security_report'):
        return Result.failure_result(
            ValueError("Storage adapter does not support security report generation"),
            error_type="ValueError"
        )
    
    # Generate report from storage
    report_result = storage.generate_security_report(
        ingestion_id=ingestion_id,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp
    )
    
    if not report_result.is_success():
        return report_result
    
    report = report_result.value
    
    # Save to file if output path provided
    if output_path:
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            return Result.success_result({
                **report,
                "saved_to": str(output_file)
            })
        except Exception as e:
            return Result.failure_result(
                ValueError(f"Failed to save report to {output_path}: {str(e)}"),
                error_type="ValueError"
            )
    
    return report_result


def print_security_report_summary(report: dict) -> None:
    """Print a human-readable summary of the security report.
    
    Parameters:
        report: Security report dictionary
    """
    print("=" * 70)
    print("SECURITY REPORT - PII Redaction Summary")
    print("=" * 70)
    
    summary = report.get('summary', {})
    print(f"\nTotal Redactions: {summary.get('total_redactions', 0)}")
    
    if report.get('ingestion_id'):
        print(f"Ingestion ID: {report['ingestion_id']}")
    
    if report.get('start_timestamp'):
        print(f"Period Start: {report['start_timestamp']}")
    if report.get('end_timestamp'):
        print(f"Period End: {report['end_timestamp']}")
    
    if summary.get('total_redactions', 0) == 0:
        print("\nWARNING: No redaction events logged.")
        print("  Note: Redaction logging may not be fully integrated.")
        print("  Redactions still occur but may not be logged to the logs table.")
    else:
        print("\nRedactions by Field:")
        for field, count in sorted(summary.get('redactions_by_field', {}).items()):
            print(f"  {field}: {count}")
        
        print("\nRedactions by Rule:")
        for rule, count in sorted(summary.get('redactions_by_rule', {}).items()):
            print(f"  {rule}: {count}")
        
        print("\nRedactions by Adapter:")
        for adapter, count in sorted(summary.get('redactions_by_adapter', {}).items()):
            print(f"  {adapter}: {count}")
    
    print("\n" + "=" * 70)

