"""Command Line Interface for Clinical-Sieve Data Ingestion Engine.

This module provides a modern CLI using Typer for running the ingestion pipeline
with comprehensive options and security features.

Security Impact:
    - All commands validate inputs before processing
    - Security reports are generated automatically
    - Audit trails are maintained for compliance
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.infrastructure.settings import settings
from src.infrastructure.config_manager import get_database_config
from src.infrastructure.redaction_logger import get_redaction_logger, reset_redaction_logger
from src.infrastructure.redaction_context import redaction_context
from src.infrastructure.security_report import generate_security_report, print_security_report_summary
from src.adapters.ingesters import get_adapter
from src.adapters.storage import DuckDBAdapter, PostgreSQLAdapter
from src.domain.ports import Result, StoragePort
from src.domain.guardrails import CircuitBreaker, CircuitBreakerConfig

# Initialize Typer app and Rich console
app = typer.Typer(
    name="datadialysis",
    help="Clinical-Sieve: Self-Securing Data Ingestion Engine",
    add_completion=False
)
console = Console()


def create_storage_adapter_cli() -> StoragePort:
    """Create storage adapter based on configuration (CLI wrapper)."""
    try:
        from src.main import create_storage_adapter
        return create_storage_adapter()
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to create storage adapter: {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def ingest(
    input_file: Path = typer.Argument(..., help="Input file path (CSV, JSON, or XML)", exists=True),
    xml_config: Optional[Path] = typer.Option(None, "--xml-config", "-c", help="XML configuration file path (required for XML sources)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size for processing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    no_security_report: bool = typer.Option(False, "--no-security-report", help="Skip security report generation"),
    streaming: Optional[bool] = typer.Option(None, "--streaming/--no-streaming", help="Force streaming mode (XML only)"),
) -> None:
    """Ingest clinical data from CSV, JSON, or XML files.
    
    This command processes data through the self-securing pipeline:
    1. Validates and redacts PII
    2. Enforces schema constraints
    3. Persists to configured storage
    4. Generates security report
    
    Examples:
        datadialysis ingest data/patients.csv
        datadialysis ingest data/encounters.xml --xml-config mappings.json
        datadialysis ingest data/observations.json --batch-size 5000 --verbose
    """
    # Set logging level
    import logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[dim]Verbose logging enabled[/dim]")
    
    # Validate XML config if needed
    if input_file.suffix.lower() == '.xml' and not xml_config:
        console.print("[red]✗[/red] XML files require --xml-config option")
        raise typer.Exit(code=1)
    
    # Print startup information
    console.print(f"\n[bold blue]Clinical-Sieve Data Ingestion Engine[/bold blue]")
    console.print(f"[dim]Input file:[/dim] {input_file}")
    console.print(f"[dim]Database:[/dim] {settings.db_config.db_type}")
    if settings.db_config.db_type == "duckdb":
        console.print(f"[dim]Database path:[/dim] {settings.get_db_path()}")
    elif settings.db_config.db_type == "postgresql":
        console.print(f"[dim]Database host:[/dim] {settings.db_config.host}")
    console.print(f"[dim]Batch size:[/dim] {batch_size or settings.batch_size}")
    console.print()
    
    # Create storage adapter
    try:
        with console.status("[bold green]Initializing storage...") as status:
            storage = create_storage_adapter_cli()
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize storage: {str(e)}")
        raise typer.Exit(code=1)
    
    # Process ingestion
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing records...", total=None)
            
            success_count, failure_count, ingestion_id = process_ingestion(
                source=str(input_file),
                storage=storage,
                xml_config_path=str(xml_config) if xml_config else None,
                batch_size=batch_size
            )
            
            progress.update(task, completed=True)
        
        # Print summary
        total_count = success_count + failure_count
        console.print("\n[bold]Ingestion Summary:[/bold]")
        
        summary_table = Table(show_header=False, box=None, padding=(0, 2))
        summary_table.add_row("Total processed:", f"[bold]{total_count:,}[/bold]")
        summary_table.add_row("Successful:", f"[green]{success_count:,}[/green]")
        summary_table.add_row("Failed:", f"[red]{failure_count:,}[/red]" if failure_count > 0 else f"{failure_count:,}")
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            summary_table.add_row("Success rate:", f"{success_rate:.2f}%")
        summary_table.add_row("Ingestion ID:", ingestion_id)
        console.print(summary_table)
        
        # Generate security report
        if not no_security_report:
            console.print("\n[bold]Security Report:[/bold]")
            try:
                report_result = generate_security_report(
                    storage=storage,
                    ingestion_id=ingestion_id
                )
                
                if report_result.is_success():
                    report = report_result.value
                    print_security_report_summary(report)
                    
                    # Save report if enabled
                    if settings.save_security_report:
                        from src.infrastructure.settings import settings
                        report_file = Path(settings.security_report_dir) / f"security_report_{ingestion_id}.json"
                        report_file.parent.mkdir(exist_ok=True)
                        save_result = generate_security_report(
                            storage=storage,
                            output_path=str(report_file),
                            ingestion_id=ingestion_id
                        )
                        if save_result.is_success():
                            console.print(f"\n[green]✓[/green] Security report saved: {report_file}")
                else:
                    console.print(f"[yellow]⚠[/yellow] Failed to generate security report: {report_result.error}")
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Error generating security report: {str(e)}")
        
        # Exit with appropriate code
        if failure_count > 0:
            console.print(f"\n[yellow]⚠[/yellow] Ingestion completed with {failure_count} failures")
            raise typer.Exit(code=1)
        else:
            console.print(f"\n[green]✓[/green] Ingestion completed successfully")
            raise typer.Exit(code=0)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Ingestion interrupted by user")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]✗[/red] Ingestion failed: {str(e)}")
        if verbose:
            import traceback
            console.print_exception()
        raise typer.Exit(code=1)
    finally:
        # Clean up storage connection
        try:
            storage.close()
        except Exception:
            pass


@app.command()
def benchmark(
    test_data_dir: Path = typer.Argument(..., help="Directory containing test XML files", exists=True, file_okay=False),
    xml_config: Path = typer.Argument(..., help="XML configuration file", exists=True),
    streaming: Optional[bool] = typer.Option(None, "--streaming/--no-streaming", help="Force streaming mode"),
    iterations: int = typer.Option(1, "--iterations", "-n", help="Number of iterations per file"),
) -> None:
    """Benchmark XML ingestion performance across different file sizes.
    
    This command runs performance benchmarks on XML test files to measure
    throughput, memory usage, and processing time.
    
    Examples:
        datadialysis benchmark test_data/ xml_config.json
        datadialysis benchmark test_data/ xml_config.json --iterations 3
    """
    console.print("[bold blue]XML Ingestion Performance Benchmark[/bold blue]\n")
    
    # Import benchmark function
    try:
        from benchmark_xml_ingestion import benchmark_file, format_time
        from pathlib import Path as PathLib
    except ImportError:
        console.print("[red]✗[/red] Benchmark module not found. Run from project root.")
        raise typer.Exit(code=1)
    
    # Find test files
    test_files = sorted(test_data_dir.glob("test_data_*.xml"))
    if not test_files:
        console.print(f"[red]✗[/red] No test files found in {test_data_dir}")
        raise typer.Exit(code=1)
    
    console.print(f"Found {len(test_files)} test files:\n")
    for f in test_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        console.print(f"  • {f.name}: {size_mb:.2f} MB")
    console.print()
    
    # Run benchmarks
    all_results = []
    for test_file in test_files:
        try:
            result = benchmark_file(
                test_file,
                xml_config,
                streaming_enabled=streaming,
                iterations=iterations
            )
            all_results.append(result)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to benchmark {test_file.name}: {str(e)}")
            continue
    
    # Print summary
    if all_results:
        console.print("\n[bold]Benchmark Summary:[/bold]\n")
        summary_table = Table(show_header=True, header_style="bold")
        summary_table.add_column("File", style="cyan")
        summary_table.add_column("Size (MB)", justify="right")
        summary_table.add_column("Time", justify="right")
        summary_table.add_column("Records/sec", justify="right")
        summary_table.add_column("MB/sec", justify="right")
        summary_table.add_column("Peak Mem (MB)", justify="right")
        
        for result in all_results:
            summary_table.add_row(
                result['file'],
                f"{result['file_size_mb']:.2f}",
                format_time(result['avg_time']),
                f"{result['avg_records_per_sec']:,.0f}",
                f"{result['avg_mb_per_sec']:.2f}",
                f"{result['avg_peak_memory_mb']:.2f}"
            )
        
        console.print(summary_table)
        console.print("\n[green]✓[/green] Benchmark complete")
    else:
        console.print("[red]✗[/red] No benchmark results to display")
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Display system information and configuration."""
    console.print("[bold blue]System Information[/bold blue]\n")
    
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_row("Application:", settings.app_name)
    info_table.add_row("Database Type:", settings.db_config.db_type)
    
    if settings.db_config.db_type == "duckdb":
        info_table.add_row("Database Path:", settings.get_db_path())
    elif settings.db_config.db_type == "postgresql":
        info_table.add_row("Database Host:", settings.db_config.host)
        info_table.add_row("Database Name:", settings.db_config.database)
    
    info_table.add_row("Batch Size:", str(settings.batch_size))
    info_table.add_row("Chunk Size:", str(settings.chunk_size))
    info_table.add_row("Circuit Breaker:", "Enabled" if settings.circuit_breaker_enabled else "Disabled")
    info_table.add_row("XML Streaming:", "Enabled" if settings.xml_streaming_enabled else "Disabled")
    info_table.add_row("XML Streaming Threshold:", f"{settings.xml_streaming_threshold / (1024*1024):.0f} MB")
    
    console.print(info_table)


@app.callback()
def main_callback(
    version: bool = typer.Option(False, "--version", help="Show version information")
) -> None:
    """Clinical-Sieve: Self-Securing Data Ingestion Engine."""
    if version:
        console.print("Clinical-Sieve v1.0.0")
        raise typer.Exit()


if __name__ == "__main__":
    app()

