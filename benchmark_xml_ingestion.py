"""Benchmark XML ingestion performance across different file sizes.

This script tests XML ingestion performance using the generated test files
and provides detailed metrics including throughput, memory usage, and timing.

Security Impact:
    - Tests ingestion pipeline with realistic data volumes
    - Measures performance characteristics for capacity planning
"""

import time
import sys
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple
from statistics import mean, median

from src.adapters.ingesters.xml_ingester import XMLIngester
from src.domain.ports import Result


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def format_time(seconds: float) -> str:
    """Format time as human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def benchmark_file(
    file_path: Path,
    config_path: Path,
    streaming_enabled: bool = None,
    iterations: int = 1
) -> Dict:
    """Benchmark ingestion of a single XML file.
    
    Parameters:
        file_path: Path to XML file to ingest
        config_path: Path to XML configuration file
        streaming_enabled: Whether to use streaming mode (None = auto-detect)
        iterations: Number of iterations to run (for averaging)
    
    Returns:
        Dictionary with benchmark results
    """
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"\n{'='*70}")
    print(f"Benchmarking: {file_path.name}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"Streaming mode: {'auto' if streaming_enabled is None else ('enabled' if streaming_enabled else 'disabled')}")
    print(f"Iterations: {iterations}")
    print(f"{'='*70}")
    
    results = []
    
    for iteration in range(1, iterations + 1):
        print(f"\nIteration {iteration}/{iterations}...")
        
        # Create ingester
        ingester = XMLIngester(
            config_path=str(config_path),
            streaming_enabled=streaming_enabled
        )
        
        # Start memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Ingest file
        success_count = 0
        failure_count = 0
        total_records = 0
        last_progress_time = start_time
        progress_interval = 5.0  # Show progress every 5 seconds
        
        try:
            print("  Processing records...", flush=True)
            for result in ingester.ingest(str(file_path)):
                total_records += 1
                if result.is_success:
                    success_count += 1
                else:
                    failure_count += 1
                    # Log first few failures for debugging
                    if failure_count <= 3:
                        print(f"    WARNING: Record {total_records} failed: {result.error_type} - {result.error_message[:100]}", flush=True)
                
                # Show progress every N seconds
                current_time = time.perf_counter()
                if current_time - last_progress_time >= progress_interval:
                    elapsed_so_far = current_time - start_time
                    rate = total_records / elapsed_so_far if elapsed_so_far > 0 else 0
                    print(f"    Progress: {total_records:,} records in {format_time(elapsed_so_far)} "
                          f"({rate:,.0f} rec/sec, {success_count:,} success, {failure_count:,} failed)", flush=True)
                    last_progress_time = current_time
                
                # Show progress every 10000 records for very large files
                if total_records % 10000 == 0 and total_records > 0:
                    elapsed_so_far = current_time - start_time
                    rate = total_records / elapsed_so_far if elapsed_so_far > 0 else 0
                    print(f"    {total_records:,} records processed ({rate:,.0f} rec/sec)...", flush=True)
        except KeyboardInterrupt:
            print(f"\n  INTERRUPTED: Stopped at {total_records:,} records")
            tracemalloc.stop()
            raise
        except Exception as e:
            print(f"\n  ERROR at record {total_records:,}: {type(e).__name__}: {str(e)}")
            import traceback
            print("  Full traceback:")
            traceback.print_exc()
            tracemalloc.stop()
            # Don't continue - we want to see the error
            raise
        
        # Stop timing
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate throughput
        records_per_second = total_records / elapsed_time if elapsed_time > 0 else 0
        mb_per_second = file_size_mb / elapsed_time if elapsed_time > 0 else 0
        
        result_data = {
            'iteration': iteration,
            'elapsed_time': elapsed_time,
            'total_records': total_records,
            'success_count': success_count,
            'failure_count': failure_count,
            'records_per_second': records_per_second,
            'mb_per_second': mb_per_second,
            'peak_memory_mb': peak / (1024 * 1024),
            'current_memory_mb': current / (1024 * 1024)
        }
        
        results.append(result_data)
        
        print(f"  Time: {format_time(elapsed_time)}")
        print(f"  Records: {total_records:,} (success: {success_count:,}, failed: {failure_count:,})")
        print(f"  Throughput: {records_per_second:,.0f} records/sec, {mb_per_second:.2f} MB/sec")
        print(f"  Peak memory: {format_bytes(peak)}")
    
    # Calculate averages if multiple iterations
    if iterations > 1:
        avg_time = mean([r['elapsed_time'] for r in results])
        avg_records_per_sec = mean([r['records_per_second'] for r in results])
        avg_mb_per_sec = mean([r['mb_per_second'] for r in results])
        avg_peak_memory = mean([r['peak_memory_mb'] for r in results])
        
        median_time = median([r['elapsed_time'] for r in results])
        
        print(f"\n  Average time: {format_time(avg_time)} (median: {format_time(median_time)})")
        print(f"  Average throughput: {avg_records_per_sec:,.0f} records/sec, {avg_mb_per_sec:.2f} MB/sec")
        print(f"  Average peak memory: {avg_peak_memory:.2f} MB")
    else:
        # Single iteration - use the single result
        avg_time = results[0]['elapsed_time']
        avg_records_per_sec = results[0]['records_per_second']
        avg_mb_per_sec = results[0]['mb_per_second']
        avg_peak_memory = results[0]['peak_memory_mb']
    
    return {
        'file': file_path.name,
        'file_size_mb': file_size_mb,
        'iterations': iterations,
        'results': results,
        'avg_time': avg_time,
        'avg_records_per_sec': avg_records_per_sec,
        'avg_mb_per_sec': avg_mb_per_sec,
        'avg_peak_memory_mb': avg_peak_memory
    }


def main():
    """Run benchmarks on all generated XML test files."""
    print("=" * 70)
    print("XML Ingestion Performance Benchmark")
    print("=" * 70)
    
    # Check for test data directory
    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        print(f"ERROR: Test data directory not found: {test_data_dir}")
        print("Please run generate_xml_test_files.py first to generate test files.")
        sys.exit(1)
    
    # Check for config file
    config_file = Path("xml_config.json")
    if not config_file.exists():
        print(f"ERROR: XML config file not found: {config_file}")
        sys.exit(1)
    
    # Find all test files and sort by size (smallest first)
    test_files = list(test_data_dir.glob("test_data_*.xml"))
    if not test_files:
        print(f"ERROR: No test files found in {test_data_dir}")
        print("Please run generate_xml_test_files.py first to generate test files.")
        sys.exit(1)
    
    # Sort by file size (smallest first) for easier debugging
    test_files.sort(key=lambda f: f.stat().st_size)
    
    print(f"\nFound {len(test_files)} test files (sorted by size):")
    for f in test_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.2f} MB")
    
    # Ask for streaming mode preference
    print("\nStreaming mode options:")
    print("  1. Auto-detect (based on file size)")
    print("  2. Always use streaming")
    print("  3. Never use streaming (traditional mode)")
    
    choice = input("\nSelect option (1-3, default=1): ").strip() or "1"
    
    if choice == "2":
        streaming_enabled = True
    elif choice == "3":
        streaming_enabled = False
    else:
        streaming_enabled = None  # Auto-detect
    
    # Ask for iterations
    iterations_input = input("Number of iterations per file (default=1): ").strip()
    iterations = int(iterations_input) if iterations_input else 1
    
    # Run benchmarks
    all_results = []
    
    try:
        for test_file in test_files[]:
            try:
                result = benchmark_file(
                    test_file,
                    config_file,
                    streaming_enabled=streaming_enabled,
                    iterations=iterations
                )
                all_results.append(result)
            except Exception as e:
                print(f"\nERROR: Failed to benchmark {test_file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue with next file
                continue
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        if all_results:
            print("Partial results available.")
        else:
            sys.exit(1)
    
    # Print summary
    if not all_results:
        print("\nERROR: No benchmark results to display.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"{'File':<20} {'Size (MB)':<12} {'Time':<15} {'Records/sec':<15} {'MB/sec':<12} {'Peak Mem (MB)':<15}")
    print("-" * 70)
    
    try:
        for result in all_results:
            print(f"{result['file']:<20} "
                  f"{result['file_size_mb']:<12.2f} "
                  f"{format_time(result['avg_time']):<15} "
                  f"{result['avg_records_per_sec']:<15,.0f} "
                  f"{result['avg_mb_per_sec']:<12.2f} "
                  f"{result['avg_peak_memory_mb']:<15.2f}")
    except Exception as e:
        print(f"\nERROR: Failed to print summary: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

