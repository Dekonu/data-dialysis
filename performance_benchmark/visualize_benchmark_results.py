"""Enhanced visualization script for benchmark results from CSV.

This script creates comprehensive, publication-quality visualizations with:
- Human-readable axis formatting
- Performance scaling analysis
- Data generation comparison (records per MB)
- Implementation vs data structure analysis
- Adaptive chunking performance comparison

Security Impact:
    - Visualizations help identify performance bottlenecks
    - Memory profiling validates resource efficiency
    - Enables optimization of security-critical paths

Academic Value:
    - Demonstrates quantitative analysis methodology
    - Provides empirical evidence for performance claims
    - Enables comparative evaluation across formats
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for academic-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_csv_results(results_path: Path) -> List[Dict]:
    """Load benchmark results from CSV file."""
    results = []
    with open(results_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            numeric_fields = [
                'file_size_mb', 'num_records', 'total_time_seconds',
                'processing_time_seconds', 'upload_time_seconds',
                'ingestion_time_seconds', 'processing_task_time_seconds',
                'persistence_time_seconds', 'raw_vault_time_seconds',
                'cdc_time_seconds', 'records_per_second', 'mb_per_second',
                'peak_memory_mb', 'avg_memory_mb', 'memory_efficiency_mb_per_record',
                'num_batches', 'avg_batch_time_seconds',
                'avg_batch_upload_time_seconds', 'avg_batch_processing_time_seconds',
                'min_batch_upload_time_seconds', 'max_batch_upload_time_seconds',
                'median_batch_upload_time_seconds', 'p95_batch_upload_time_seconds',
                'p99_batch_upload_time_seconds', 'min_batch_processing_time_seconds',
                'max_batch_processing_time_seconds', 'median_batch_processing_time_seconds',
                'p95_batch_processing_time_seconds', 'p99_batch_processing_time_seconds',
                'records_successful', 'records_failed', 'success_rate',
                'cpu_usage_percent', 'batch_size',
            'circuit_breaker_failure_rate', 'circuit_breaker_threshold',
            'circuit_breaker_total_processed', 'circuit_breaker_total_failures',
            'xml_memory_efficiency', 'adaptive_chunking_enabled'
            ]
            for field in numeric_fields:
                if field in row and row[field]:
                    try:
                        row[field] = float(row[field])
                    except (ValueError, TypeError):
                        row[field] = None
                else:
                    row[field] = None
            
            # Convert boolean fields
            boolean_fields = ['circuit_breaker_opened', 'xml_streaming_enabled', 'adaptive_chunking_enabled']
            for field in boolean_fields:
                if field in row and row[field]:
                    row[field] = row[field].lower() in ('true', '1', 'yes')
                else:
                    row[field] = False
            
            # Map adapter_type to file_format for compatibility
            if 'adapter_type' in row:
                row['file_format'] = row['adapter_type']
            
            results.append(row)
    
    return results


def format_large_number(value: float) -> str:
    """Format large numbers in human-readable format."""
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.1f}"


def format_time(value: float) -> str:
    """Format time values in human-readable format with 4-5 significant figures."""
    if value >= 3600:
        return f"{value/3600:.2f}h"
    elif value >= 60:
        return f"{value/60:.2f}m"
    elif value >= 1:
        # For seconds >= 1, show 2-3 decimal places
        if value >= 10:
            return f"{value:.1f}s"
        else:
            return f"{value:.2f}s"
    elif value >= 0.001:
        # For milliseconds, show 1-2 decimal places
        ms_value = value * 1000
        if ms_value >= 10:
            return f"{ms_value:.1f}ms"
        else:
            return f"{ms_value:.2f}ms"
    else:
        # For microseconds, show 1-2 decimal places
        us_value = value * 1000000
        if us_value >= 10:
            return f"{us_value:.1f}μs"
        else:
            return f"{us_value:.2f}μs"


def format_time_adaptive(values: List[float]) -> str:
    """Format time values adaptively based on the range of values.
    
    If all values are very small (< 0.01 seconds), convert to milliseconds
    to show 4-5 significant figures.
    """
    if not values:
        return "s"
    
    # Filter out None values
    valid_values = [v for v in values if v is not None and v > 0]
    if not valid_values:
        return "s"
    
    max_value = max(valid_values)
    min_value = min(valid_values)
    
    # If all values are less than 0.01 seconds, use milliseconds
    # This ensures we get 4-5 significant figures
    if max_value < 0.01:
        return "ms"
    elif max_value < 1:
        # Mixed range - check if most values are small
        small_count = sum(1 for v in valid_values if v < 0.01)
        if small_count / len(valid_values) > 0.7:  # More than 70% are small
            return "ms"
        return "s"
    else:
        return "s"


def format_time_value(value: float, unit: str = "s") -> str:
    """Format a single time value with the specified unit."""
    if unit == "ms":
        ms_value = value * 1000
        if ms_value >= 1000:
            return f"{ms_value/1000:.2f}s"
        elif ms_value >= 10:
            return f"{ms_value:.1f}ms"
        else:
            return f"{ms_value:.2f}ms"
    elif unit == "μs":
        us_value = value * 1000000
        if us_value >= 1000:
            return f"{us_value/1000:.2f}ms"
        elif us_value >= 10:
            return f"{us_value:.1f}μs"
        else:
            return f"{us_value:.2f}μs"
    else:  # seconds
        if value >= 3600:
            return f"{value/3600:.2f}h"
        elif value >= 60:
            return f"{value/60:.2f}m"
        elif value >= 1:
            if value >= 10:
                return f"{value:.1f}s"
            else:
                return f"{value:.2f}s"
        elif value >= 0.001:
            ms_value = value * 1000
            if ms_value >= 10:
                return f"{ms_value:.1f}ms"
            else:
                return f"{ms_value:.2f}ms"
        else:
            us_value = value * 1000000
            if us_value >= 10:
                return f"{us_value:.1f}μs"
            else:
                return f"{us_value:.2f}μs"


def calculate_statistics(results: List[Dict], metric: str) -> Dict:
    """Calculate statistical summary for a metric."""
    values = [r[metric] for r in results if metric in r and r[metric] is not None]
    if not values:
        return {}
    
    return {
        'mean': mean(values),
        'median': median(values),
        'std': stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def group_results(results: List[Dict], group_by: List[str]) -> Dict[Tuple, List[Dict]]:
    """Group results by specified fields."""
    grouped = defaultdict(list)
    for result in results:
        key = tuple(result.get(field) for field in group_by)
        grouped[key].append(result)
    return dict(grouped)


def plot_throughput_vs_size(results: List[Dict], output_path: Path):
    """Plot throughput vs file size with human-readable formatting."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    format_colors = {'csv': '#2ecc71', 'json': '#e74c3c', 'xml': '#f39c12'}
    format_markers = {'csv': 'o', 'json': 's', 'xml': '^'}
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results 
                         if r.get('file_format') == format_type 
                         and r.get('test_scenario') == 'happy_path']
        if not format_results:
            continue
        
        # Group by file size
        grouped = group_results(format_results, ['file_size_mb'])
        sizes = sorted([k[0] for k in grouped.keys() if k[0] is not None])
        throughputs = []
        throughput_stds = []
        
        for size in sizes:
            size_results = grouped[(size,)]
            stats = calculate_statistics(size_results, 'records_per_second')
            if stats:
                throughputs.append(stats['mean'])
                throughput_stds.append(stats['std'])
            else:
                throughputs.append(0)
                throughput_stds.append(0)
        
        ax.errorbar(
            sizes, throughputs, yerr=throughput_stds,
            label=format_type.upper(), 
            marker=format_markers[format_type],
            color=format_colors[format_type],
            capsize=5,
            linewidth=2,
            markersize=8
        )
    
    ax.set_xlabel('File Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (Records/Second)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput vs File Size by Format (Scalability Analysis)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    # Format y-axis with human-readable numbers
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: format_large_number(x)
    ))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{x:.1f}" if x < 1 else f"{int(x)}"
    ))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved throughput plot to {output_path}")
    plt.close()


def plot_memory_efficiency(results: List[Dict], output_path: Path):
    """Plot memory efficiency with analysis of scaling behavior."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    format_colors = {'csv': '#2ecc71', 'json': '#e74c3c', 'xml': '#f39c12'}
    format_markers = {'csv': 'o', 'json': 's', 'xml': '^'}
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results 
                         if r.get('file_format') == format_type 
                         and r.get('test_scenario') == 'happy_path']
        if not format_results:
            continue
        
        # Extract memory efficiency and file sizes
        sizes = []
        efficiencies = []
        
        for result in format_results:
            if result.get('memory_efficiency_mb_per_record') is not None:
                sizes.append(result.get('file_size_mb'))
                efficiencies.append(result.get('memory_efficiency_mb_per_record'))
        
        if sizes and efficiencies:
            ax.plot(
                sizes, efficiencies,
                label=format_type.upper(),
                marker=format_markers[format_type],
                color=format_colors[format_type],
                linewidth=2,
                markersize=8
            )
    
    ax.set_xlabel('File Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory Efficiency (MB per Record)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Efficiency vs File Size (Memory Complexity Validation)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Format axes
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{x:.1f}" if x < 1 else f"{int(x)}"
    ))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{x:.3f}" if x < 0.001 else f"{x:.2f}"
    ))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved memory efficiency plot to {output_path}")
    plt.close()


def plot_processing_time_loglog(results: List[Dict], output_path: Path):
    """Plot processing time vs file size on log-log scale for complexity analysis."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    format_colors = {'csv': '#2ecc71', 'json': '#e74c3c', 'xml': '#f39c12'}
    format_markers = {'csv': 'o', 'json': 's', 'xml': '^'}
    
    all_times = []
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results 
                         if r.get('file_format') == format_type 
                         and r.get('test_scenario') == 'happy_path']
        if not format_results:
            continue
        
        # Group by file size
        grouped = group_results(format_results, ['file_size_mb'])
        sizes = sorted([k[0] for k in grouped.keys() if k[0] is not None])
        times = []
        time_stds = []
        
        for size in sizes:
            size_results = grouped[(size,)]
            stats = calculate_statistics(size_results, 'processing_time_seconds')
            if stats:
                times.append(stats['mean'])
                time_stds.append(stats['std'])
                all_times.append(stats['mean'])
            else:
                times.append(0)
                time_stds.append(0)
        
        ax.errorbar(
            sizes, times, yerr=time_stds,
            label=format_type.upper(),
            marker=format_markers[format_type],
            color=format_colors[format_type],
            capsize=5,
            linewidth=2,
            markersize=8
        )
    
    # Determine appropriate unit based on all time values
    time_unit = format_time_adaptive(all_times)
    if time_unit == "ms":
        # Convert all times to milliseconds for display
        ylabel = 'Processing Time (Milliseconds)'
        # Update y-axis formatter to show milliseconds
        def format_ms(x, p):
            if x <= 0:
                return "0"
            return format_time_value(x, "ms")
        y_formatter = ticker.FuncFormatter(format_ms)
    else:
        ylabel = 'Processing Time (Seconds)'
        y_formatter = ticker.FuncFormatter(lambda x, p: format_time(x))
    
    ax.set_xlabel('File Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title('Processing Time vs File Size (Log-Log) (Complexity Analysis)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Format axes
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{x:.1f}" if x < 1 else f"{int(x)}"
    ))
    ax.yaxis.set_major_formatter(y_formatter)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved processing time plot to {output_path}")
    plt.close()


def plot_format_comparison(results: List[Dict], output_path: Path):
    """Compare formats side-by-side for key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    format_colors = {'csv': '#2ecc71', 'json': '#e74c3c', 'xml': '#f39c12'}
    
    # Filter happy path results
    happy_path_results = [r for r in results if r.get('test_scenario') == 'happy_path']
    
    # Calculate averages per format
    formats = ['csv', 'json', 'xml']
    metrics = {
        'throughput': 'records_per_second',
        'memory_efficiency': 'memory_efficiency_mb_per_record',
        'processing_time': 'processing_time_seconds'
    }
    
    for idx, (metric_name, metric_key) in enumerate(metrics.items()):
        ax = axes[idx]
        format_values = []
        format_labels = []
        
        for format_type in formats:
            format_results = [r for r in happy_path_results 
                            if r.get('file_format') == format_type]
            if not format_results:
                continue
            
            values = [r[metric_key] for r in format_results 
                     if r.get(metric_key) is not None]
            if values:
                format_values.append(mean(values))
                format_labels.append(format_type.upper())
        
        if format_values:
            bars = ax.bar(format_labels, format_values, 
                         color=[format_colors[f.lower()] for f in format_labels],
                         alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if metric_name == 'throughput':
                    label = format_large_number(height)
                elif metric_name == 'memory_efficiency':
                    label = f"{height:.4f}"
                else:  # processing_time
                    label = format_time(height)
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontweight='bold')
        
        # Set labels and title
        if metric_name == 'throughput':
            ax.set_ylabel('Average Throughput (Records/Second)', fontsize=11, fontweight='bold')
            ax.set_title('Average Throughput by Format', fontsize=12, fontweight='bold')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, p: format_large_number(x)
            ))
        elif metric_name == 'memory_efficiency':
            ax.set_ylabel('Average Memory Efficiency (MB/Record)', fontsize=11, fontweight='bold')
            ax.set_title('Average Memory Efficiency by Format', fontsize=12, fontweight='bold')
        else:  # processing_time
            # Determine if we should use milliseconds
            processing_times = [r.get('processing_time_seconds', 0) for r in happy_path_results 
                               if r.get('processing_time_seconds') is not None]
            time_unit = format_time_adaptive(processing_times) if processing_times else "s"
            
            if time_unit == "ms":
                ax.set_ylabel('Average Processing Time (Milliseconds)', fontsize=11, fontweight='bold')
                # Convert values to milliseconds for display
                format_values_ms = [v * 1000 if v else 0 for v in format_values]
                bars = ax.bar(format_labels, format_values_ms, 
                             color=[format_colors[f.lower()] for f in format_labels],
                             alpha=0.7, edgecolor='black', linewidth=1.5)
                # Update labels with milliseconds
                for bar in bars:
                    height = bar.get_height()
                    label = format_time_value(height / 1000, "ms") if height > 0 else "0ms"
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontweight='bold')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, p: format_time_value(x / 1000, "ms") if x > 0 else "0ms"
                ))
            else:
                ax.set_ylabel('Average Processing Time (Seconds)', fontsize=11, fontweight='bold')
                # Re-plot bars with original values
                bars = ax.bar(format_labels, format_values, 
                             color=[format_colors[f.lower()] for f in format_labels],
                             alpha=0.7, edgecolor='black', linewidth=1.5)
                # Update labels
                for bar in bars:
                    height = bar.get_height()
                    label = format_time(height) if height > 0 else "0s"
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontweight='bold')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, p: format_time(x)
                ))
            ax.set_title('Average Processing Time by Format', fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved format comparison plot to {output_path}")
    plt.close()


def plot_scalability_analysis(results: List[Dict], output_path: Path):
    """Analyze scalability by plotting throughput vs number of records."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    format_colors = {'csv': '#2ecc71', 'json': '#e74c3c', 'xml': '#f39c12'}
    format_markers = {'csv': 'o', 'json': 's', 'xml': '^'}
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results 
                         if r.get('file_format') == format_type 
                         and r.get('test_scenario') == 'happy_path']
        if not format_results:
            continue
        
        # Extract records and throughput
        records = []
        throughputs = []
        
        for result in format_results:
            if (result.get('num_records') is not None and 
                result.get('records_per_second') is not None):
                records.append(result.get('num_records'))
                throughputs.append(result.get('records_per_second'))
        
        if records and throughputs:
            # Sort by records
            sorted_data = sorted(zip(records, throughputs))
            records, throughputs = zip(*sorted_data)
            
            ax.plot(
                records, throughputs,
                label=format_type.upper(),
                marker=format_markers[format_type],
                color=format_colors[format_type],
                linewidth=2,
                markersize=8
            )
    
    ax.set_xlabel('Number of Records', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (Records/Second)', fontsize=12, fontweight='bold')
    ax.set_title('Scalability Analysis: Throughput vs Dataset Size (Batch-Oriented Throughput)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    # Format axes
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: format_large_number(x)
    ))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: format_large_number(x)
    ))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved scalability plot to {output_path}")
    plt.close()


def plot_memory_usage(results: List[Dict], output_path: Path):
    """Plot peak memory usage vs file size."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    format_colors = {'csv': '#2ecc71', 'json': '#e74c3c', 'xml': '#f39c12'}
    format_markers = {'csv': 'o', 'json': 's', 'xml': '^'}
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results 
                         if r.get('file_format') == format_type 
                         and r.get('test_scenario') == 'happy_path']
        if not format_results:
            continue
        
        # Group by file size
        grouped = group_results(format_results, ['file_size_mb'])
        sizes = sorted([k[0] for k in grouped.keys() if k[0] is not None])
        memories = []
        memory_stds = []
        
        for size in sizes:
            size_results = grouped[(size,)]
            stats = calculate_statistics(size_results, 'peak_memory_mb')
            if stats:
                memories.append(stats['mean'])
                memory_stds.append(stats['std'])
            else:
                memories.append(0)
                memory_stds.append(0)
        
        ax.errorbar(
            sizes, memories, yerr=memory_stds,
            label=format_type.upper(),
            marker=format_markers[format_type],
            color=format_colors[format_type],
            capsize=5,
            linewidth=2,
            markersize=8
        )
    
    ax.set_xlabel('File Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage vs File Size (Streaming Architecture Validation)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    # Format axes
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{x:.1f}" if x < 1 else f"{int(x)}"
    ))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{int(x)}"
    ))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved memory usage plot to {output_path}")
    plt.close()


def plot_batch_efficiency(results: List[Dict], output_path: Path):
    """Analyze batch processing efficiency."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    format_colors = {'csv': '#2ecc71', 'json': '#e74c3c', 'xml': '#f39c12'}
    format_markers = {'csv': 'o', 'json': 's', 'xml': '^'}
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results 
                         if r.get('file_format') == format_type 
                         and r.get('test_scenario') == 'happy_path']
        if not format_results:
            continue
        
        # Extract batch counts and throughput
        batches = []
        throughputs = []
        file_sizes = []
        
        for result in format_results:
            if (result.get('num_batches') is not None and 
                result.get('records_per_second') is not None):
                batches.append(result.get('num_batches'))
                throughputs.append(result.get('records_per_second'))
                file_sizes.append(result.get('file_size_mb', 1))
        
        if batches and throughputs:
            # Use file size for bubble size
            ax.scatter(
                batches, throughputs,
                s=[s*50 for s in file_sizes],  # Scale bubble size
                label=format_type.upper(),
                marker=format_markers[format_type],
                color=format_colors[format_type],
                alpha=0.6,
                edgecolors='black',
                linewidths=1
            )
    
    ax.set_xlabel('Number of Batches', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (Records/Second)', fontsize=12, fontweight='bold')
    ax.set_title('Batch Processing Efficiency (Bubble size = File Size)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: format_large_number(x)
    ))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved batch efficiency plot to {output_path}")
    plt.close()


def plot_success_rate(results: List[Dict], output_path: Path):
    """Plot success rate vs file size."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    format_colors = {'csv': '#2ecc71', 'json': '#e74c3c', 'xml': '#f39c12'}
    format_markers = {'csv': 'o', 'json': 's', 'xml': '^'}
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results 
                         if r.get('file_format') == format_type 
                         and r.get('test_scenario') == 'happy_path']
        if not format_results:
            continue
        
        # Group by file size
        grouped = group_results(format_results, ['file_size_mb'])
        sizes = sorted([k[0] for k in grouped.keys() if k[0] is not None])
        success_rates = []
        
        for size in sizes:
            size_results = grouped[(size,)]
            stats = calculate_statistics(size_results, 'success_rate')
            if stats:
                success_rates.append(stats['mean'] * 100)  # Convert to percentage
            else:
                success_rates.append(0)
        
        ax.plot(
            sizes, success_rates,
            label=format_type.upper(),
            marker=format_markers[format_type],
            color=format_colors[format_type],
            linewidth=2,
            markersize=8
        )
    
    ax.set_xlabel('File Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Data Quality: Success Rate vs File Size', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_ylim([95, 105])  # Focus on high success rates
    
    # Format axes
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{x:.1f}" if x < 1 else f"{int(x)}"
    ))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{int(x)}%"
    ))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved success rate plot to {output_path}")
    plt.close()


def plot_throughput_heatmap(results: List[Dict], output_path: Path):
    """Create heatmap of throughput by format and file size."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    # Prepare data
    happy_path_results = [r for r in results if r.get('test_scenario') == 'happy_path']
    
    # Get unique file sizes and formats
    file_sizes = sorted(set([r.get('file_size_mb') for r in happy_path_results 
                             if r.get('file_size_mb') is not None]))
    formats = ['csv', 'json', 'xml']
    
    # Create matrix
    heatmap_data = []
    for format_type in formats:
        row = []
        for size in file_sizes:
            format_size_results = [r for r in happy_path_results 
                                  if r.get('file_format') == format_type 
                                  and r.get('file_size_mb') == size]
            if format_size_results:
                throughputs = [r.get('records_per_second') for r in format_size_results 
                             if r.get('records_per_second') is not None]
                if throughputs:
                    row.append(mean(throughputs))
                else:
                    row.append(0)
            else:
                row.append(0)
        heatmap_data.append(row)
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(file_sizes)))
    ax.set_xticklabels([f"{s:.1f}MB" if s < 1 else f"{int(s)}MB" for s in file_sizes])
    ax.set_yticks(range(len(formats)))
    ax.set_yticklabels([f.upper() for f in formats])
    
    # Add text annotations
    for i in range(len(formats)):
        for j in range(len(file_sizes)):
            value = heatmap_data[i][j]
            text = ax.text(j, i, format_large_number(value),
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Throughput (Records/Second)', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('File Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Format', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Heatmap: Records/Second by Format and File Size', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved throughput heatmap to {output_path}")
    plt.close()


def analyze_data_generation_differences(results: List[Dict]) -> Dict:
    """Analyze differences in data generation (records per MB) across formats."""
    analysis = {}
    
    happy_path_results = [r for r in results if r.get('test_scenario') == 'happy_path']
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in happy_path_results 
                         if r.get('file_format') == format_type]
        
        if not format_results:
            continue
        
        records_per_mb = []
        for result in format_results:
            if (result.get('file_size_mb') and result.get('num_records') and 
                result.get('file_size_mb') > 0):
                records_per_mb.append(result.get('num_records') / result.get('file_size_mb'))
        
        if records_per_mb:
            analysis[format_type] = {
                'mean': mean(records_per_mb),
                'std': stdev(records_per_mb) if len(records_per_mb) > 1 else 0,
                'min': min(records_per_mb),
                'max': max(records_per_mb)
            }
    
    return analysis


def analyze_performance_scaling(results: List[Dict]) -> Dict:
    """Analyze how performance scales with file size (linear, sublinear, superlinear)."""
    analysis = {}
    
    happy_path_results = [r for r in results if r.get('test_scenario') == 'happy_path']
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in happy_path_results 
                         if r.get('file_format') == format_type]
        
        if not format_results or len(format_results) < 2:
            continue
        
        # Sort by file size
        sorted_results = sorted(format_results, key=lambda x: x.get('file_size_mb', 0))
        
        # Calculate scaling factors
        sizes = [r.get('file_size_mb') for r in sorted_results if r.get('file_size_mb')]
        throughputs = [r.get('records_per_second') for r in sorted_results 
                      if r.get('records_per_second')]
        processing_times = [r.get('processing_time_seconds') for r in sorted_results 
                           if r.get('processing_time_seconds')]
        
        if len(sizes) >= 2 and len(throughputs) >= 2:
            # Calculate throughput scaling (how throughput changes with size)
            size_ratio = sizes[-1] / sizes[0] if sizes[0] > 0 else 1
            throughput_ratio = throughputs[-1] / throughputs[0] if throughputs[0] > 0 else 1
            
            # Calculate time scaling (how time changes with size)
            if len(processing_times) >= 2:
                time_ratio = processing_times[-1] / processing_times[0] if processing_times[0] > 0 else 1
            else:
                time_ratio = None
            
            analysis[format_type] = {
                'size_increase': size_ratio,
                'throughput_change': throughput_ratio,
                'time_increase': time_ratio,
                'throughput_scaling': 'improving' if throughput_ratio > 1.1 else 
                                     ('degrading' if throughput_ratio < 0.9 else 'stable'),
                'time_scaling': 'sublinear' if time_ratio and time_ratio < size_ratio * 0.8 else
                               ('superlinear' if time_ratio and time_ratio > size_ratio * 1.2 else 'linear')
            }
    
    return analysis


def generate_summary_statistics(results: List[Dict], output_path: Path):
    """Generate comprehensive summary statistics CSV."""
    happy_path_results = [r for r in results if r.get('test_scenario') == 'happy_path']
    
    summary_data = []
    
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in happy_path_results 
                         if r.get('file_format') == format_type]
        
        if not format_results:
            continue
        
        # Calculate statistics
        summary_data.append({
            'format': format_type.upper(),
            'avg_throughput_records_per_sec': mean([r.get('records_per_second', 0) 
                                                   for r in format_results 
                                                   if r.get('records_per_second')]),
            'avg_memory_efficiency_mb_per_record': mean([r.get('memory_efficiency_mb_per_record', 0) 
                                                        for r in format_results 
                                                        if r.get('memory_efficiency_mb_per_record')]),
            'avg_processing_time_seconds': mean([r.get('processing_time_seconds', 0) 
                                                for r in format_results 
                                                if r.get('processing_time_seconds')]),
            'avg_peak_memory_mb': mean([r.get('peak_memory_mb', 0) 
                                      for r in format_results 
                                      if r.get('peak_memory_mb')]),
            'avg_success_rate': mean([r.get('success_rate', 0) 
                                     for r in format_results 
                                     if r.get('success_rate')]) * 100,
            'avg_records_per_mb': mean([r.get('num_records', 0) / r.get('file_size_mb', 1) 
                                       for r in format_results 
                                       if r.get('file_size_mb') and r.get('num_records')])
        })
    
    # Write to CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved summary statistics to {output_path}")


def main():
    """Main entry point for visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize benchmark results from CSV with enhanced analysis'
    )
    parser.add_argument(
        'results_file',
        type=Path,
        help='Path to benchmark results CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots (default: same as results file parent)'
    )
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        logger.error(f"Results file not found: {args.results_file}")
        return
    
    output_dir = args.output_dir or args.results_file.parent / "benchmark_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logger.info(f"Loading results from {args.results_file}")
    results = load_csv_results(args.results_file)
    logger.info(f"Loaded {len(results)} benchmark results")
    
    # Generate plots
    logger.info("Generating visualizations...")
    plot_throughput_vs_size(results, output_dir / '1_throughput_vs_size.png')
    plot_memory_efficiency(results, output_dir / '2_memory_efficiency.png')
    plot_processing_time_loglog(results, output_dir / '3_processing_time_loglog.png')
    plot_format_comparison(results, output_dir / '4_format_comparison.png')
    plot_scalability_analysis(results, output_dir / '5_scalability_analysis.png')
    plot_memory_usage(results, output_dir / '6_memory_usage.png')
    plot_batch_efficiency(results, output_dir / '7_batch_efficiency.png')
    plot_success_rate(results, output_dir / '8_success_rate.png')
    plot_throughput_heatmap(results, output_dir / '9_throughput_heatmap.png')
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    generate_summary_statistics(results, output_dir / 'summary_statistics.csv')
    
    # Generate analysis reports
    logger.info("Analyzing performance differences...")
    data_gen_analysis = analyze_data_generation_differences(results)
    scaling_analysis = analyze_performance_scaling(results)
    
    # Print analysis
    logger.info("\n" + "="*80)
    logger.info("DATA GENERATION ANALYSIS (Records per MB)")
    logger.info("="*80)
    for format_type, stats in data_gen_analysis.items():
        logger.info(f"{format_type.upper()}: {stats['mean']:.1f} ± {stats['std']:.1f} records/MB "
                   f"(range: {stats['min']:.1f} - {stats['max']:.1f})")
    
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE SCALING ANALYSIS")
    logger.info("="*80)
    for format_type, stats in scaling_analysis.items():
        logger.info(f"{format_type.upper()}:")
        logger.info(f"  Throughput scaling: {stats['throughput_scaling']} "
                   f"({stats['throughput_change']:.2f}x change for {stats['size_increase']:.2f}x size increase)")
        if stats.get('time_increase'):
            logger.info(f"  Time scaling: {stats['time_scaling']} "
                       f"({stats['time_increase']:.2f}x time increase for {stats['size_increase']:.2f}x size increase)")
    
    # Generate adaptive chunking comparison if data available
    adaptive_results = [r for r in results if r.get('adaptive_chunking_enabled') is not None]
    if adaptive_results:
        logger.info("\n" + "="*80)
        logger.info("ADAPTIVE CHUNKING ANALYSIS")
        logger.info("="*80)
        
        for format_type in ['csv', 'json']:
            with_chunking = [r for r in adaptive_results 
                           if r.get('file_format') == format_type 
                           and r.get('adaptive_chunking_enabled') is True]
            without_chunking = [r for r in adaptive_results 
                              if r.get('file_format') == format_type 
                              and r.get('adaptive_chunking_enabled') is False]
            
            if with_chunking and without_chunking:
                with_throughput = mean([r.get('records_per_second', 0) for r in with_chunking 
                                       if r.get('records_per_second')])
                without_throughput = mean([r.get('records_per_second', 0) for r in without_chunking 
                                          if r.get('records_per_second')])
                improvement = ((with_throughput - without_throughput) / without_throughput * 100) if without_throughput > 0 else 0
                
                logger.info(f"{format_type.upper()}:")
                logger.info(f"  With adaptive chunking: {with_throughput:.0f} rec/s")
                logger.info(f"  Without adaptive chunking: {without_throughput:.0f} rec/s")
                logger.info(f"  Improvement: {improvement:+.1f}%")
    
    logger.info(f"\nVisualization complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
