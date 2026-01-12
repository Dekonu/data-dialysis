"""Analyze and visualize benchmark results.

This script processes benchmark results and generates:
1. Statistical summaries
2. Comparative analysis
3. Visualizations (charts, graphs)
4. Academic-quality performance analysis report

Security Impact:
    - Analysis helps identify performance bottlenecks
    - Memory profiling validates resource efficiency
    - Latency analysis ensures acceptable response times

Academic Value:
    - Demonstrates quantitative analysis methodology
    - Provides empirical evidence for performance claims
    - Enables comparative evaluation
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for academic-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results(results_path: Path) -> List[Dict]:
    """Load benchmark results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


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


def plot_throughput_vs_file_size(results: List[Dict], output_path: Path):
    """Plot throughput vs file size for different formats."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by format and file size
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results if r['file_format'] == format_type]
        if not format_results:
            continue
        
        grouped = group_results(format_results, ['file_size_mb'])
        
        sizes = sorted(grouped.keys())
        throughputs = []
        throughput_stds = []
        
        for size in sizes:
            size_results = grouped[size]
            stats = calculate_statistics(size_results, 'records_per_second')
            if stats:
                throughputs.append(stats['mean'])
                throughput_stds.append(stats['std'])
            else:
                throughputs.append(0)
                throughput_stds.append(0)
        
        sizes_list = [s[0] for s in sizes]
        axes[0].errorbar(
            sizes_list, throughputs, yerr=throughput_stds,
            label=format_type.upper(), marker='o', capsize=5
        )
    
    axes[0].set_xlabel('File Size (MB)', fontsize=12)
    axes[0].set_ylabel('Throughput (records/sec)', fontsize=12)
    axes[0].set_title('Throughput vs File Size by Format', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # MB/sec plot
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results if r['file_format'] == format_type]
        if not format_results:
            continue
        
        grouped = group_results(format_results, ['file_size_mb'])
        
        sizes = sorted(grouped.keys())
        mbps = []
        mbps_stds = []
        
        for size in sizes:
            size_results = grouped[size]
            stats = calculate_statistics(size_results, 'mb_per_second')
            if stats:
                mbps.append(stats['mean'])
                mbps_stds.append(stats['std'])
            else:
                mbps.append(0)
                mbps_stds.append(0)
        
        sizes_list = [s[0] for s in sizes]
        axes[1].errorbar(
            sizes_list, mbps, yerr=mbps_stds,
            label=format_type.upper(), marker='s', capsize=5
        )
    
    axes[1].set_xlabel('File Size (MB)', fontsize=12)
    axes[1].set_ylabel('Throughput (MB/sec)', fontsize=12)
    axes[1].set_title('Data Throughput vs File Size by Format', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved throughput plot to {output_path}")
    plt.close()


def plot_memory_usage(results: List[Dict], output_path: Path):
    """Plot memory usage analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Peak memory vs file size
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results if r['file_format'] == format_type]
        if not format_results:
            continue
        
        grouped = group_results(format_results, ['file_size_mb'])
        
        sizes = sorted(grouped.keys())
        peak_memories = []
        peak_memory_stds = []
        
        for size in sizes:
            size_results = grouped[size]
            stats = calculate_statistics(size_results, 'peak_memory_mb')
            if stats:
                peak_memories.append(stats['mean'])
                peak_memory_stds.append(stats['std'])
            else:
                peak_memories.append(0)
                peak_memory_stds.append(0)
        
        sizes_list = [s[0] for s in sizes]
        axes[0].errorbar(
            sizes_list, peak_memories, yerr=peak_memory_stds,
            label=format_type.upper(), marker='o', capsize=5
        )
    
    axes[0].set_xlabel('File Size (MB)', fontsize=12)
    axes[0].set_ylabel('Peak Memory Usage (MB)', fontsize=12)
    axes[0].set_title('Peak Memory Usage vs File Size', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Memory efficiency (MB per record)
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results if r['file_format'] == format_type]
        if not format_results:
            continue
        
        # Calculate memory per record
        efficiencies = []
        file_sizes = []
        
        for result in format_results:
            if result['records_processed'] > 0:
                efficiency = result['peak_memory_mb'] / result['records_processed']
                efficiencies.append(efficiency)
                file_sizes.append(result['file_size_mb'])
        
        if efficiencies:
            axes[1].scatter(file_sizes, efficiencies, label=format_type.upper(), alpha=0.6)
    
    axes[1].set_xlabel('File Size (MB)', fontsize=12)
    axes[1].set_ylabel('Memory per Record (MB)', fontsize=12)
    axes[1].set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved memory plot to {output_path}")
    plt.close()


def plot_latency_analysis(results: List[Dict], output_path: Path):
    """Plot latency percentile analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    percentiles = ['latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms', 'latency_p99_9_ms']
    titles = ['P50 Latency', 'P95 Latency', 'P99 Latency', 'P99.9 Latency']
    
    for idx, (percentile, title) in enumerate(zip(percentiles, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for format_type in ['csv', 'json', 'xml']:
            format_results = [r for r in results if r['file_format'] == format_type]
            if not format_results:
                continue
            
            grouped = group_results(format_results, ['file_size_mb'])
            
            sizes = sorted(grouped.keys())
            latencies = []
            latency_stds = []
            
            for size in sizes:
                size_results = grouped[size]
                stats = calculate_statistics(size_results, percentile)
                if stats:
                    latencies.append(stats['mean'])
                    latency_stds.append(stats['std'])
                else:
                    latencies.append(0)
                    latency_stds.append(0)
            
            sizes_list = [s[0] for s in sizes]
            ax.errorbar(
                sizes_list, latencies, yerr=latency_stds,
                label=format_type.upper(), marker='o', capsize=5
            )
        
        ax.set_xlabel('File Size (MB)', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved latency plot to {output_path}")
    plt.close()


def plot_database_comparison(results: List[Dict], output_path: Path):
    """Compare DuckDB vs PostgreSQL performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    databases = ['duckdb', 'postgresql']
    
    for db_type in databases:
        db_results = [r for r in results if r['database_type'] == db_type]
        if not db_results:
            continue
        
        grouped = group_results(db_results, ['file_size_mb'])
        
        sizes = sorted(grouped.keys())
        throughputs = []
        
        for size in sizes:
            size_results = grouped[size]
            stats = calculate_statistics(size_results, 'records_per_second')
            if stats:
                throughputs.append(stats['mean'])
            else:
                throughputs.append(0)
        
        sizes_list = [s[0] for s in sizes]
        axes[0].plot(sizes_list, throughputs, label=db_type.upper(), marker='o', linewidth=2)
    
    axes[0].set_xlabel('File Size (MB)', fontsize=12)
    axes[0].set_ylabel('Throughput (records/sec)', fontsize=12)
    axes[0].set_title('Database Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Memory comparison
    for db_type in databases:
        db_results = [r for r in results if r['database_type'] == db_type]
        if not db_results:
            continue
        
        grouped = group_results(db_results, ['file_size_mb'])
        
        sizes = sorted(grouped.keys())
        memories = []
        
        for size in sizes:
            size_results = grouped[size]
            stats = calculate_statistics(size_results, 'peak_memory_mb')
            if stats:
                memories.append(stats['mean'])
            else:
                memories.append(0)
        
        sizes_list = [s[0] for s in sizes]
        axes[1].plot(sizes_list, memories, label=db_type.upper(), marker='s', linewidth=2)
    
    axes[1].set_xlabel('File Size (MB)', fontsize=12)
    axes[1].set_ylabel('Peak Memory (MB)', fontsize=12)
    axes[1].set_title('Database Memory Usage Comparison', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved database comparison plot to {output_path}")
    plt.close()


def plot_batch_size_optimization(results: List[Dict], output_path: Path):
    """Analyze batch size optimization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group by batch size
    batch_results = [r for r in results if r['batch_size'] is not None]
    if not batch_results:
        logger.warning("No batch size variation in results")
        return
    
    grouped = group_results(batch_results, ['batch_size'])
    
    batch_sizes = sorted([k[0] for k in grouped.keys() if k[0] is not None])
    throughputs = []
    throughput_stds = []
    
    for batch_size in batch_sizes:
        size_results = grouped[(batch_size,)]
        stats = calculate_statistics(size_results, 'records_per_second')
        if stats:
            throughputs.append(stats['mean'])
            throughput_stds.append(stats['std'])
        else:
            throughputs.append(0)
            throughput_stds.append(0)
    
    ax.errorbar(batch_sizes, throughputs, yerr=throughput_stds, marker='o', capsize=5, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Throughput (records/sec)', fontsize=12)
    ax.set_title('Batch Size Optimization Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved batch size optimization plot to {output_path}")
    plt.close()


def generate_summary_statistics(results: List[Dict]) -> Dict:
    """Generate comprehensive statistical summary."""
    summary = {
        'total_runs': len(results),
        'throughput': calculate_statistics(results, 'records_per_second'),
        'data_throughput': calculate_statistics(results, 'mb_per_second'),
        'peak_memory': calculate_statistics(results, 'peak_memory_mb'),
        'avg_memory': calculate_statistics(results, 'avg_memory_mb'),
        'latency_p50': calculate_statistics(results, 'latency_p50_ms'),
        'latency_p95': calculate_statistics(results, 'latency_p95_ms'),
        'latency_p99': calculate_statistics(results, 'latency_p99_ms'),
        'avg_cpu': calculate_statistics(results, 'avg_cpu_percent'),
        'success_rate': calculate_statistics(results, 'success_rate'),
    }
    
    # Format-specific summaries
    for format_type in ['csv', 'json', 'xml']:
        format_results = [r for r in results if r['file_format'] == format_type]
        if format_results:
            summary[f'{format_type}_throughput'] = calculate_statistics(
                format_results, 'records_per_second'
            )
    
    # Database-specific summaries
    for db_type in ['duckdb', 'postgresql']:
        db_results = [r for r in results if r['database_type'] == db_type]
        if db_results:
            summary[f'{db_type}_throughput'] = calculate_statistics(
                db_results, 'records_per_second'
            )
    
    return summary


def main():
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize benchmark results'
    )
    parser.add_argument(
        'results_file',
        type=Path,
        help='Path to benchmark results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots and analysis (default: same as results file)'
    )
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        logger.error(f"Results file not found: {args.results_file}")
        return
    
    output_dir = args.output_dir or args.results_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logger.info(f"Loading results from {args.results_file}")
    results = load_results(args.results_file)
    logger.info(f"Loaded {len(results)} benchmark results")
    
    # Generate plots
    logger.info("Generating visualizations...")
    plot_throughput_vs_file_size(
        results, output_dir / 'throughput_vs_file_size.png'
    )
    plot_memory_usage(results, output_dir / 'memory_usage.png')
    plot_latency_analysis(results, output_dir / 'latency_analysis.png')
    plot_database_comparison(results, output_dir / 'database_comparison.png')
    plot_batch_size_optimization(results, output_dir / 'batch_size_optimization.png')
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    summary = generate_summary_statistics(results)
    
    # Save summary
    summary_path = output_dir / 'summary_statistics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary statistics to {summary_path}")
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()

