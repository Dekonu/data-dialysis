'use client';

import { useWebSocket } from '@/hooks/use-websocket';
import { MetricsCard } from '@/components/dashboard/metrics-card';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Wifi, WifiOff } from 'lucide-react';
import type { PerformanceMetrics, TimeRange } from '@/types/api';

interface RealtimePerformanceMetricsProps {
  initialMetrics: PerformanceMetrics;
  timeRange: TimeRange;
}

export function RealtimePerformanceMetrics({
  initialMetrics,
  timeRange,
}: RealtimePerformanceMetricsProps) {
  const {
    isConnected,
    performanceMetrics,
  } = useWebSocket(timeRange);

  // Use WebSocket data if available, otherwise fall back to initial metrics
  const metrics = performanceMetrics || initialMetrics;

  const formatLatency = (ms: number | null | undefined) => {
    if (ms === null || ms === undefined) return 'N/A';
    if (ms < 1000) return `${ms.toFixed(1)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  };

  const formatThroughput = (value: number) => {
    return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}`;
  };

  const formatMemory = (mb: number | null | undefined) => {
    if (mb === null || mb === undefined) return 'N/A';
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
  };

  return (
    <div className="space-y-6">
      {/* Connection Status Indicator */}
      <div className="flex items-center justify-end">
        <Badge
          variant={isConnected ? 'default' : 'secondary'}
          className="flex items-center gap-2"
        >
          {isConnected ? (
            <>
              <Wifi className="h-3 w-3" />
              <span>Live</span>
            </>
          ) : (
            <>
              <WifiOff className="h-3 w-3" />
              <span>Static</span>
            </>
          )}
        </Badge>
      </div>

      {/* Throughput Metrics */}
      <div>
        <h3 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">Throughput</h3>
        <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          <MetricsCard
            title="Records/Second"
            value={formatThroughput(metrics.throughput.records_per_second)}
            description="Average processing rate"
            variant="default"
            formatType="plain"
          />

          {metrics.throughput.mb_per_second !== null &&
            metrics.throughput.mb_per_second !== undefined && (
              <MetricsCard
                title="Data Throughput"
                value={`${formatThroughput(metrics.throughput.mb_per_second)} MB/s`}
                description="Data processing rate"
                variant="default"
                formatType="plain"
              />
            )}

          {metrics.throughput.peak_records_per_second !== null &&
            metrics.throughput.peak_records_per_second !== undefined && (
              <MetricsCard
                title="Peak Throughput"
                value={`${formatThroughput(metrics.throughput.peak_records_per_second)} rec/s`}
                description="Maximum processing rate"
                variant="default"
                formatType="plain"
              />
            )}
        </div>
      </div>

      {/* Latency Metrics */}
      <div>
        <h3 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">Latency</h3>
        <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          <MetricsCard
            title="Average Latency"
            value={formatLatency(metrics.latency.avg_processing_time_ms)}
            description="Mean time per batch"
            variant="default"
            formatType="plain"
          />

          <MetricsCard
            title="P50 Latency"
            value={formatLatency(metrics.latency.p50_ms)}
            description="Median time per batch"
            variant="default"
            formatType="plain"
          />

          <MetricsCard
            title="P95 Latency"
            value={formatLatency(metrics.latency.p95_ms)}
            description="95th percentile per batch"
            variant="default"
            formatType="plain"
          />

          <MetricsCard
            title="P99 Latency"
            value={formatLatency(metrics.latency.p99_ms)}
            description="99th percentile per batch"
            variant="default"
            formatType="plain"
          />
        </div>
      </div>

      {/* File Processing Metrics */}
      <div>
        <h3 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">File Processing</h3>
        <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          <MetricsCard
            title="Total Files"
            value={metrics.file_processing.total_files}
            description="Files processed"
            variant="default"
            formatType="number"
          />

          {metrics.file_processing.avg_file_size_mb !== null &&
            metrics.file_processing.avg_file_size_mb !== undefined && (
              <MetricsCard
                title="Avg File Size"
                value={`${formatThroughput(metrics.file_processing.avg_file_size_mb)} MB`}
                description="Average file size"
                variant="default"
                formatType="plain"
              />
            )}

          {metrics.file_processing.total_data_processed_mb !== null &&
            metrics.file_processing.total_data_processed_mb !== undefined && (
              <MetricsCard
                title="Total Data Processed"
                value={formatMemory(metrics.file_processing.total_data_processed_mb)}
                description="Total data volume"
                variant="default"
                formatType="plain"
              />
            )}
        </div>
      </div>

      {/* Memory Metrics */}
      <div>
        <h3 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">Memory Usage</h3>
        <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2">
          {metrics.memory.avg_peak_memory_mb !== null &&
            metrics.memory.avg_peak_memory_mb !== undefined && (
              <MetricsCard
                title="Average Peak Memory"
                value={formatMemory(metrics.memory.avg_peak_memory_mb)}
                description="Average peak memory usage"
                variant="default"
                formatType="plain"
              />
            )}

          {metrics.memory.max_peak_memory_mb !== null &&
            metrics.memory.max_peak_memory_mb !== undefined && (
              <MetricsCard
                title="Max Peak Memory"
                value={formatMemory(metrics.memory.max_peak_memory_mb)}
                description="Maximum peak memory usage"
                variant="default"
                formatType="plain"
              />
            )}
        </div>
      </div>

      {/* Detailed Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Throughput Details</CardTitle>
            <CardDescription>Detailed throughput statistics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Records/Second</span>
                <span className="text-sm font-bold">
                  {formatThroughput(metrics.throughput.records_per_second)}
                </span>
              </div>
              {metrics.throughput.mb_per_second !== null &&
                metrics.throughput.mb_per_second !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">MB/Second</span>
                    <span className="text-sm text-muted-foreground">
                      {formatThroughput(metrics.throughput.mb_per_second)}
                    </span>
                  </div>
                )}
              {metrics.throughput.peak_records_per_second !== null &&
                metrics.throughput.peak_records_per_second !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Peak Records/Second</span>
                    <span className="text-sm text-muted-foreground">
                      {formatThroughput(metrics.throughput.peak_records_per_second)}
                    </span>
                  </div>
                )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Latency Percentiles</CardTitle>
            <CardDescription>Batch processing time distribution</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Average</span>
                <span className="text-sm font-bold">
                  {formatLatency(metrics.latency.avg_processing_time_ms)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">P50 (Median)</span>
                <span className="text-sm text-muted-foreground">
                  {formatLatency(metrics.latency.p50_ms)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">P95</span>
                <span className="text-sm text-muted-foreground">
                  {formatLatency(metrics.latency.p95_ms)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">P99</span>
                <span className="text-sm text-muted-foreground">
                  {formatLatency(metrics.latency.p99_ms)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

