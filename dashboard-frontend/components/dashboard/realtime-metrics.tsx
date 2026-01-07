'use client';

import { useEffect, useState } from 'react';
import { useWebSocket } from '@/hooks/use-websocket';
import { MetricsCard } from '@/components/dashboard/metrics-card';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Wifi, WifiOff } from 'lucide-react';
import type { OverviewMetrics, TimeRange } from '@/types/api';

interface RealtimeMetricsProps {
  initialMetrics: OverviewMetrics;
  timeRange: TimeRange;
}

export function RealtimeMetrics({ initialMetrics, timeRange }: RealtimeMetricsProps) {
  const {
    isConnected,
    overviewMetrics,
  } = useWebSocket(timeRange);

  // Use WebSocket data if available, otherwise fall back to initial metrics
  const metrics = overviewMetrics || initialMetrics;

  const successRate = metrics.ingestions.success_rate;
  const recordSuccessRate =
    metrics.records.total_processed > 0
      ? metrics.records.total_successful / metrics.records.total_processed
      : 0;

  const getIngestionVariant = () => {
    if (metrics.ingestions.total === 0) return 'default';
    if (successRate >= 0.95) return 'success';
    if (successRate >= 0.8) return 'warning';
    return 'destructive';
  };

  const getRecordVariant = () => {
    if (metrics.records.total_processed === 0) return 'default';
    if (recordSuccessRate >= 0.95) return 'success';
    if (recordSuccessRate >= 0.8) return 'warning';
    return 'destructive';
  };

  const getCircuitBreakerVariant = () => {
    if (!metrics.circuit_breaker || !metrics.circuit_breaker.status) return 'default';
    if (metrics.circuit_breaker.status === 'closed') return 'success';
    if (metrics.circuit_breaker.status === 'half_open') return 'warning';
    return 'destructive';
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

      {/* Metrics Cards */}
      <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <MetricsCard
          title="Total Ingestions"
          value={metrics.ingestions.total}
          description={`${metrics.ingestions.successful} successful, ${metrics.ingestions.failed} failed`}
          trend={metrics.ingestions.total > 0 ? successRate : null}
          trendLabel="success rate"
          variant={getIngestionVariant()}
          formatType="number"
        />

        <MetricsCard
          title="Records Processed"
          value={metrics.records.total_processed}
          description={`${metrics.records.total_successful} successful, ${metrics.records.total_failed} failed`}
          trend={metrics.records.total_processed > 0 ? recordSuccessRate : null}
          trendLabel="success rate"
          variant={getRecordVariant()}
          formatType="number"
        />

        <MetricsCard
          title="PII Redactions"
          value={metrics.redactions.total}
          description="Total redactions performed"
          variant="default"
          formatType="number"
        />

        <MetricsCard
          title="Circuit Breaker"
          value={metrics.circuit_breaker?.status || 'N/A'}
          description={
            metrics.circuit_breaker?.failure_rate !== null &&
            metrics.circuit_breaker?.failure_rate !== undefined
              ? `${(metrics.circuit_breaker.failure_rate * 100).toFixed(1)}% failure rate`
              : 'Status unknown'
          }
          variant={getCircuitBreakerVariant()}
          formatType="plain"
        />
      </div>

      {/* Additional metrics section */}
      <div className="grid gap-3 sm:gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Ingestion Metrics</CardTitle>
            <CardDescription>Detailed ingestion statistics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Success Rate</span>
                <span className="text-sm font-bold">
                  {(successRate * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Successful</span>
                <span className="text-sm text-muted-foreground">
                  {metrics.ingestions.successful.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Failed</span>
                <span className="text-sm text-muted-foreground">
                  {metrics.ingestions.failed.toLocaleString()}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Record Processing</CardTitle>
            <CardDescription>Record processing statistics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Success Rate</span>
                <span className="text-sm font-bold">
                  {(recordSuccessRate * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Successful</span>
                <span className="text-sm text-muted-foreground">
                  {metrics.records.total_successful.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Failed</span>
                <span className="text-sm text-muted-foreground">
                  {metrics.records.total_failed.toLocaleString()}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Redaction Summary</CardTitle>
            <CardDescription>PII redaction statistics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Total Redactions</span>
                <span className="text-sm font-bold">
                  {metrics.redactions.total.toLocaleString()}
                </span>
              </div>
              {Object.keys(metrics.redactions.by_field).length > 0 && (
                <div className="mt-4">
                  <p className="text-xs font-medium mb-2">By Field:</p>
                  <div className="space-y-1">
                    {Object.entries(metrics.redactions.by_field)
                      .slice(0, 3)
                      .map(([field, count]) => (
                        <div key={field} className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">{field}</span>
                          <span>{count.toLocaleString()}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

