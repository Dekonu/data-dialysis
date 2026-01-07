'use client';

import { useEffect, useState } from 'react';
import { useWebSocket } from '@/hooks/use-websocket';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from '@/components/ui/table';
import { AlertCircle, CheckCircle2, Wifi, WifiOff } from 'lucide-react';
import type { CircuitBreakerStatus, TimeRange } from '@/types/api';

interface RealtimeCircuitBreakerProps {
  initialStatus: CircuitBreakerStatus;
  timeRange: TimeRange;
}

export function RealtimeCircuitBreaker({
  initialStatus,
  timeRange,
}: RealtimeCircuitBreakerProps) {
  const {
    isConnected,
    circuitBreakerStatus,
  } = useWebSocket(timeRange);

  // Use WebSocket data if available, otherwise fall back to initial status
  const status = circuitBreakerStatus || initialStatus;

  const statusIcon = status.is_open ? (
    <AlertCircle className="h-5 w-5" />
  ) : (
    <CheckCircle2 className="h-5 w-5" />
  );
  const statusText = status.is_open ? 'OPEN' : 'CLOSED';

  // Calculate percentage for gauge
  const failureRatePercent = status.failure_rate ?? 0;
  const thresholdPercent = status.threshold ?? 100;
  const gaugePercentage = Math.min((failureRatePercent / thresholdPercent) * 100, 100);

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

      {/* Status Card */}
      <Card>
        <CardHeader>
          <CardTitle>Circuit Breaker Status</CardTitle>
          <CardDescription>Current circuit breaker state and failure metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div
              className={`flex items-center gap-2 p-4 rounded-lg border-2 ${
                status.is_open
                  ? 'border-destructive bg-destructive/10'
                  : 'border-green-500 bg-green-500/10'
              }`}
            >
              {statusIcon}
              <div>
                <div className="text-sm font-medium text-muted-foreground">Status</div>
                <div className="text-2xl font-bold">{statusText}</div>
              </div>
            </div>

            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Failure Rate</span>
                <span className="text-sm font-bold">
                  {failureRatePercent.toFixed(2)}% / {thresholdPercent.toFixed(2)}%
                </span>
              </div>
              <div className="w-full bg-muted rounded-full h-4">
                <div
                  className={`h-4 rounded-full transition-all ${
                    gaugePercentage >= 100
                      ? 'bg-destructive'
                      : gaugePercentage >= 80
                      ? 'bg-yellow-500'
                      : 'bg-green-500'
                  }`}
                  style={{ width: `${gaugePercentage}%` }}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Statistics Table */}
      <Card>
        <CardHeader>
          <CardTitle>Statistics</CardTitle>
          <CardDescription>Circuit breaker operational metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableBody>
              <TableRow>
                <TableHead className="w-[200px]">Total Processed</TableHead>
                <TableCell>{status.total_processed.toLocaleString()}</TableCell>
              </TableRow>
              <TableRow>
                <TableHead>Total Failures</TableHead>
                <TableCell>
                  <span className={status.total_failures > 0 ? 'text-destructive' : ''}>
                    {status.total_failures.toLocaleString()}
                  </span>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableHead>Failure Rate</TableHead>
                <TableCell>
                  <Badge
                    variant={
                      failureRatePercent >= thresholdPercent
                        ? 'destructive'
                        : failureRatePercent >= thresholdPercent * 0.8
                        ? 'secondary'
                        : 'default'
                    }
                  >
                    {failureRatePercent.toFixed(2)}%
                  </Badge>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableHead>Threshold</TableHead>
                <TableCell>{thresholdPercent.toFixed(2)}%</TableCell>
              </TableRow>
              <TableRow>
                <TableHead>Window Size</TableHead>
                <TableCell>{status.window_size.toLocaleString()}</TableCell>
              </TableRow>
              <TableRow>
                <TableHead>Records in Window</TableHead>
                <TableCell>{status.records_in_window.toLocaleString()}</TableCell>
              </TableRow>
              <TableRow>
                <TableHead>Failures in Window</TableHead>
                <TableCell>
                  <span
                    className={status.failures_in_window > 0 ? 'text-destructive font-medium' : ''}
                  >
                    {status.failures_in_window.toLocaleString()}
                  </span>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableHead>Min Records Before Check</TableHead>
                <TableCell>{status.min_records_before_check.toLocaleString()}</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Status Explanation */}
      <Card>
        <CardHeader>
          <CardTitle>Status Explanation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            <p>
              <strong>CLOSED:</strong> Circuit breaker is allowing requests. System is operating
              normally.
            </p>
            <p>
              <strong>OPEN:</strong> Circuit breaker has opened due to high failure rate. Requests
              are being blocked to prevent further failures.
            </p>
            <p className="text-muted-foreground mt-4">
              The circuit breaker monitors the failure rate over a sliding window of{' '}
              {status.window_size} records. When the failure rate exceeds{' '}
              {thresholdPercent.toFixed(2)}%, the circuit opens to protect the system.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

