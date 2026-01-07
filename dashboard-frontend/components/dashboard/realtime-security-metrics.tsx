'use client';

import { useWebSocket } from '@/hooks/use-websocket';
import { MetricsCard } from '@/components/dashboard/metrics-card';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Wifi, WifiOff } from 'lucide-react';
import { TimeSeriesChart } from '@/components/dashboard/time-series-chart';
import { DistributionChart } from '@/components/dashboard/distribution-chart';
import type { SecurityMetrics, TimeRange } from '@/types/api';

interface RealtimeSecurityMetricsProps {
  initialMetrics: SecurityMetrics;
  timeRange: TimeRange;
}

export function RealtimeSecurityMetrics({
  initialMetrics,
  timeRange,
}: RealtimeSecurityMetricsProps) {
  const {
    isConnected,
    securityMetrics,
  } = useWebSocket(timeRange);

  // Use WebSocket data if available, otherwise fall back to initial metrics
  const metrics = securityMetrics || initialMetrics;

  // Prepare chart data
  const redactionTrendData = metrics.redactions.trend.map((point) => ({
    date: point.date,
    value: point.count,
  }));

  const redactionByRuleData = Object.entries(metrics.redactions.by_rule).map(([name, value]) => ({
    name,
    value,
  }));

  const redactionByAdapterData = Object.entries(metrics.redactions.by_adapter).map(
    ([name, value]) => ({
      name,
      value,
    })
  );

  const auditBySeverityData = Object.entries(metrics.audit_events.by_severity).map(
    ([name, value]) => ({
      name,
      value,
    })
  );

  const auditByTypeData = Object.entries(metrics.audit_events.by_type).map(([name, value]) => ({
    name,
    value,
  }));

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

      {/* Summary Cards */}
      <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <MetricsCard
          title="Total Redactions"
          value={metrics.redactions.total}
          description="PII redactions performed"
          variant="default"
          formatType="number"
        />

        <MetricsCard
          title="Audit Events"
          value={metrics.audit_events.total}
          description="Total security audit events"
          variant="default"
          formatType="number"
        />

        <MetricsCard
          title="Redaction Rules"
          value={Object.keys(metrics.redactions.by_rule).length}
          description="Active redaction rules"
          variant="default"
          formatType="number"
        />

        <MetricsCard
          title="Source Adapters"
          value={Object.keys(metrics.redactions.by_adapter).length}
          description="Adapters with redactions"
          variant="default"
          formatType="number"
        />
      </div>

      {/* Redaction Charts */}
      <div className="grid gap-4 md:grid-cols-2">
        <TimeSeriesChart
          title="Redaction Trend"
          description="Redactions over time"
          data={redactionTrendData}
          dataKey="value"
          color="#ef4444"
          unit=" redactions"
        />

        <DistributionChart
          title="Redactions by Rule"
          description="Distribution of redactions by rule type"
          data={redactionByRuleData}
          chartType="pie"
        />
      </div>

      {/* Distribution Charts */}
      <div className="grid gap-4 md:grid-cols-2">
        <DistributionChart
          title="Redactions by Adapter"
          description="Redactions per data source adapter"
          data={redactionByAdapterData}
          chartType="bar"
          color="#3b82f6"
        />

        <DistributionChart
          title="Audit Events by Severity"
          description="Distribution of audit events by severity level"
          data={auditBySeverityData}
          chartType="pie"
        />
      </div>

      {/* Audit Event Details */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Audit Event Summary</CardTitle>
            <CardDescription>Breakdown of audit events</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Total Events</span>
                <span className="text-sm font-bold">{metrics.audit_events.total.toLocaleString()}</span>
              </div>
              {Object.entries(metrics.audit_events.by_type).length > 0 && (
                <div className="mt-4">
                  <p className="text-xs font-medium mb-2">By Type:</p>
                  <div className="space-y-1">
                    {Object.entries(metrics.audit_events.by_type)
                      .slice(0, 5)
                      .map(([type, count]) => (
                        <div key={type} className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">{type}</span>
                          <span>{count.toLocaleString()}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <DistributionChart
          title="Audit Events by Type"
          description="Distribution of audit events by event type"
          data={auditByTypeData}
          chartType="bar"
          color="#10b981"
        />
      </div>
    </div>
  );
}

