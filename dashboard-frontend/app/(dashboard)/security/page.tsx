import { Suspense, lazy } from 'react';
import { api } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { TimeRangeSelector } from '@/components/dashboard/time-range-selector';
import type { TimeRange } from '@/types/api';

// Force dynamic rendering - don't statically generate this page
export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Lazy load heavy components for better initial load performance
const RealtimeSecurityMetrics = lazy(() =>
  import('@/components/dashboard/realtime-security-metrics').then((mod) => ({
    default: mod.RealtimeSecurityMetrics,
  }))
);

async function SecurityMetricsContent({ timeRange }: { timeRange: TimeRange }) {
  let metrics;
  let error: string | null = null;

  try {
    metrics = await api.metrics.security(timeRange);
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to fetch security metrics';
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle>Error Loading Security Metrics</CardTitle>
          <CardDescription>Unable to fetch security metrics from the API</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-destructive">{error}</p>
          <p className="text-sm text-muted-foreground mt-2">
            Make sure the backend API is running on{' '}
            {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
          </p>
        </CardContent>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-muted-foreground">No security metrics available</p>
        </CardContent>
      </Card>
    );
  }

  return <RealtimeSecurityMetrics initialMetrics={metrics} timeRange={timeRange} />;
}

export default async function SecurityPage({
  searchParams,
}: {
  searchParams: Promise<{ timeRange?: TimeRange }>;
}) {
  const params = await searchParams;
  const timeRange = (params.timeRange as TimeRange) || '7d';

  return (
    <div className="space-y-4 sm:space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">Security Metrics</h2>
          <p className="text-sm sm:text-base text-muted-foreground mt-1">
            Monitor redactions, audit events, and security compliance
          </p>
        </div>
        <Suspense fallback={<div className="w-full sm:w-[180px] h-10 bg-muted animate-pulse rounded-md" />}>
          <TimeRangeSelector defaultValue={timeRange} />
        </Suspense>
      </div>

      <Suspense
        fallback={
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[...Array(6)].map((_, i) => (
              <Card key={i} className="animate-pulse">
                <CardHeader>
                  <div className="h-4 bg-muted rounded w-24" />
                </CardHeader>
                <CardContent>
                  <div className="h-8 bg-muted rounded w-32 mb-2" />
                  <div className="h-3 bg-muted rounded w-48" />
                </CardContent>
              </Card>
            ))}
          </div>
        }
      >
        <SecurityMetricsContent timeRange={timeRange} />
      </Suspense>
    </div>
  );
}

