import { Suspense } from 'react';
import { api } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { RealtimeCircuitBreaker } from '@/components/dashboard/realtime-circuit-breaker';

// Force dynamic rendering - don't statically generate this page
export const dynamic = 'force-dynamic';
export const revalidate = 0;

async function CircuitBreakerContent() {
  let status;
  let error: string | null = null;

  try {
    status = await api.circuitBreaker.status();
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to fetch circuit breaker status';
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle>Error Loading Circuit Breaker Status</CardTitle>
          <CardDescription>Unable to fetch circuit breaker status from the API</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-destructive">{error}</p>
        </CardContent>
      </Card>
    );
  }

  if (!status) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-muted-foreground">Circuit breaker status unavailable</p>
        </CardContent>
      </Card>
    );
  }

  // Pass status to real-time component for WebSocket updates
  // Circuit breaker doesn't use time range, but WebSocket hook needs it
  return <RealtimeCircuitBreaker initialStatus={status} timeRange="24h" />;
}

export default async function CircuitBreakerPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Circuit Breaker</h2>
        <p className="text-muted-foreground">
          Monitor circuit breaker status and failure metrics
        </p>
      </div>

      <Suspense
        fallback={
          <Card>
            <CardContent className="pt-6">
              <div className="animate-pulse space-y-4">
                <div className="h-4 bg-muted rounded w-3/4" />
                <div className="h-4 bg-muted rounded w-1/2" />
                <div className="h-4 bg-muted rounded w-2/3" />
              </div>
            </CardContent>
          </Card>
        }
      >
        <CircuitBreakerContent />
      </Suspense>
    </div>
  );
}

