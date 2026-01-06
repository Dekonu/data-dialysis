import { api } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export default async function HomePage() {
  let health;
  let error: string | null = null;

  try {
    health = await api.health();
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to fetch health status';
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Dashboard Overview</h2>
        <p className="text-muted-foreground">
          Monitor the health and performance of your Data-Dialysis pipeline
        </p>
      </div>

      {error ? (
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle>Connection Error</CardTitle>
            <CardDescription>Unable to connect to the API</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-destructive">{error}</p>
            <p className="text-sm text-muted-foreground mt-2">
              Make sure the backend API is running on {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
            </p>
          </CardContent>
        </Card>
      ) : health ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle>API Status</CardTitle>
              <CardDescription>Backend API health</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Status</span>
                <Badge
                  variant={
                    health.status === 'healthy'
                      ? 'default'
                      : health.status === 'degraded'
                      ? 'secondary'
                      : 'destructive'
                  }
                >
                  {health.status}
                </Badge>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Database</CardTitle>
              <CardDescription>Database connection status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Status</span>
                  <Badge
                    variant={health.database.status === 'connected' ? 'default' : 'destructive'}
                  >
                    {health.database.status}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Type</span>
                  <span className="text-sm text-muted-foreground">{health.database.type}</span>
                </div>
                {health.database.response_time_ms && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Response Time</span>
                    <span className="text-sm text-muted-foreground">
                      {health.database.response_time_ms.toFixed(2)}ms
                    </span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Version</CardTitle>
              <CardDescription>API version information</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Version</span>
                <span className="text-sm text-muted-foreground">{health.version}</span>
              </div>
              <div className="flex items-center justify-between mt-2">
                <span className="text-sm font-medium">Last Updated</span>
                <span className="text-sm text-muted-foreground">
                  {new Date(health.timestamp).toLocaleString()}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : null}
    </div>
  );
}
