import { Suspense } from 'react';
import { api } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { TimeRangeSelector } from '@/components/dashboard/time-range-selector';
import type { TimeRange } from '@/types/api';

interface RedactionLogsContentProps {
  searchParams: {
    timeRange?: TimeRange;
    limit?: string;
    offset?: string;
    field_name?: string;
    rule_triggered?: string;
  };
}

async function RedactionLogsContent({ searchParams }: RedactionLogsContentProps) {
  const limit = parseInt(searchParams.limit || '100');
  const offset = parseInt(searchParams.offset || '0');
  const timeRange = (searchParams.timeRange as TimeRange) || '24h';

  let redactionLogs;
  let error: string | null = null;

  try {
    redactionLogs = await api.audit.redactionLogs({
      time_range: timeRange,
      limit,
      offset,
      field_name: searchParams.field_name || undefined,
      rule_triggered: searchParams.rule_triggered || undefined,
      sort_by: 'timestamp',
      sort_order: 'DESC',
    });
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to fetch redaction logs';
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle>Error Loading Redaction Logs</CardTitle>
          <CardDescription>Unable to fetch redaction logs from the API</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-destructive">{error}</p>
        </CardContent>
      </Card>
    );
  }

  if (!redactionLogs) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-muted-foreground">No redaction logs available</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Redaction Logs</CardTitle>
          <CardDescription>
            PII redaction events and compliance tracking. Total: {redactionLogs.pagination.total}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {redactionLogs.logs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No redaction logs found for the selected filters.
            </div>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Field</TableHead>
                    <TableHead>Rule</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead>Record ID</TableHead>
                    <TableHead>Hash</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {redactionLogs.logs.map((log) => (
                    <TableRow key={log.log_id}>
                      <TableCell className="font-mono text-xs">
                        {new Date(log.timestamp).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{log.field_name}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary" className="font-mono text-xs">
                          {log.rule_triggered}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {log.source_adapter ? (
                          <Badge variant="outline" className="font-normal">
                            {log.source_adapter}
                          </Badge>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {log.record_id ? (
                          <span className="text-xs">{log.record_id}</span>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {log.original_hash.substring(0, 16)}...
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}

          {/* Pagination */}
          {redactionLogs.pagination.total > 0 && (
            <div className="flex items-center justify-between mt-4">
              <div className="text-sm text-muted-foreground">
                Showing {offset + 1} to {Math.min(offset + limit, redactionLogs.pagination.total)}{' '}
                of {redactionLogs.pagination.total} entries
              </div>
              <div className="flex gap-2">
                {redactionLogs.pagination.has_previous && (
                  <a href={`?offset=${Math.max(0, offset - limit)}&limit=${limit}&timeRange=${timeRange}`}>
                    <button className="px-3 py-1 text-sm border rounded hover:bg-muted">
                      Previous
                    </button>
                  </a>
                )}
                {redactionLogs.pagination.has_next && (
                  <a href={`?offset=${offset + limit}&limit=${limit}&timeRange=${timeRange}`}>
                    <button className="px-3 py-1 text-sm border rounded hover:bg-muted">
                      Next
                    </button>
                  </a>
                )}
              </div>
            </div>
          )}

          {/* Summary */}
          {redactionLogs.summary && (
            <div className="mt-6 p-4 bg-muted/50 rounded-md">
              <h3 className="text-sm font-semibold mb-3">Summary</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="font-medium mb-1">Total Redactions</p>
                  <p className="text-2xl font-bold">{redactionLogs.summary.total.toLocaleString()}</p>
                </div>
                <div>
                  <p className="font-medium mb-1">Fields Affected</p>
                  <p className="text-lg">
                    {Object.keys(redactionLogs.summary.by_field).length}
                  </p>
                </div>
                <div>
                  <p className="font-medium mb-1">Rules Triggered</p>
                  <p className="text-lg">
                    {Object.keys(redactionLogs.summary.by_rule).length}
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default async function RedactionsPage({
  searchParams,
}: {
  searchParams: {
    timeRange?: TimeRange;
    limit?: string;
    offset?: string;
    field_name?: string;
    rule_triggered?: string;
  };
}) {
  const timeRange = (searchParams.timeRange as TimeRange) || '24h';

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">Redaction Logs</h2>
          <p className="text-sm sm:text-base text-muted-foreground mt-1">
            Track PII redactions and compliance events
          </p>
        </div>
        <Suspense fallback={<div className="w-full sm:w-[180px] h-10 bg-muted animate-pulse rounded-md" />}>
          <TimeRangeSelector defaultValue={timeRange} />
        </Suspense>
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
        <RedactionLogsContent searchParams={searchParams} />
      </Suspense>
    </div>
  );
}

