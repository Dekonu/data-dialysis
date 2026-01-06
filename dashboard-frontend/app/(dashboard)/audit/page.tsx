import { Suspense } from 'react';
import { api } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { AuditExportButtons } from '@/components/dashboard/audit-export-buttons';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import type { TimeRange } from '@/types/api';

interface AuditLogsContentProps {
  searchParams: {
    timeRange?: TimeRange;
    limit?: string;
    offset?: string;
    severity?: string;
    event_type?: string;
  };
}

async function AuditLogsContent({ searchParams }: AuditLogsContentProps) {
  const limit = parseInt(searchParams.limit || '100');
  const offset = parseInt(searchParams.offset || '0');

  let auditLogs;
  let error: string | null = null;

  try {
    auditLogs = await api.audit.logs({
      limit,
      offset,
      severity: searchParams.severity || undefined,
      event_type: searchParams.event_type || undefined,
      sort_by: 'event_timestamp',
      sort_order: 'DESC',
    });
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to fetch audit logs';
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle>Error Loading Audit Logs</CardTitle>
          <CardDescription>Unable to fetch audit logs from the API</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-destructive">{error}</p>
        </CardContent>
      </Card>
    );
  }

  if (!auditLogs) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-muted-foreground">No audit logs available</p>
        </CardContent>
      </Card>
    );
  }

  const getSeverityVariant = (severity: string | null | undefined) => {
    if (!severity) return 'secondary';
    switch (severity.toUpperCase()) {
      case 'CRITICAL':
        return 'destructive';
      case 'ERROR':
        return 'destructive';
      case 'WARNING':
        return 'secondary';
      default:
        return 'default';
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Audit Logs</CardTitle>
              <CardDescription>
                View system audit trail and event logs. Total: {auditLogs.pagination.total}
              </CardDescription>
            </div>
            <AuditExportButtons />
          </div>
        </CardHeader>
        <CardContent>
          {auditLogs.logs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No audit logs found for the selected filters.
            </div>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Event Type</TableHead>
                    <TableHead>Severity</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead>Table</TableHead>
                    <TableHead>Record ID</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TooltipProvider>
                    {auditLogs.logs.map((log) => {
                      // Get source adapter
                      let source = log.source_adapter;
                      if ((!source || !source.trim()) && log.details) {
                        const details = log.details as Record<string, unknown>;
                        source = 
                          (details?.source_adapter as string) ||
                          (details?.source as string) ||
                          (details?.adapter as string) ||
                          null;
                      }
                      
                      const hasTableInfo = log.table_name && log.table_name.trim();
                      const hasRowCount = log.row_count !== null && log.row_count !== undefined;
                      
                      return (
                        <TableRow key={log.audit_id}>
                          <TableCell className="font-mono text-xs">
                            {new Date(log.event_timestamp).toLocaleString()}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">{log.event_type}</Badge>
                          </TableCell>
                          <TableCell>
                            {log.severity ? (
                              <Badge variant={getSeverityVariant(log.severity)}>
                                {log.severity}
                              </Badge>
                            ) : (
                              <span className="text-xs text-muted-foreground">—</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {source && typeof source === 'string' && source.trim() ? (
                              <Badge variant="outline" className="font-normal">
                                {source.trim()}
                              </Badge>
                            ) : (
                              <span className="text-xs text-muted-foreground">—</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {hasTableInfo ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Badge variant="secondary" className="font-mono text-xs cursor-help">
                                    {log.table_name}
                                  </Badge>
                                </TooltipTrigger>
                                {hasRowCount && (
                                  <TooltipContent>
                                    <p className="text-sm">
                                      <span className="font-semibold">Rows:</span>{' '}
                                      {log.row_count!.toLocaleString()}
                                    </p>
                                  </TooltipContent>
                                )}
                              </Tooltip>
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
                        </TableRow>
                      );
                    })}
                  </TooltipProvider>
                </TableBody>
              </Table>
            </div>
          )}

          {/* Pagination */}
          {auditLogs.pagination.total > 0 && (
            <div className="flex items-center justify-between mt-4">
              <div className="text-sm text-muted-foreground">
                Showing {offset + 1} to {Math.min(offset + limit, auditLogs.pagination.total)} of{' '}
                {auditLogs.pagination.total} entries
              </div>
              <div className="flex gap-2">
                {auditLogs.pagination.has_previous && (
                  <a href={`?offset=${Math.max(0, offset - limit)}&limit=${limit}`}>
                    <Button variant="outline" size="sm">Previous</Button>
                  </a>
                )}
                {auditLogs.pagination.has_next && (
                  <a href={`?offset=${offset + limit}&limit=${limit}`}>
                    <Button variant="outline" size="sm">Next</Button>
                  </a>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default async function AuditPage({
  searchParams,
}: {
  searchParams: {
    timeRange?: TimeRange;
    limit?: string;
    offset?: string;
    severity?: string;
    event_type?: string;
  };
}) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Audit Log Explorer</h2>
          <p className="text-muted-foreground">
            Browse system audit logs and event history
          </p>
        </div>
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
        <AuditLogsContent searchParams={searchParams} />
      </Suspense>
    </div>
  );
}

