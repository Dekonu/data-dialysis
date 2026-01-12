import { Suspense } from 'react';
import { api } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

// Force dynamic rendering - don't statically generate this page
export const dynamic = 'force-dynamic';
export const revalidate = 0;
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { TimeRangeSelector } from '@/components/dashboard/time-range-selector';
import { MetricsCard } from '@/components/dashboard/metrics-card';
import { ChangeHistoryFilters } from '@/components/dashboard/change-history-filters';
import type { TimeRange } from '@/types/api';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { ChangeHistoryExportButtons } from '@/components/dashboard/change-history-export-buttons';

interface ChangeHistoryContentProps {
  searchParams: Promise<{
    timeRange?: TimeRange;
    limit?: string;
    offset?: string;
    table_name?: string;
    record_id?: string;
    field_name?: string;
    change_type?: string;
  }>;
}

function formatDate(dateString: string): string {
  try {
    const date = new Date(dateString);
    return date.toLocaleString();
  } catch {
    return dateString;
  }
}

function truncateValue(value: string | null | undefined, maxLength: number = 50): string {
  if (!value) return '—';
  if (value.length <= maxLength) return value;
  return `${value.substring(0, maxLength)}...`;
}

function getChangeTypeVariant(changeType: string): 'default' | 'secondary' {
  switch (changeType) {
    case 'INSERT':
      return 'default';
    case 'UPDATE':
      return 'secondary';
    default:
      return 'secondary';
  }
}

async function ChangeHistoryContent({ searchParams }: ChangeHistoryContentProps) {
  const params = await searchParams;
  const limit = parseInt(params.limit || '100');
  const offset = parseInt(params.offset || '0');
  const timeRange = (params.timeRange || '24h') as TimeRange;

  let changeHistory;
  let summary;
  let error: string | null = null;

  try {
    [changeHistory, summary] = await Promise.all([
      api.changeHistory.list({
        limit,
        offset,
        table_name: params.table_name || undefined,
        record_id: params.record_id || undefined,
        field_name: params.field_name || undefined,
        change_type: params.change_type as 'INSERT' | 'UPDATE' | undefined,
        sort_by: 'changed_at',
        sort_order: 'DESC',
      }),
      api.changeHistory.summary({
        time_range: timeRange,
        table_name: params.table_name || undefined,
      }),
    ]);
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to fetch change history';
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle>Error Loading Change History</CardTitle>
          <CardDescription>Unable to fetch change history from the API</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-destructive">{error}</p>
        </CardContent>
      </Card>
    );
  }


  const hasNext = changeHistory && offset + limit < changeHistory.total;
  const hasPrevious = offset > 0;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      {summary && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricsCard
            title="Total Changes"
            value={summary.total_changes.toLocaleString()}
            description="All field-level changes"
            variant="default"
            formatType="number"
          />
          <MetricsCard
            title="Unique Records"
            value={summary.unique_records.toLocaleString()}
            description="Records affected"
            variant="default"
            formatType="number"
          />
          <MetricsCard
            title="Tables Affected"
            value={summary.tables_affected.toLocaleString()}
            description="Number of tables"
            variant="default"
            formatType="number"
          />
          <MetricsCard
            title="Fields Changed"
            value={summary.fields_changed.toLocaleString()}
            description="Unique fields modified"
            variant="default"
            formatType="number"
          />
        </div>
      )}

      {/* Change Type Breakdown */}
      {summary && (
        <div className="grid gap-4 md:grid-cols-2">
          <MetricsCard
            title="Inserts"
            value={summary.inserts.toLocaleString()}
            description="New records created"
            variant="default"
            formatType="number"
          />
          <MetricsCard
            title="Updates"
            value={summary.updates.toLocaleString()}
            description="Existing records modified"
            variant="default"
            formatType="number"
          />
        </div>
      )}

      {/* Change History Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Change History</CardTitle>
              <CardDescription>
                Field-level change audit trail. Total: {changeHistory?.total || 0} changes
              </CardDescription>
            </div>
            <ChangeHistoryExportButtons
              table_name={params.table_name}
              change_type={params.change_type}
            />
          </div>
        </CardHeader>
        <CardContent>
          {!changeHistory || changeHistory.changes.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">No change history available</p>
          ) : (
            <>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Timestamp</TableHead>
                      <TableHead>Table</TableHead>
                      <TableHead>Record ID</TableHead>
                      <TableHead>Field</TableHead>
                      <TableHead>Change Type</TableHead>
                      <TableHead>Old Value</TableHead>
                      <TableHead>New Value</TableHead>
                      <TableHead>Source</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {changeHistory.changes.map((change) => (
                      <TableRow key={change.change_id}>
                        <TableCell className="font-mono text-xs">
                          {formatDate(change.changed_at)}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{change.table_name}</Badge>
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {change.record_id}
                        </TableCell>
                        <TableCell>{change.field_name}</TableCell>
                        <TableCell>
                          <Badge variant={getChangeTypeVariant(change.change_type)}>
                            {change.change_type}
                          </Badge>
                        </TableCell>
                        <TableCell className="max-w-[200px] truncate font-mono text-xs">
                          {truncateValue(change.old_value)}
                        </TableCell>
                        <TableCell className="max-w-[200px] truncate font-mono text-xs">
                          {truncateValue(change.new_value)}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {change.source_adapter || '—'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              <div className="flex items-center justify-between mt-4">
                <div className="text-sm text-muted-foreground">
                  Showing {offset + 1} to {Math.min(offset + limit, changeHistory.total)} of{' '}
                  {changeHistory.total} changes
                </div>
                <div className="flex items-center gap-2">
                  <Link
                    href={`?limit=${limit}&offset=${Math.max(0, offset - limit)}${params.table_name ? `&table_name=${params.table_name}` : ''}${params.record_id ? `&record_id=${params.record_id}` : ''}${params.field_name ? `&field_name=${params.field_name}` : ''}${params.change_type ? `&change_type=${params.change_type}` : ''}${timeRange ? `&timeRange=${timeRange}` : ''}`}
                  >
                    <Button variant="outline" size="sm" disabled={!hasPrevious}>
                      <ArrowLeft className="h-4 w-4 mr-2" />
                      Previous
                    </Button>
                  </Link>
                  <Link
                    href={`?limit=${limit}&offset=${offset + limit}${params.table_name ? `&table_name=${params.table_name}` : ''}${params.record_id ? `&record_id=${params.record_id}` : ''}${params.field_name ? `&field_name=${params.field_name}` : ''}${params.change_type ? `&change_type=${params.change_type}` : ''}${timeRange ? `&timeRange=${timeRange}` : ''}`}
                  >
                    <Button variant="outline" size="sm" disabled={!hasNext}>
                      Next
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </Button>
                  </Link>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function ChangeHistoryPage({
  searchParams,
}: {
  searchParams: Promise<{
    timeRange?: TimeRange;
    limit?: string;
    offset?: string;
    table_name?: string;
    record_id?: string;
    field_name?: string;
    change_type?: string;
  }>;
}) {
  return (
    <div className="container mx-auto py-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Change History</h1>
          <p className="text-sm sm:text-base text-muted-foreground mt-1">
            Field-level change data capture (CDC) audit trail
          </p>
        </div>
        <Suspense fallback={<div className="w-full sm:w-[180px] h-10 bg-muted animate-pulse rounded-md" />}>
          <TimeRangeSelector />
        </Suspense>
      </div>
      <Suspense fallback={<div className="h-64 bg-muted animate-pulse rounded-lg" />}>
        <ChangeHistoryFilters />
      </Suspense>
      <Suspense fallback={<div className="h-64 bg-muted animate-pulse rounded-lg" />}>
        <ChangeHistoryContent searchParams={searchParams} />
      </Suspense>
    </div>
  );
}
