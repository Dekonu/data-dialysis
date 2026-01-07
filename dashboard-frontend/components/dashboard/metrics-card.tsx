'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface MetricsCardProps {
  title: string;
  value: number | string;
  description?: string;
  trend?: number | null;
  trendLabel?: string;
  variant?: 'default' | 'success' | 'warning' | 'destructive';
  formatType?: 'number' | 'percentage' | 'currency' | 'plain';
}

export function MetricsCard({
  title,
  value,
  description,
  trend,
  trendLabel,
  variant = 'default',
  formatType = 'number',
}: MetricsCardProps) {
  const formatValue = (val: number | string): string => {
    if (formatType === 'plain') return String(val);
    if (formatType === 'percentage') {
      const num = typeof val === 'number' ? val : parseFloat(String(val));
      return `${(num * 100).toFixed(1)}%`;
    }
    if (formatType === 'currency') {
      const num = typeof val === 'number' ? val : parseFloat(String(val));
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(num);
    }
    // Default: number format
    const num = typeof val === 'number' ? val : parseFloat(String(val));
    return num.toLocaleString();
  };

  const formattedValue = formatValue(value);

  const getTrendIcon = () => {
    if (trend === null || trend === undefined) return null;
    if (trend > 0) return <TrendingUp className="h-4 w-4" />;
    if (trend < 0) return <TrendingDown className="h-4 w-4" />;
    return <Minus className="h-4 w-4" />;
  };

  const getTrendColor = () => {
    if (trend === null || trend === undefined) return 'text-muted-foreground';
    if (trend > 0) return 'text-green-600 dark:text-green-400';
    if (trend < 0) return 'text-red-600 dark:text-red-400';
    return 'text-muted-foreground';
  };

  const getVariantStyles = () => {
    switch (variant) {
      case 'success':
        return 'border-green-200 dark:border-green-800';
      case 'warning':
        return 'border-yellow-200 dark:border-yellow-800';
      case 'destructive':
        return 'border-red-200 dark:border-red-800';
      default:
        return '';
    }
  };

  return (
    <Card className={cn(getVariantStyles(), 'transition-all duration-200 hover:shadow-md hover:scale-[1.02]')}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-xs sm:text-sm font-medium">{title}</CardTitle>
        {variant !== 'default' && (
          <Badge
            variant={
              variant === 'success'
                ? 'default'
                : variant === 'warning'
                ? 'secondary'
                : 'destructive'
            }
            className="text-xs"
          >
            {variant}
          </Badge>
        )}
      </CardHeader>
      <CardContent>
        <div className="text-xl sm:text-2xl font-bold">{formattedValue}</div>
        {description && (
          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{description}</p>
        )}
        {trend !== null && trend !== undefined && (
          <div className={cn('flex items-center gap-1 mt-2 text-xs', getTrendColor())}>
            {getTrendIcon()}
            <span>
              {trend > 0 ? '+' : ''}
              {typeof trend === 'number' ? (trend * 100).toFixed(1) : trend}%
            </span>
            {trendLabel && <span className="text-muted-foreground hidden sm:inline">vs {trendLabel}</span>}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

