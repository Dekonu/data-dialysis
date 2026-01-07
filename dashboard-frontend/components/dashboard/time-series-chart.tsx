'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export interface TimeSeriesDataPoint {
  date: string;
  value: number;
  label?: string;
}

export interface TimeSeriesChartProps {
  title: string;
  description?: string;
  data: TimeSeriesDataPoint[];
  dataKey: string;
  color?: string;
  unit?: string;
}

export function TimeSeriesChart({
  title,
  description,
  data,
  dataKey,
  color = '#8884d8',
  unit = '',
}: TimeSeriesChartProps) {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const formatValue = (value: number) => {
    return `${value.toLocaleString()}${unit}`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              style={{ fontSize: '12px' }}
            />
            <YAxis tickFormatter={formatValue} style={{ fontSize: '12px' }} />
            <Tooltip
              labelFormatter={(label) => `Date: ${formatDate(label)}`}
              formatter={(value: number | undefined) => formatValue(value ?? 0)}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

