'use client';

import { Button } from '@/components/ui/button';
import { Download } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function AuditExportButtons() {
  const handleExport = (format: 'json' | 'csv') => {
    const url = `${API_BASE_URL}/api/audit-logs/export?format=${format}`;
    window.open(url, '_blank');
  };

  return (
    <div className="flex gap-2">
      <Button
        variant="outline"
        size="sm"
        onClick={() => handleExport('json')}
      >
        <Download className="h-4 w-4 mr-2" />
        Export JSON
      </Button>
      <Button
        variant="outline"
        size="sm"
        onClick={() => handleExport('csv')}
      >
        <Download className="h-4 w-4 mr-2" />
        Export CSV
      </Button>
    </div>
  );
}

