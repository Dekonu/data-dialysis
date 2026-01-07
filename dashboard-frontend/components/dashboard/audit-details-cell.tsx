'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';

interface AuditDetailsCellProps {
  details: Record<string, unknown> | null | undefined;
}

export function AuditDetailsCell({ details }: AuditDetailsCellProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!details || Object.keys(details).length === 0) {
    return <span className="text-xs text-muted-foreground">None</span>;
  }

  const fieldCount = Object.keys(details).length;

  return (
    <div className="relative">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="h-6 px-2 text-xs">
            {fieldCount} {fieldCount === 1 ? 'field' : 'fields'}
            {isOpen ? (
              <ChevronUp className="ml-1 h-3 w-3" />
            ) : (
              <ChevronDown className="ml-1 h-3 w-3" />
            )}
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="absolute left-0 top-full z-50 mt-1">
          <div className="rounded-md border bg-background shadow-lg p-3 space-y-2 max-w-md max-h-96 overflow-auto">
            {Object.entries(details).map(([key, value]) => {
              let displayValue: string;
              if (value === null || value === undefined) {
                displayValue = 'null';
              } else if (typeof value === 'object') {
                displayValue = JSON.stringify(value, null, 2);
              } else {
                displayValue = String(value);
              }

              return (
                <div key={key} className="text-xs">
                  <span className="font-semibold text-foreground">{key}:</span>{' '}
                  <span className="text-muted-foreground font-mono break-all">
                    {displayValue}
                  </span>
                </div>
              );
            })}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

