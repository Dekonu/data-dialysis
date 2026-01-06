'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { LayoutDashboard, FileText, AlertCircle, Shield, Gauge, Lock } from 'lucide-react';
import { ThemeToggle } from '@/components/theme-toggle';

const navigation = [
  {
    name: 'Dashboard',
    href: '/',
    icon: LayoutDashboard,
  },
  {
    name: 'Security',
    href: '/security',
    icon: Shield,
  },
  {
    name: 'Performance',
    href: '/performance',
    icon: Gauge,
  },
  {
    name: 'Audit Logs',
    href: '/audit',
    icon: FileText,
  },
  {
    name: 'Redactions',
    href: '/redactions',
    icon: Lock,
  },
  {
    name: 'Circuit Breaker',
    href: '/circuit-breaker',
    icon: AlertCircle,
  },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <div className="flex items-center gap-2 sm:gap-4">
      <nav className="flex space-x-1">
        {navigation.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center gap-2 px-2 sm:px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 hover:scale-105',
                isActive
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )}
            >
              <item.icon className="h-4 w-4 flex-shrink-0" />
              <span className="hidden sm:inline">{item.name}</span>
            </Link>
          );
        })}
      </nav>
      <ThemeToggle />
    </div>
  );
}

