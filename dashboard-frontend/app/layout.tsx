import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Navigation } from '@/components/dashboard/navigation';
import { ThemeProvider } from '@/components/theme-provider';
import { ErrorBoundary } from '@/components/error-boundary';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Data-Dialysis Dashboard',
  description: 'Health monitoring dashboard for Data-Dialysis pipeline',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Runtime API URL configuration - auto-detects from current host */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                // Auto-detect API URL from current hostname
                const hostname = window.location.hostname;
                const protocol = window.location.protocol;
                
                // If not localhost, use same hostname with port 8000
                if (hostname && hostname !== 'localhost' && hostname !== '127.0.0.1') {
                  const apiUrl = protocol === 'https:' 
                    ? 'https://' + hostname + ':8000'
                    : 'http://' + hostname + ':8000';
                  window.__API_URL__ = apiUrl;
                  window.__WS_URL__ = apiUrl.replace(/^https?:/, protocol === 'https:' ? 'wss:' : 'ws:');
                } else {
                  // Fallback to build-time env or default
                  window.__API_URL__ = '${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}';
                  window.__WS_URL__ = '${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}';
                }
              })();
            `,
          }}
        />
      </head>
      <body className={inter.className} suppressHydrationWarning>
        <ThemeProvider>
          <div className="min-h-screen bg-background transition-colors">
            <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
              <div className="container mx-auto px-4 py-4">
                <div className="flex items-center justify-between">
                  <h1 className="text-xl sm:text-2xl font-bold">Data-Dialysis Dashboard</h1>
                  <Navigation />
                </div>
              </div>
            </nav>
            <main className="container mx-auto px-4 py-4 sm:py-8 animate-fade-in">
              <ErrorBoundary>{children}</ErrorBoundary>
            </main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
