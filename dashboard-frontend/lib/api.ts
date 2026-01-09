import type {
  OverviewMetrics,
  SecurityMetrics,
  PerformanceMetrics,
  TimeRange,
  AuditLogsResponse,
  RedactionLogsResponse,
  CircuitBreakerStatus,
  ChangeHistoryResponse,
  ChangeSummary,
  RecordChangeHistoryResponse,
} from '@/types/api';

// Get API URL at runtime - works for both build-time and runtime configuration
// Priority: window.__API_URL__ (runtime) > NEXT_PUBLIC_API_URL (build-time) > default
function getApiBaseUrl(): string {
  // Runtime configuration (set via window object or script tag)
  if (typeof window !== 'undefined' && (window as any).__API_URL__) {
    return (window as any).__API_URL__;
  }
  
  // Build-time configuration
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  
  // Auto-detect from current host (for same-origin deployments)
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    // If accessing from EC2, use the same hostname but port 8000
    if (host && host !== 'localhost' && host !== '127.0.0.1') {
      return `http://${host}:8000`;
    }
  }
  
  // Default fallback
  return 'http://localhost:8000';
}

const API_BASE_URL = getApiBaseUrl();

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  database: {
    status: 'connected' | 'disconnected';
    type: string;
    response_time_ms?: number;
  };
}

export async function apiRequest<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => response.statusText);
      throw new Error(`API error: ${response.status} ${errorText}`);
    }

    return response.json();
  } catch (error) {
    // Re-throw with more context if it's a network error
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error(`Network error: Failed to connect to ${API_BASE_URL}${endpoint}`);
    }
    throw error;
  }
}

export const api = {
  health: () => apiRequest<HealthResponse>('/api/health'),
  metrics: {
    overview: (timeRange: TimeRange = '24h') =>
      apiRequest<OverviewMetrics>(`/api/metrics/overview?time_range=${timeRange}`),
    security: (timeRange: TimeRange = '7d') =>
      apiRequest<SecurityMetrics>(`/api/metrics/security?time_range=${timeRange}`),
    performance: (timeRange: TimeRange = '24h') =>
      apiRequest<PerformanceMetrics>(`/api/metrics/performance?time_range=${timeRange}`),
  },
  audit: {
    logs: (params?: {
      limit?: number;
      offset?: number;
      severity?: string;
      event_type?: string;
      start_date?: string;
      end_date?: string;
      source_adapter?: string;
      sort_by?: string;
      sort_order?: 'ASC' | 'DESC';
    }) => {
      const searchParams = new URLSearchParams();
      if (params) {
        Object.entries(params).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            searchParams.append(key, String(value));
          }
        });
      }
      const query = searchParams.toString();
      return apiRequest<AuditLogsResponse>(`/api/audit-logs${query ? `?${query}` : ''}`);
    },
    redactionLogs: (params?: {
      field_name?: string;
      time_range?: TimeRange;
      limit?: number;
      offset?: number;
      rule_triggered?: string;
      source_adapter?: string;
      ingestion_id?: string;
      sort_by?: string;
      sort_order?: 'ASC' | 'DESC';
    }) => {
      const searchParams = new URLSearchParams();
      if (params) {
        Object.entries(params).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            searchParams.append(key, String(value));
          }
        });
      }
      const query = searchParams.toString();
      return apiRequest<RedactionLogsResponse>(`/api/redaction-logs${query ? `?${query}` : ''}`);
    },
    export: (params: {
      format: 'json' | 'csv';
      severity?: string;
      event_type?: string;
      start_date?: string;
      end_date?: string;
    }) => {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });
      return fetch(`${API_BASE_URL}/api/audit-logs/export?${searchParams.toString()}`).then(
        (response) => {
          if (!response.ok) {
            throw new Error(`Export failed: ${response.statusText}`);
          }
          return response.blob();
        }
      );
    },
  },
  circuitBreaker: {
    status: () => apiRequest<CircuitBreakerStatus>('/api/circuit-breaker/status'),
  },
  changeHistory: {
    list: (params?: {
      limit?: number;
      offset?: number;
      table_name?: string;
      record_id?: string;
      field_name?: string;
      change_type?: 'INSERT' | 'UPDATE';
      start_date?: string;
      end_date?: string;
      ingestion_id?: string;
      source_adapter?: string;
      sort_by?: string;
      sort_order?: 'ASC' | 'DESC';
    }) => {
      const searchParams = new URLSearchParams();
      if (params) {
        Object.entries(params).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            searchParams.append(key, String(value));
          }
        });
      }
      const query = searchParams.toString();
      return apiRequest<ChangeHistoryResponse>(`/api/change-history${query ? `?${query}` : ''}`);
    },
    summary: (params?: {
      time_range?: TimeRange;
      table_name?: string;
    }) => {
      const searchParams = new URLSearchParams();
      if (params) {
        Object.entries(params).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            searchParams.append(key, String(value));
          }
        });
      }
      const query = searchParams.toString();
      return apiRequest<ChangeSummary>(`/api/change-history/summary${query ? `?${query}` : ''}`);
    },
    record: (tableName: string, recordId: string, params?: {
      limit?: number;
    }) => {
      const searchParams = new URLSearchParams();
      if (params) {
        Object.entries(params).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            searchParams.append(key, String(value));
          }
        });
      }
      const query = searchParams.toString();
      return apiRequest<RecordChangeHistoryResponse>(
        `/api/change-history/record/${tableName}/${recordId}${query ? `?${query}` : ''}`
      );
    },
    export: (params: {
      format: 'json' | 'csv';
      table_name?: string;
      start_date?: string;
      end_date?: string;
      change_type?: 'INSERT' | 'UPDATE';
    }) => {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });
      return fetch(`${API_BASE_URL}/api/change-history/export?${searchParams.toString()}`).then(
        (response) => {
          if (!response.ok) {
            throw new Error(`Export failed: ${response.statusText}`);
          }
          return response.blob();
        }
      );
    },
  },
};

