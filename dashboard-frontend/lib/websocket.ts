/**
 * WebSocket client for real-time dashboard updates.
 * 
 * Provides a connection manager with automatic reconnection,
 * fallback to polling, and message handling.
 */

import type {
  OverviewMetrics,
  SecurityMetrics,
  PerformanceMetrics,
  CircuitBreakerStatus,
} from '@/types/api';

// Determine WebSocket URL based on API URL
const getWebSocketUrl = (): string => {
  // Runtime configuration (set via window object)
  if (typeof window !== 'undefined' && (window as any).__WS_URL__) {
    return (window as any).__WS_URL__;
  }
  
  // Build-time configuration
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  
  // Convert http:// to ws:// and https:// to wss://
  if (apiUrl.startsWith('https://')) {
    return apiUrl.replace('https://', 'wss://');
  }
  if (apiUrl.startsWith('http://')) {
    return apiUrl.replace('http://', 'ws://');
  }
  
  // Fallback to explicit WebSocket URL or default
  return process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
};

const WS_BASE_URL = getWebSocketUrl();

export type WebSocketMessageType =
  | 'connection'
  | 'heartbeat'
  | 'metrics_update'
  | 'security_metrics_update'
  | 'performance_metrics_update'
  | 'circuit_breaker_update'
  | 'error';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  timestamp: string;
  message?: string;
  server_time?: string;
  data?: OverviewMetrics | SecurityMetrics | PerformanceMetrics | CircuitBreakerStatus;
  error?: string;
  error_type?: string;
}

export type WebSocketMessageHandler = (message: WebSocketMessage) => void;

export interface WebSocketClientOptions {
  timeRange?: string;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
  onMessage?: WebSocketMessageHandler;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private options: Required<WebSocketClientOptions>;
  private reconnectAttempts = 0;
  private reconnectTimeoutId: NodeJS.Timeout | null = null;
  private isManualClose = false;
  private messageHandlers: Set<WebSocketMessageHandler> = new Set();

  constructor(options: WebSocketClientOptions = {}) {
    const timeRange = options.timeRange || '24h';
    this.url = `${WS_BASE_URL}/ws/realtime?time_range=${timeRange}`;
    
    this.options = {
      timeRange,
      autoReconnect: options.autoReconnect ?? true,
      reconnectInterval: options.reconnectInterval ?? 3000,
      maxReconnectAttempts: options.maxReconnectAttempts ?? 10,
      onConnect: options.onConnect ?? (() => {}),
      onDisconnect: options.onDisconnect ?? (() => {}),
      onError: options.onError ?? (() => {}),
      onMessage: options.onMessage ?? (() => {}),
    };

    // Add initial message handler if provided
    if (options.onMessage) {
      this.messageHandlers.add(options.onMessage);
    }
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    this.isManualClose = false;
    
    // Check if WebSocket is supported
    if (typeof WebSocket === 'undefined') {
      this.handleError(new Error('WebSocket is not supported in this environment'));
      return;
    }
    
    try {
      console.log(`Connecting to WebSocket: ${this.url}`);
      this.ws = new WebSocket(this.url);
      this.setupEventHandlers();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create WebSocket connection';
      console.error('WebSocket connection error:', errorMessage);
      this.handleError(new Error(errorMessage));
      this.scheduleReconnect();
    }
  }

  disconnect(): void {
    this.isManualClose = true;
    this.clearReconnectTimeout();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  addMessageHandler(handler: WebSocketMessageHandler): () => void {
    this.messageHandlers.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.messageHandlers.delete(handler);
    };
  }

  removeMessageHandler(handler: WebSocketMessageHandler): void {
    this.messageHandlers.delete(handler);
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.options.onConnect();
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = () => {
      // WebSocket error event doesn't provide detailed error info
      // The actual error will be available in onclose event
      console.warn('WebSocket error event triggered. Connection may have failed.');
      // Don't call handleError here - let onclose handle it with more context
    };

    this.ws.onclose = (event) => {
      // Handle different close codes appropriately
      const isNormalClose = event.code === 1000; // Normal closure
      const isServiceRestart = event.code === 1012; // Service restart (expected)
      const isGoingAway = event.code === 1001; // Going away (server shutdown)
      
      // Only treat as error if it's an unexpected close (not manual, not normal, not expected codes)
      if (!isNormalClose && !isServiceRestart && !isGoingAway && !this.isManualClose) {
        const errorMessage = this.getCloseErrorMessage(event.code, event.reason);
        console.warn(`WebSocket closed unexpectedly: ${errorMessage} (code: ${event.code})`);
        this.handleError(new Error(errorMessage));
      } else if (isServiceRestart || isGoingAway) {
        // Service restart or going away - expected, just log at debug level
        console.debug(`WebSocket closed: ${this.getCloseErrorMessage(event.code, event.reason)} (code: ${event.code})`);
      }
      
      this.options.onDisconnect();
      
      // Always attempt to reconnect unless it was a manual close
      if (!this.isManualClose && this.options.autoReconnect) {
        this.scheduleReconnect();
      }
    };
  }

  private handleMessage(message: WebSocketMessage): void {
    // Call all registered handlers
    this.messageHandlers.forEach((handler) => {
      try {
        handler(message);
      } catch (error) {
        console.error('Error in WebSocket message handler:', error);
      }
    });

    // Call the main onMessage handler
    this.options.onMessage(message);
  }

  private handleError(error: Error): void {
    // Only log to console if it's not a connection refused error (which is expected if backend is down)
    const isConnectionRefused = error.message.includes('connection') || 
                                 error.message.includes('ECONNREFUSED') ||
                                 error.message.includes('Failed to fetch');
    
    if (!isConnectionRefused) {
      console.error('WebSocket error:', error);
    } else {
      console.debug('WebSocket connection unavailable (backend may be down):', error.message);
    }
    
    this.options.onError(error);
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeoutId) {
      return; // Already scheduled
    }

    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.options.reconnectInterval * this.reconnectAttempts;

    this.reconnectTimeoutId = setTimeout(() => {
      this.reconnectTimeoutId = null;
      this.connect();
    }, delay);
  }

  private clearReconnectTimeout(): void {
    if (this.reconnectTimeoutId) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private getCloseErrorMessage(code: number, reason?: string): string {
    // WebSocket close codes: https://developer.mozilla.org/en-US/docs/Web/API/CloseEvent
    switch (code) {
      case 1000:
        return 'Normal closure';
      case 1001:
        return 'Going away - server is shutting down';
      case 1002:
        return 'Protocol error';
      case 1003:
        return 'Unsupported data type';
      case 1006:
        return 'Abnormal closure - connection lost';
      case 1007:
        return 'Invalid data';
      case 1008:
        return 'Policy violation';
      case 1009:
        return 'Message too large';
      case 1011:
        return 'Server error';
      case 1012:
        return 'Service restart';
      case 1013:
        return 'Try again later';
      case 1014:
        return 'Bad gateway';
      case 1015:
        return 'TLS handshake failed';
      default:
        return reason || `Connection closed (code: ${code})`;
    }
  }
}

