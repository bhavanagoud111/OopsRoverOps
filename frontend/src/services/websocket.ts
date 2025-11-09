import type { WebSocketMessage, MissionState } from '../types/mission';

export class WebSocketService {
  private ws: WebSocket | null = null;
  private missionId: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private isConnected = false;

  constructor(private apiUrl: string = import.meta.env.VITE_WS_URL || 'ws://localhost:8000') {}

  connect(missionId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.missionId = missionId;
      const wsUrl = `${this.apiUrl}/ws/mission/${missionId}`;
      
      try {
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          resolve();
        };
        
        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };
        
        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.isConnected = false;
          this.attemptReconnect();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts && this.missionId) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        this.connect(this.missionId!);
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  private handleMessage(message: WebSocketMessage) {
    // Emit to all listeners for this message type
    const typeListeners = this.listeners.get(message.type);
    if (typeListeners) {
      typeListeners.forEach(listener => listener(message));
    }
    
    // Also emit to 'message' listeners for all messages
    const allListeners = this.listeners.get('message');
    if (allListeners) {
      allListeners.forEach(listener => listener(message));
    }
  }

  on(messageType: string, callback: (data: any) => void) {
    if (!this.listeners.has(messageType)) {
      this.listeners.set(messageType, new Set());
    }
    this.listeners.get(messageType)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(messageType);
      if (listeners) {
        listeners.delete(callback);
      }
    };
  }

  off(messageType: string, callback: (data: any) => void) {
    const listeners = this.listeners.get(messageType);
    if (listeners) {
      listeners.delete(callback);
    }
  }

  send(message: any) {
    if (this.ws && this.isConnected) {
      this.ws.send(JSON.stringify(message));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.missionId = null;
    this.isConnected = false;
    this.listeners.clear();
  }

  getConnectionState(): boolean {
    return this.isConnected;
  }
}

// Singleton instance
export const websocketService = new WebSocketService();

