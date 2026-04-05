import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_URL } from '../utils/constants';

/**
 * WebSocket hook for real-time streaming analysis.
 */
export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [progress, setProgress] = useState(0);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log('[NeuroLens WS] Connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);

          if (data.type === 'connected') {
            setSessionId(data.session_id);
          } else if (data.type === 'progress') {
            setProgress(data.progress || 0);
          }
        } catch (e) {
          console.error('[NeuroLens WS] Parse error:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('[NeuroLens WS] Disconnected');
        // Auto-reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(connect, 3000);
      };

      ws.onerror = (error) => {
        console.error('[NeuroLens WS] Error:', error);
      };
    } catch (e) {
      console.error('[NeuroLens WS] Connection failed:', e);
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, []);

  const sendMessage = useCallback((text, explain = true) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ text, explain }));
      setProgress(0);
    }
  }, []);

  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  return {
    isConnected,
    sessionId,
    lastMessage,
    progress,
    connect,
    disconnect,
    sendMessage,
  };
}
