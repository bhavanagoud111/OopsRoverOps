import React, { useEffect, useRef } from 'react';
import { Card } from './Card';

export interface LogEntry {
  id: string;
  timestamp: string;
  agent: 'Planner' | 'Navigator' | 'Safety' | 'Reporter';
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

interface AgentLogsProps {
  logs: LogEntry[];
}

export function AgentLogs({ logs }: AgentLogsProps) {
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const getAgentColor = (agent: string) => {
    switch (agent) {
      case 'Planner': return 'var(--color-primary)';
      case 'Navigator': return 'var(--color-secondary)';
      case 'Safety': return 'var(--color-accent)';
      case 'Reporter': return 'var(--color-success)';
      default: return 'var(--color-text-primary)';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'success': return '✓';
      case 'warning': return '⚠';
      case 'error': return '✗';
      default: return '•';
    }
  };

  return (
    <Card className="h-full flex flex-col">
      <h2 className="mb-4">Agent Logs</h2>
      <div 
        className="flex-1 overflow-y-auto space-y-2 monospace"
        role="log"
        aria-live="polite"
        aria-label="Agent activity logs"
      >
        {logs.length === 0 ? (
          <p className="text-[var(--color-text-secondary)] text-center py-8">
            <small>Waiting for mission to start...</small>
          </p>
        ) : (
          logs.map((log) => (
            <div
              key={log.id}
              className="p-3 bg-[var(--color-background)] rounded border-l-4"
              style={{ borderLeftColor: getAgentColor(log.agent) }}
            >
              <div className="flex items-start gap-2">
                <span className="text-[var(--color-text-secondary)] flex-shrink-0">
                  <small>{log.timestamp}</small>
                </span>
                <span 
                  className="flex-shrink-0"
                  style={{ color: getAgentColor(log.agent) }}
                  aria-label={`${log.agent} agent`}
                >
                  <small>[{log.agent}]</small>
                </span>
                <span className="flex-shrink-0" aria-hidden="true">
                  <small>{getTypeIcon(log.type)}</small>
                </span>
                <p className="text-[var(--color-text-primary)] flex-1">
                  <small>{log.message}</small>
                </p>
              </div>
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>
    </Card>
  );
}
