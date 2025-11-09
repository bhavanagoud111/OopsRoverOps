import { useState, useEffect, useRef } from 'react';
import { Button } from '../Button';
import { Card } from '../Card';
import { MissionInput } from '../MissionInput';
import { AgentLogs, type LogEntry } from '../AgentLogs';
import { MissionCanvas } from '../MissionCanvas';
import { FileDown } from 'lucide-react';
import { startMission, getMissionStatus, getMissionReport, getAPOD } from '../../services/api';
import { websocketService } from '../../services/websocket';
import type { MissionState, MissionLog } from '../../types/mission';
import { AgentType, MissionStatus } from '../../types/mission';

interface Position {
  x: number;
  y: number;
}

interface DashboardPageProps {
  onNavigate: (page: string, data?: any) => void;
}

const missionLogToLogEntry = (log: MissionLog): LogEntry => {
  const agentMap: Record<AgentType, LogEntry['agent']> = {
    [AgentType.PLANNER]: 'Planner',
    [AgentType.ROVER]: 'Navigator',
    [AgentType.SAFETY]: 'Safety',
    [AgentType.REPORTER]: 'Reporter',
    [AgentType.SUPERVISOR]: 'Planner',
    [AgentType.SYSTEM]: 'Planner',
  };

  return {
    id: `${log.mission_id}-${log.timestamp}`,
    timestamp: new Date(log.timestamp).toLocaleTimeString(),
    agent: agentMap[log.agent_type] || 'Planner',
    message: log.message,
    type: log.level as LogEntry['type'],
  };
};

export function DashboardPage({ onNavigate }: DashboardPageProps) {
  const [missionGoal, setMissionGoal] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [path, setPath] = useState<Position[]>([]);
  const [missionComplete, setMissionComplete] = useState(false);
  const [currentMissionId, setCurrentMissionId] = useState<string | null>(null);
  const [missionState, setMissionState] = useState<MissionState | null>(null);
  const [apodBackground, setApodBackground] = useState<any>(null);
  const pathHistoryRef = useRef<Position[]>([]);
  const logsProcessedRef = useRef<Set<string>>(new Set());

  // Fetch APOD background on mount
  useEffect(() => {
    getAPOD()
      .then(data => {
        setApodBackground(data);
      })
      .catch(err => {
        console.error('Failed to fetch APOD:', err);
      });
  }, []);

  // Fetch initial mission state
  useEffect(() => {
    if (currentMissionId && !missionState) {
      getMissionStatus(currentMissionId)
        .then(state => {
          setMissionState(state);
          if (state.rover_position) {
            const startPos: Position = {
              x: state.rover_position.x,
              y: state.rover_position.y,
            };
            pathHistoryRef.current = [startPos];
            setPath([startPos]);
          }
        })
        .catch(console.error);
    }
  }, [currentMissionId, missionState]);


  // WebSocket connection
  useEffect(() => {
    if (!currentMissionId) return;

    const connectWebSocket = async () => {
      try {
        await websocketService.connect(currentMissionId);
        
        websocketService.on('status', (message: any) => {
          if (message.data) {
            setMissionState(prev => {
              if (!prev) return prev;
              return {
                ...prev,
                rover_position: message.data.rover_position || prev.rover_position,
                current_step: message.data.current_step ?? prev.current_step,
                agent_states: message.data.agent_states || prev.agent_states,
              };
            });
          }

          if (message.data?.rover_position) {
            const pos = message.data.rover_position;
            const newPosition: Position = { x: pos.x, y: pos.y };
            const lastPosition = pathHistoryRef.current[pathHistoryRef.current.length - 1];
            if (!lastPosition || lastPosition.x !== newPosition.x || lastPosition.y !== newPosition.y) {
              pathHistoryRef.current = [...pathHistoryRef.current, newPosition];
              setPath([...pathHistoryRef.current]);
            }
          }
          
          if (message.status === MissionStatus.COMPLETE || message.status === 'complete') {
            setIsRunning(false);
            setMissionComplete(true);
          }
        });
        
        websocketService.on('update', (message: any) => {
          if (message.data?.rover_position) {
            const pos = message.data.rover_position;
            const newPosition: Position = { x: pos.x, y: pos.y };
            const lastPosition = pathHistoryRef.current[pathHistoryRef.current.length - 1];
            if (!lastPosition || lastPosition.x !== newPosition.x || lastPosition.y !== newPosition.y) {
              pathHistoryRef.current = [...pathHistoryRef.current, newPosition];
              setPath([...pathHistoryRef.current]);
            }

            setMissionState(prev => {
              if (!prev) return prev;
              return {
                ...prev,
                rover_position: pos,
              };
            });
          }
          
          if (message.data?.logs) {
            const newLogs = message.data.logs
              .filter((log: MissionLog) => {
                const logId = `${log.mission_id}-${log.timestamp}`;
                if (logsProcessedRef.current.has(logId)) return false;
                logsProcessedRef.current.add(logId);
                return true;
              })
              .map(missionLogToLogEntry);
            
            if (newLogs.length > 0) {
              setLogs(prev => [...prev, ...newLogs]);
            }
          }
        });

        websocketService.on('log', (message: any) => {
          if (message.data?.logs) {
            const newLogs = message.data.logs
              .filter((log: MissionLog) => {
                const logId = `${log.mission_id}-${log.timestamp}`;
                if (logsProcessedRef.current.has(logId)) return false;
                logsProcessedRef.current.add(logId);
                return true;
              })
              .map(missionLogToLogEntry);
            
            if (newLogs.length > 0) {
              setLogs(prev => [...prev, ...newLogs]);
            }
          }
        });

        websocketService.on('complete', () => {
          setIsRunning(false);
          setMissionComplete(true);
          if (currentMissionId) {
            getMissionStatus(currentMissionId).then(setMissionState).catch(console.error);
          }
        });

        websocketService.on('message', (message: any) => {
          if (message.type === 'log' && message.data?.message) {
            const logId = `${message.mission_id}-${Date.now()}`;
            if (logsProcessedRef.current.has(logId)) return;
            logsProcessedRef.current.add(logId);
            
            const logEntry: LogEntry = {
              id: logId,
              timestamp: new Date().toLocaleTimeString(),
              agent: 'Planner',
              message: message.data.message || message.message || '',
              type: message.data.level || 'info',
            };
            setLogs(prev => [...prev, logEntry]);
          }
        });
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
      }
    };

    connectWebSocket();

    return () => {
      websocketService.disconnect();
      logsProcessedRef.current.clear();
    };
  }, [currentMissionId]);

  const handleStartMission = async () => {
    try {
      setIsRunning(true);
      setMissionComplete(false);
      setLogs([]);
      setPath([]);
      setMissionState(null);
      pathHistoryRef.current = [];
      logsProcessedRef.current.clear();

      const response = await startMission(missionGoal);
      setCurrentMissionId(response.mission_id);
      
      const initialLog: LogEntry = {
        id: `${response.mission_id}-start`,
        timestamp: new Date().toLocaleTimeString(),
        agent: 'Planner',
        message: `Mission started: ${missionGoal}`,
        type: 'info',
      };
      logsProcessedRef.current.add(initialLog.id);
      setLogs([initialLog]);
    } catch (error) {
      console.error('Failed to start mission:', error);
      setLogs(prev => [...prev, {
        id: `error-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        agent: 'Planner',
        message: `Failed to start mission: ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'error',
      }]);
      setIsRunning(false);
    }
  };

  const handleStopMission = () => {
    setIsRunning(false);
    setMissionComplete(false);
    websocketService.disconnect();
  };

  const handleGenerateReport = () => {
    if (!currentMissionId) return;
    
    // Navigate immediately with existing data - no waiting for API call
    onNavigate('report', {
      missionGoal,
      logs,
      path,
      missionState,
      reportData: null, // Will be fetched on report page
      duration: missionState ? 
        `${Math.floor((new Date(missionState.updated_at).getTime() - new Date(missionState.created_at).getTime()) / 1000)}s` : 
        '0s',
      currentMissionId // Pass mission ID to fetch report in background
    });
    
    // Fetch report data in background (non-blocking) - will be used when PDF is generated
    getMissionReport(currentMissionId).then(() => {
      console.log('✅ Report data fetched in background');
    }).catch(error => {
      console.error('Failed to fetch mission report:', error);
      // Non-critical - report page can still work with existing data
    });
  };

  return (
    <main className="flex-1 overflow-y-auto relative">
      {/* APOD Background */}
      {apodBackground && apodBackground.image_url && (
        <div 
          className="fixed inset-0 -z-10 opacity-10 pointer-events-none"
          style={{
            backgroundImage: `url(${apodBackground.image_url})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundRepeat: 'no-repeat',
          }}
        />
      )}
      <div className="max-w-[1440px] mx-auto px-6 py-6 relative z-0">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar */}
          <aside className="lg:col-span-3 flex flex-col gap-6">
            <MissionInput
              missionGoal={missionGoal}
              onMissionGoalChange={setMissionGoal}
              onStartMission={handleStartMission}
              onStopMission={handleStopMission}
              isRunning={isRunning}
            />

            {missionComplete && (
              <Card>
                <h3 className="mb-4">Mission Complete</h3>
                <Button
                  onClick={handleGenerateReport}
                  variant="secondary"
                  size="md"
                  className="w-full"
                >
                  <FileDown size={20} aria-hidden="true" />
                  Generate Report
                </Button>
              </Card>
            )}
          </aside>

          {/* Center - Grid - FIXED HEIGHT, NO SCROLLING */}
          <section className="lg:col-span-6 flex flex-col gap-4">
            <Card style={{ height: '650px', overflow: 'hidden', position: 'relative', flexShrink: 0 }}>
              <h2 className="mb-4">Rover Simulation</h2>
              <div style={{ height: '600px', overflow: 'hidden', position: 'relative' }}>
                {!missionState ? (
                  <div className="text-center px-6 py-12">
                    <p className="text-[var(--color-text-secondary)] mb-4">
                      Enter a mission goal and start the mission to begin simulation
                    </p>
                    <p className="text-[var(--color-text-secondary)]">
                      <small>The AI agents will plan and execute your mission in real-time</small>
                    </p>
                  </div>
                ) : (
                  <div style={{ width: '100%', height: '100%', overflow: 'hidden', position: 'relative' }}>
                    <MissionCanvas
                      roverPosition={missionState.rover_position}
                      obstacles={missionState.obstacles || []}
                      goalPositions={missionState.goal_positions || []}
                      path={path}
                      currentStep={path.length - 1}
                    />
                  </div>
                )}
              </div>
            </Card>

          </section>

          {/* Right Sidebar - Logs - ONLY THIS SCROLLS */}
          <aside className="lg:col-span-3 flex flex-col" style={{ height: '650px', flexShrink: 0 }}>
            <AgentLogs logs={logs} />
          </aside>
        </div>
      </div>
    </main>
  );
}
