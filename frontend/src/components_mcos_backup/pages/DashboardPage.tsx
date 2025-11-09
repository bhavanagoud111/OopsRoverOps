import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Button } from '../Button';
import { Card } from '../Card';
import { MissionInput } from '../MissionInput';
import { AgentLogs, LogEntry } from '../AgentLogs';
import { EnhancedRoverCanvas } from '../EnhancedRoverCanvas';
import { fetchRoverPhoto, RoverPhoto } from '../../lib/nasaApi';
import { FileDown, Loader2 } from 'lucide-react';
import { ImageWithFallback } from '../figma/ImageWithFallback';
import { startMission, getMissionStatus, getMissionReport } from '../../services/api';
import { websocketService } from '../../services/websocket';
import { MissionState, MissionLog, AgentType, MissionStatus } from '../../types/mission';

interface Position {
  x: number;
  y: number;
}

interface DashboardPageProps {
  onNavigate: (page: string, data?: any) => void;
}

// Helper to convert MissionLog to LogEntry
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
  const [roverPhoto, setRoverPhoto] = useState<RoverPhoto | null>(null);
  const [isLoadingPhoto, setIsLoadingPhoto] = useState(false);
  const [currentMissionId, setCurrentMissionId] = useState<string | null>(null);
  const [missionState, setMissionState] = useState<MissionState | null>(null);
  const pathHistoryRef = useRef<Position[]>([]);

  // Update path from mission state
  useEffect(() => {
    if (missionState?.rover_position) {
      const newPosition: Position = {
        x: missionState.rover_position.x,
        y: missionState.rover_position.y,
      };
      
      // Add to path history if it's a new position
      const lastPosition = pathHistoryRef.current[pathHistoryRef.current.length - 1];
      if (!lastPosition || lastPosition.x !== newPosition.x || lastPosition.y !== newPosition.y) {
        pathHistoryRef.current = [...pathHistoryRef.current, newPosition];
        setPath([...pathHistoryRef.current]);
      }
    }
  }, [missionState?.rover_position]);

  // WebSocket connection and message handling
  useEffect(() => {
    if (!currentMissionId) return;

    const connectWebSocket = async () => {
      try {
        await websocketService.connect(currentMissionId);
        
        // Listen for status updates
        websocketService.on('status', (message: any) => {
          if (message.data?.rover_position) {
            const pos = message.data.rover_position;
            const newPosition: Position = { x: pos.x, y: pos.y };
            pathHistoryRef.current = [...pathHistoryRef.current, newPosition];
            setPath([...pathHistoryRef.current]);
          }
          
          // Update mission state if available
          if (message.data) {
            getMissionStatus(currentMissionId).then(setMissionState).catch(console.error);
          }
          
          if (message.status === MissionStatus.COMPLETE || message.status === 'complete') {
            setIsRunning(false);
            setMissionComplete(true);
          }
        });
        
        // Listen for update messages
        websocketService.on('update', (message: any) => {
          if (message.data?.rover_position) {
            const pos = message.data.rover_position;
            const newPosition: Position = { x: pos.x, y: pos.y };
            pathHistoryRef.current = [...pathHistoryRef.current, newPosition];
            setPath([...pathHistoryRef.current]);
          }
          
          if (message.data?.logs) {
            const newLogs = message.data.logs.map(missionLogToLogEntry);
            setLogs(prev => [...prev, ...newLogs]);
          }
          
          // Update mission state
          if (currentMissionId) {
            getMissionStatus(currentMissionId).then(setMissionState).catch(console.error);
          }
        });

        // Listen for log messages
        websocketService.on('log', (message: any) => {
          if (message.data?.logs) {
            const newLogs = message.data.logs.map(missionLogToLogEntry);
            setLogs(prev => [...prev, ...newLogs]);
          }
        });

        // Listen for complete messages
        websocketService.on('complete', (message: any) => {
          setIsRunning(false);
          setMissionComplete(true);
          
          // Fetch final mission state
          if (currentMissionId) {
            getMissionStatus(currentMissionId).then(setMissionState);
          }
        });

        // Listen for all messages
        websocketService.on('message', (message: any) => {
          if (message.type === 'log' && message.data?.message) {
            const logEntry: LogEntry = {
              id: `${message.mission_id}-${Date.now()}`,
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
    };
  }, [currentMissionId]);

  const handleStartMission = async () => {
    try {
      setIsRunning(true);
      setMissionComplete(false);
      setRoverPhoto(null);
      setLogs([]);
      setPath([]);
      pathHistoryRef.current = [];

      const response = await startMission(missionGoal);
      setCurrentMissionId(response.mission_id);
      
      // Add initial log
      setLogs([{
        id: `${response.mission_id}-start`,
        timestamp: new Date().toLocaleTimeString(),
        agent: 'Planner',
        message: `Mission started: ${missionGoal}`,
        type: 'info',
      }]);
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

  const handleTargetReached = useCallback(async () => {
    setIsRunning(false);
    
    setIsLoadingPhoto(true);
    const photo = await fetchRoverPhoto();
    setIsLoadingPhoto(false);
    
    if (photo) {
      setRoverPhoto(photo);
      setLogs(prev => [...prev, {
        id: `${Date.now()}-photo`,
        timestamp: new Date().toLocaleTimeString(),
        agent: 'Reporter',
        message: 'Mars rover imagery received successfully!',
        type: 'success'
      }]);
    }
    
    setMissionComplete(true);
  }, []);

  const handleGenerateReport = async () => {
    if (!currentMissionId) return;
    
    try {
      // Fetch mission report from RoverOps backend
      const reportData = await getMissionReport(currentMissionId);
      onNavigate('report', {
        missionGoal,
        logs,
        path,
        roverPhoto,
        missionState,
        reportData,
        duration: missionState ? 
          `${Math.floor((new Date(missionState.updated_at).getTime() - new Date(missionState.created_at).getTime()) / 1000)}s` : 
          '0s'
      });
    } catch (error) {
      console.error('Failed to fetch mission report:', error);
      // Fallback to basic report
      onNavigate('report', {
        missionGoal,
        logs,
        path,
        roverPhoto,
        missionState,
        duration: missionState ? 
          `${Math.floor((new Date(missionState.updated_at).getTime() - new Date(missionState.created_at).getTime()) / 1000)}s` : 
          '0s'
      });
    }
  };

  return (
    <main className="flex-1">
      <div className="max-w-[1440px] mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-200px)]">
          {/* Left Sidebar - Mission Input */}
          <aside className="lg:col-span-3 flex flex-col gap-6" aria-label="Mission controls">
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
                  aria-label="Generate mission report"
                >
                  <FileDown size={20} aria-hidden="true" />
                  Generate Report
                </Button>
              </Card>
            )}
          </aside>

          {/* Main Content - Enhanced Canvas */}
          <section 
            className="lg:col-span-6 flex flex-col gap-4"
            aria-label="Mission visualization"
          >
            <Card className="flex-1 flex flex-col min-h-0">
              <h2 className="mb-4">Rover Simulation</h2>
              <div className="flex-1 flex items-center justify-center overflow-hidden">
                {path.length === 0 ? (
                  <div className="text-center px-6 py-12">
                    <p className="text-[var(--color-text-secondary)] mb-4">
                      Enter a mission goal and start the mission to begin simulation
                    </p>
                    <p className="text-[var(--color-text-secondary)]">
                      <small>The AI agents will plan and execute your mission in real-time</small>
                    </p>
                  </div>
                ) : (
                  <EnhancedRoverCanvas
                    isRunning={isRunning}
                    onTargetReached={handleTargetReached}
                    path={path}
                  />
                )}
              </div>
            </Card>

            {/* NASA Image Display */}
            {(roverPhoto || isLoadingPhoto) && (
              <Card>
                <h3 className="mb-4">NASA Rover Imagery</h3>
                {isLoadingPhoto ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="animate-spin text-[var(--color-primary)]" size={40} aria-hidden="true" />
                    <span className="sr-only">Loading NASA imagery</span>
                  </div>
                ) : roverPhoto ? (
                  <div>
                    <ImageWithFallback
                      src={roverPhoto.img_src}
                      alt={`Mars rover photo captured by ${roverPhoto.rover.name} on ${roverPhoto.earth_date}`}
                      className="w-full h-auto rounded-lg"
                    />
                    <p className="mt-3 text-[var(--color-text-secondary)]">
                      <small>
                        Captured by {roverPhoto.rover.name} • {roverPhoto.camera.full_name} • {roverPhoto.earth_date}
                      </small>
                    </p>
                  </div>
                ) : null}
              </Card>
            )}
          </section>

          {/* Right Sidebar - Logs */}
          <aside className="lg:col-span-3 h-full" aria-label="Agent activity logs">
            <AgentLogs logs={logs} />
          </aside>
        </div>
      </div>
    </main>
  );
}