import { useRef, useState, useEffect } from 'react';
import { Button } from '../Button';
import { Card } from '../Card';
import { Download, CheckCircle, XCircle, ArrowLeft } from 'lucide-react';
import type { LogEntry } from '../AgentLogs';
import { generatePDF } from '../../services/pdfGenerator';
import { getMissionReport, abortMission } from '../../services/api';
import { websocketService } from '../../services/websocket';
import { MissionStatus } from '../../types/mission';

interface Position {
  x: number;
  y: number;
}

interface ReportData {
  missionGoal: string;
  logs: LogEntry[];
  path: Position[];
  duration: string;
  missionState?: any;
  reportData?: any;
  currentMissionId?: string;
}

interface ReportPageProps {
  onNavigate: (page: string) => void;
  reportData: ReportData | null;
}

export function ReportPage({ onNavigate, reportData }: ReportPageProps) {
  const reportRef = useRef<HTMLDivElement>(null);
  const [fetchedReportData, setFetchedReportData] = useState<any>(null);
  // CRITICAL: Local state for mission status - driven by websocket and report data
  const [missionStatus, setMissionStatus] = useState<"in_progress" | "complete" | "aborted">("in_progress");

  // Fetch report data in background if mission ID is provided
  useEffect(() => {
    if (reportData?.currentMissionId) {
      // Always fetch fresh report data to get latest status
      getMissionReport(reportData.currentMissionId)
        .then(data => {
          setFetchedReportData(data);
          console.log('✅ Report data fetched:', {
            mission_status: data?.mission_status,
            status: data?.status,
            steps_completed: data?.steps_completed,
            total_steps: data?.total_steps,
            rover_pos: data?.rover_final_position
          });
          
          // CRITICAL: Immediately update status if we have it
          if (data?.mission_status === "complete") {
            setMissionStatus("complete");
          } else if (data?.mission_status === "aborted") {
            setMissionStatus("aborted");
          } else if (data?.status === "complete" || data?.status === MissionStatus.COMPLETE) {
            setMissionStatus("complete");
          } else if (data?.status === "aborted" || data?.status === MissionStatus.ABORTED) {
            setMissionStatus("aborted");
          } else if (data?.steps_completed && data?.total_steps && data?.steps_completed >= data?.total_steps) {
            // All steps done - check if rover is at base
            const roverPos = data?.rover_final_position || data?.rover_position;
            if (roverPos && roverPos.x === 0 && roverPos.y === 0) {
              console.log('✅ All steps done and rover at base - setting complete');
              setMissionStatus("complete");
            }
          }
        })
        .catch(error => {
          console.error('Failed to fetch mission report:', error);
        });
    }
  }, [reportData?.currentMissionId]);

  // CRITICAL: Hydrate missionStatus from report.mission_status field
  useEffect(() => {
    const actualReportData = fetchedReportData || reportData?.reportData;
    
    console.log('🔍 Checking mission status from:', {
      fetchedReportData: fetchedReportData?.mission_status || fetchedReportData?.status,
      reportData: reportData?.reportData?.mission_status || reportData?.reportData?.status,
      missionState: reportData?.missionState?.status,
      actualReportData: actualReportData?.mission_status || actualReportData?.status
    });
    
    // Check mission_status field first (most reliable)
    if (actualReportData?.mission_status === "complete") {
      console.log('✅ Setting status to complete from mission_status field');
      setMissionStatus("complete");
      return;
    } else if (actualReportData?.mission_status === "aborted") {
      console.log('❌ Setting status to aborted from mission_status field');
      setMissionStatus("aborted");
      return;
    }
    
    // Check status field as fallback
    const statusValue = actualReportData?.status;
    if (statusValue === "complete" || statusValue === MissionStatus.COMPLETE || statusValue === "COMPLETE" || statusValue === "executing") {
      // CRITICAL: If status is "executing" but mission appears done, check other indicators
      if (statusValue === "executing") {
        // Check if mission is actually complete by looking at steps and rover position
        const stepsCompleted = actualReportData?.steps_completed || actualReportData?.completed_steps;
        const totalSteps = actualReportData?.total_steps || actualReportData?.steps?.length;
        const roverPos = actualReportData?.rover_final_position || actualReportData?.rover_position;
        
        if (stepsCompleted && totalSteps && stepsCompleted >= totalSteps) {
          if (roverPos && roverPos.x === 0 && roverPos.y === 0) {
            console.log('✅ Mission appears complete (all steps done, rover at base)');
            setMissionStatus("complete");
            return;
          }
        }
      } else {
        console.log('✅ Setting status to complete from status field');
        setMissionStatus("complete");
        return;
      }
    } else if (statusValue === "aborted" || statusValue === MissionStatus.ABORTED || statusValue === "ABORTED") {
      console.log('❌ Setting status to aborted from status field');
      setMissionStatus("aborted");
      return;
    }
    
    // Check missionState status as last resort
    const missionStateStatus = reportData?.missionState?.status;
    if (missionStateStatus === "complete" || missionStateStatus === MissionStatus.COMPLETE || missionStateStatus === "COMPLETE") {
      console.log('✅ Setting status to complete from missionState');
      setMissionStatus("complete");
      return;
    } else if (missionStateStatus === "aborted" || missionStateStatus === MissionStatus.ABORTED || missionStateStatus === "ABORTED") {
      console.log('❌ Setting status to aborted from missionState');
      setMissionStatus("aborted");
      return;
    }
    
    // CRITICAL: If we have report data but no explicit status, check if mission appears complete
    // (all steps completed, rover at base, etc.)
    if (actualReportData) {
      const stepsCompleted = actualReportData.steps_completed || actualReportData.completed_steps;
      const totalSteps = actualReportData.total_steps || actualReportData.steps?.length;
      const roverPos = actualReportData.rover_final_position || actualReportData.rover_position;
      
      console.log('🔍 Checking completion indicators:', {
        stepsCompleted,
        totalSteps,
        roverPos,
        allStepsDone: stepsCompleted && totalSteps && stepsCompleted >= totalSteps,
        atBase: roverPos && roverPos.x === 0 && roverPos.y === 0
      });
      
      // If all steps completed and rover is at base, assume complete
      if (stepsCompleted && totalSteps && stepsCompleted >= totalSteps) {
        if (roverPos && roverPos.x === 0 && roverPos.y === 0) {
          console.log('✅ Inferring complete: all steps done and rover at base');
          setMissionStatus("complete");
          return;
        }
      }
      
      // Also check if we have a state field that indicates completion
      if (actualReportData.state?.status === "complete" || actualReportData.state?.status === MissionStatus.COMPLETE) {
        console.log('✅ Setting status to complete from state field');
        setMissionStatus("complete");
        return;
      }
    }
    
    // CRITICAL: If we're on the report page and mission appears done, default to complete
    // This handles cases where backend status might not be set correctly
    if (actualReportData) {
      const stepsCompleted = actualReportData.steps_completed || actualReportData.completed_steps || 0;
      const totalSteps = actualReportData.total_steps || actualReportData.steps?.length || 0;
      const roverPos = actualReportData.rover_final_position || actualReportData.rover_position;
      
      // If we have steps and all are done, assume complete (unless explicitly aborted)
      if (totalSteps > 0 && stepsCompleted >= totalSteps) {
        // If rover is at base, definitely complete
        if (roverPos && roverPos.x === 0 && roverPos.y === 0) {
          console.log('✅ All steps done and rover at base - forcing complete');
          setMissionStatus("complete");
          return;
        }
        // Even if not at base, if all steps done and not aborted, assume complete
        // (rover might have completed mission tasks)
        if (actualReportData.status !== "aborted" && actualReportData.mission_status !== "aborted") {
          console.log('✅ All steps done - assuming complete');
          setMissionStatus("complete");
          return;
        }
      }
    }
    
    console.log('⚠️ No completion indicators found, keeping status as in_progress');
  }, [fetchedReportData, reportData?.reportData, reportData?.missionState]);

  // CRITICAL: Listen for mission_status frames from websocket
  useEffect(() => {
    if (!reportData?.currentMissionId) return;

    const connectWebSocket = async () => {
      try {
        await websocketService.connect(reportData.currentMissionId!);
        
        // Listen for mission_status frames
        const unsubscribe = websocketService.on('mission_status', (message: any) => {
          if (message.data?.status) {
            const status = message.data.status;
            if (status === "complete" || status === "aborted") {
              setMissionStatus(status);
              console.log('✅ Mission status updated from websocket:', status);
            }
          }
        });

        // Cleanup on unmount
        return () => {
          unsubscribe();
          websocketService.disconnect();
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
      }
    };

    connectWebSocket();
  }, [reportData?.currentMissionId]);

  if (!reportData) {
    return (
      <main className="flex-1 flex items-center justify-center">
        <Card className="text-center max-w-md">
          <h2 className="mb-4">No Mission Data</h2>
          <p className="text-[var(--color-text-secondary)] mb-6">
            Complete a mission first to generate a report.
          </p>
          <Button onClick={() => onNavigate('dashboard')} variant="primary">
            Go to Dashboard
          </Button>
        </Card>
      </main>
    );
  }

  const { missionGoal, logs, path, duration, missionState, reportData: reportDataFromBackend } = reportData;
  
  // Use fetched report data if available, otherwise use passed data
  const actualReportData = fetchedReportData || reportDataFromBackend;
  
  // CRITICAL: Derive status label and tone strictly from missionStatus state
  // Never default to "Incomplete" unless status is aborted
  const statusLabel = missionStatus === "complete"
    ? "Mission Complete"
    : missionStatus === "aborted"
    ? "Mission Incomplete"
    : "In Progress";
  
  const statusTone = missionStatus === "complete" ? "success"
    : missionStatus === "aborted" ? "danger"
    : "neutral";

  const handleDownload = () => {
    try {
      console.log('📄 Generating PDF report...');
      
      // CRITICAL FIX: Use actualReportData (correct mission) not missionState (might be current mission)
      // Always use the report data from backend which has the correct completed mission data
      let missionData: any = null;
      
      if (actualReportData && actualReportData.state) {
        // Use state from backend report - this is the correct completed mission
        missionData = actualReportData.state;
        
        // CRITICAL: Check for abort status first - this takes priority
        const isAborted = 
          missionData.status === 'aborted' || 
          missionData.status === 'abort' ||
          actualReportData.mission_status === 'aborted' ||
          actualReportData.mission_status === 'abort' ||
          String(actualReportData.mission_status || '').toLowerCase() === 'aborted';
        
        // CRITICAL: If mission is aborted, preserve abort status
        if (isAborted) {
          missionData.status = 'aborted';
          console.log('⚠️ Mission is aborted - preserving abort status');
        } else {
          // Override status if all steps are completed and NOT aborted
          const stepsCompleted = missionData.steps?.filter((s: any) => s.completed).length || 0;
          const totalSteps = missionData.steps?.length || 0;
          if (totalSteps > 0 && stepsCompleted >= totalSteps) {
            missionData.status = 'complete';
          } else if (missionData.status === 'error') {
            // CRITICAL: Never pass "error" status - map to aborted if not complete
            missionData.status = 'aborted';
            console.log('⚠️ Converting error status to aborted');
          }
        }
        console.log('✅ Using backend report state');
      } else if (actualReportData) {
        // Fallback: create from report data
        // CRITICAL: Check for abort status first - this takes priority
        const isAborted = 
          actualReportData.mission_status === 'aborted' ||
          actualReportData.mission_status === 'abort' ||
          actualReportData.status === 'aborted' ||
          actualReportData.status === 'abort' ||
          missionStatus === "aborted" ||
          String(actualReportData.mission_status || actualReportData.status || '').toLowerCase() === 'aborted';
        
        let derivedStatus: string;
        if (isAborted) {
          // Mission is aborted - preserve abort status
          derivedStatus = 'aborted';
          console.log('⚠️ Mission is aborted - using abort status');
        } else {
          // Use mission_status if available, otherwise derive from steps
          derivedStatus = actualReportData.mission_status || actualReportData.status || '';
          if (!derivedStatus || derivedStatus === 'error') {
            // Check if all steps are completed
            const stepsCompleted = actualReportData.steps_completed || actualReportData.completed_steps || 0;
            const totalSteps = actualReportData.total_steps || actualReportData.steps?.length || 0;
            if (totalSteps > 0 && stepsCompleted >= totalSteps) {
              derivedStatus = 'complete';
            } else {
              // CRITICAL: Never use "error" - map to aborted if not complete
              // Use missionStatus if available, otherwise default to aborted
              if (missionStatus === "complete") {
                derivedStatus = 'complete';
              } else {
                derivedStatus = 'aborted'; // Default to aborted for error/unknown/in_progress status
              }
              console.log('⚠️ Converting error/unknown status to aborted');
            }
          }
          // CRITICAL: If status is still "error", convert to aborted
          if (derivedStatus === 'error') {
            derivedStatus = 'aborted';
            console.log('⚠️ Converting error status to aborted');
          }
        }
        
        missionData = {
          mission_id: actualReportData.mission_id || `mission-${Date.now()}`,
          goal: actualReportData.goal || missionGoal || 'Mission completed',
          status: derivedStatus,
          current_step: actualReportData.steps_completed || path.length,
          rover_position: path.length > 0 ? path[path.length - 1] : { x: 0, y: 0 },
          obstacles: [],
          goal_positions: [],
          steps: actualReportData.steps || path.map((pos, idx) => ({
            step_number: idx + 1,
            action: 'move',
            target_position: pos,
            description: `Move to position (${pos.x}, ${pos.y})`,
            completed: true,
          })),
          logs: actualReportData.logs || logs.map(log => ({
            mission_id: actualReportData.mission_id || '',
            timestamp: new Date().toISOString(),
            agent_type: log.agent.toLowerCase(),
            message: log.message,
            level: log.type,
          })),
          agent_states: {},
          nasa_images: actualReportData.mission_photos || [],
          created_at: actualReportData.start_time || new Date().toISOString(),
          updated_at: actualReportData.end_time || new Date().toISOString(),
        };
        console.log('✅ Using report data fallback');
      } else if (missionState) {
        // Use missionState if available
        missionData = missionState;
        console.log('✅ Using missionState');
      } else {
        // Last resort: use current data
        missionData = {
          mission_id: `mission-${Date.now()}`,
          goal: missionGoal || 'Mission completed',
          status: missionStatus === "complete" ? 'complete' : missionStatus === "aborted" ? 'aborted' : 'in_progress',
          current_step: path.length,
          rover_position: path.length > 0 ? path[path.length - 1] : { x: 0, y: 0 },
          obstacles: [],
          goal_positions: [],
          steps: path.map((pos, idx) => ({
            step_number: idx + 1,
            action: 'move',
            target_position: pos,
            description: `Move to position (${pos.x}, ${pos.y})`,
            completed: true,
          })),
          logs: logs.map(log => ({
            mission_id: '',
            timestamp: new Date().toISOString(),
            agent_type: log.agent.toLowerCase(),
            message: log.message,
            level: log.type,
          })),
          agent_states: {},
          nasa_images: [],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        console.log('✅ Using minimal state');
      }
      
      // Ensure required fields exist
      if (!missionData.mission_id) {
        missionData.mission_id = `mission-${Date.now()}`;
      }
      if (!missionData.goal) {
        missionData.goal = missionGoal || 'Mission completed';
      }
      if (!missionData.steps) {
        missionData.steps = [];
      }
      if (!missionData.logs) {
        missionData.logs = [];
      }
      
      console.log('📥 Calling generatePDF with:', {
        mission_id: missionData.mission_id,
        goal: missionData.goal,
        steps: missionData.steps?.length || 0,
        logs: missionData.logs?.length || 0,
      });
      
      // Generate and download PDF
      generatePDF(missionData);
      
      console.log('✅ PDF generated successfully');
    } catch (error) {
      console.error('❌ Error generating PDF:', error);
      alert('Failed to generate PDF. Please check the console for details.');
    }
  };

  return (
    <main className="flex-1">
      <div className="max-w-[1440px] mx-auto px-6 py-8">
        <div className="mb-6 flex justify-between items-center">
          <Button
            onClick={() => onNavigate('dashboard')}
            variant="outline"
            size="sm"
            aria-label="Return to dashboard"
          >
            <ArrowLeft size={20} aria-hidden="true" />
            Back to Dashboard
          </Button>
          {missionStatus === "in_progress" && reportData?.currentMissionId && (
            <Button
              onClick={async () => {
                try {
                  await abortMission(reportData.currentMissionId!);
                  console.log('✅ Mission aborted');
                } catch (error) {
                  console.error('Failed to abort mission:', error);
                }
              }}
              variant="outline"
              size="sm"
              aria-label="Abort mission"
            >
              Abort Mission
            </Button>
          )}
        </div>

        <div ref={reportRef} className="space-y-6">
          {/* Header */}
          <header className="text-center mb-8">
            <h1 className="mb-4">Mission Report</h1>
            <p className={`status ${statusTone} flex items-center justify-center gap-2`}>
              {missionStatus === "complete" ? (
                <>
                  <CheckCircle className="text-[var(--color-success)]" size={24} aria-hidden="true" />
                  <span className="text-[var(--color-success)]">{statusLabel}</span>
                </>
              ) : missionStatus === "aborted" ? (
                <>
                  <XCircle className="text-[var(--color-error)]" size={24} aria-hidden="true" />
                  <span className="text-[var(--color-error)]">{statusLabel}</span>
                </>
              ) : (
                <span>{statusLabel}</span>
              )}
            </p>
          </header>

          {/* Mission Summary */}
          <section aria-labelledby="summary-heading">
            <Card>
              <h2 id="summary-heading" className="mb-4">Mission Summary</h2>
              <dl className="space-y-3">
                <div>
                  <dt className="text-[var(--color-text-secondary)] mb-1">
                    <small>Mission Goal</small>
                  </dt>
                  <dd className="text-[var(--color-text-primary)]">{missionGoal}</dd>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <dt className="text-[var(--color-text-secondary)] mb-1">
                      <small>Waypoints</small>
                    </dt>
                    <dd className="text-[var(--color-text-primary)]">{path.length}</dd>
                  </div>
                  <div>
                    <dt className="text-[var(--color-text-secondary)] mb-1">
                      <small>Duration</small>
                    </dt>
                    <dd className="text-[var(--color-text-primary)]">{duration}</dd>
                  </div>
                  <div>
                    <dt className="text-[var(--color-text-secondary)] mb-1">
                      <small>Status</small>
                    </dt>
                    <dd className={missionStatus === "complete" ? 'text-[var(--color-success)]' : missionStatus === "aborted" ? 'text-[var(--color-error)]' : ''}>
                      {missionStatus === "complete" ? 'Complete' : missionStatus === "aborted" ? 'Incomplete' : 'In Progress'}
                    </dd>
                  </div>
                </div>
              </dl>
            </Card>
          </section>

          {/* Activity Log */}
          <section aria-labelledby="log-heading">
            <Card>
              <h2 id="log-heading" className="mb-4">Mission Activity Log</h2>
              <div className="space-y-2 monospace max-h-96 overflow-y-auto">
                {logs.map((log) => (
                  <div
                    key={log.id}
                    className="p-3 bg-[var(--color-background)] rounded"
                  >
                    <p>
                      <small>
                        <span className="text-[var(--color-text-secondary)]">[{log.timestamp}]</span>
                        {' '}
                        <span className="text-[var(--color-primary)]">{log.agent}:</span>
                        {' '}
                        {log.message}
                      </small>
                    </p>
                  </div>
                ))}
              </div>
            </Card>
          </section>

              {/* Download Button */}
              <div className="flex justify-center">
                <Button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🔘 Download button clicked');
                    handleDownload();
                  }}
                  variant="primary"
                  size="lg"
                  aria-label="Download mission report"
                  type="button"
                >
                  <Download size={20} aria-hidden="true" />
                  Download Report
                </Button>
              </div>
        </div>
      </div>
    </main>
  );
}