import React, { useRef } from 'react';
import { Button } from '../Button';
import { Card } from '../Card';
import { Download, CheckCircle, XCircle, ArrowLeft } from 'lucide-react';
import { LogEntry } from '../AgentLogs';
import { RoverPhoto } from '../../lib/nasaApi';
import { ImageWithFallback } from '../figma/ImageWithFallback';

interface Position {
  x: number;
  y: number;
}

interface ReportData {
  missionGoal: string;
  logs: LogEntry[];
  path: Position[];
  roverPhoto: RoverPhoto | null;
  duration: string;
}

interface ReportPageProps {
  onNavigate: (page: string) => void;
  reportData: ReportData | null;
}

export function ReportPage({ onNavigate, reportData }: ReportPageProps) {
  const reportRef = useRef<HTMLDivElement>(null);

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

  const { missionGoal, logs, path, roverPhoto, duration } = reportData;
  const missionSuccess = logs.some(log => 
    log.message.includes('Mission successful') || log.message.includes('Target destination reached')
  );

  const handleDownload = () => {
    // Create a formatted text report
    const reportContent = `
OOPSROVEROPS - MISSION REPORT
=====================================

Mission Goal:
${missionGoal}

Status: ${missionSuccess ? 'SUCCESS' : 'INCOMPLETE'}
Duration: ${duration}
Waypoints: ${path.length}

AGENT ACTIVITY LOG:
-------------------
${logs.map(log => `[${log.timestamp}] ${log.agent}: ${log.message}`).join('\n')}

${roverPhoto ? `
NASA IMAGERY:
-------------
Rover: ${roverPhoto.rover.name}
Camera: ${roverPhoto.camera.full_name}
Date: ${roverPhoto.earth_date}
Image URL: ${roverPhoto.img_src}
` : ''}

Generated: ${new Date().toLocaleString()}
    `.trim();

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mission-report-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <main className="flex-1">
      <div className="max-w-[1440px] mx-auto px-6 py-8">
        <div className="mb-6">
          <Button
            onClick={() => onNavigate('dashboard')}
            variant="outline"
            size="sm"
            aria-label="Return to dashboard"
          >
            <ArrowLeft size={20} aria-hidden="true" />
            Back to Dashboard
          </Button>
        </div>

        <div ref={reportRef} className="space-y-6">
          {/* Header */}
          <header className="text-center mb-8">
            <h1 className="mb-4">Mission Report</h1>
            <div className="flex items-center justify-center gap-2">
              {missionSuccess ? (
                <>
                  <CheckCircle className="text-[var(--color-success)]" size={24} aria-hidden="true" />
                  <span className="text-[var(--color-success)]">Mission Successful</span>
                </>
              ) : (
                <>
                  <XCircle className="text-[var(--color-error)]" size={24} aria-hidden="true" />
                  <span className="text-[var(--color-error)]">Mission Incomplete</span>
                </>
              )}
            </div>
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
                    <dd className={missionSuccess ? 'text-[var(--color-success)]' : 'text-[var(--color-error)]'}>
                      {missionSuccess ? 'Success' : 'Incomplete'}
                    </dd>
                  </div>
                </div>
              </dl>
            </Card>
          </section>

          {/* NASA Imagery */}
          {roverPhoto && (
            <section aria-labelledby="imagery-heading">
              <Card>
                <h2 id="imagery-heading" className="mb-4">Captured NASA Rover Imagery</h2>
                <ImageWithFallback
                  src={roverPhoto.img_src}
                  alt={`Mars rover photo captured by ${roverPhoto.rover.name} on ${roverPhoto.earth_date}`}
                  className="w-full h-auto rounded-lg mb-3"
                />
                <p className="text-[var(--color-text-secondary)]">
                  <small>
                    Rover: {roverPhoto.rover.name} • Camera: {roverPhoto.camera.full_name} • Date: {roverPhoto.earth_date}
                  </small>
                </p>
              </Card>
            </section>
          )}

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
              onClick={handleDownload}
              variant="primary"
              size="lg"
              aria-label="Download mission report"
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