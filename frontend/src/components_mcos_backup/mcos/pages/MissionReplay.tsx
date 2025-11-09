import React, { useState } from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import { MCOSButton } from '../MCOSButton';
import { MapCanvas } from '../MapCanvas';
import { EventLogItem, EventLevel } from '../EventLogItem';

const mockEvents = [
  { id: '1', timestamp: '00:00:00', level: 'info' as EventLevel, message: 'Mission started' },
  { id: '2', timestamp: '00:02:15', level: 'info' as EventLevel, message: 'Waypoint 1 reached' },
  { id: '3', timestamp: '00:04:30', level: 'info' as EventLevel, message: 'Image captured' },
  { id: '4', timestamp: '00:06:45', level: 'warn' as EventLevel, message: 'Low battery warning' },
  { id: '5', timestamp: '00:08:00', level: 'info' as EventLevel, message: 'Waypoint 2 reached' },
  { id: '6', timestamp: '00:10:00', level: 'info' as EventLevel, message: 'Mission complete' },
];

export function MissionReplay() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState(1);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleProgressChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setProgress(Number(e.target.value));
  };

  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed);
  };

  // Calculate rover position based on progress
  const roverPosition = {
    x: 150 + (progress / 100) * 400,
    y: 300 - (progress / 100) * 100,
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 320px',
        gridTemplateRows: '1fr auto',
        gap: 'var(--mcos-space-3)',
        height: '100%',
        padding: 'var(--mcos-space-3)',
        overflow: 'hidden',
      }}
    >
      {/* Center: Map */}
      <div style={{ position: 'relative', minHeight: 0 }}>
        <MapCanvas
          roverPosition={roverPosition}
          waypoints={[
            { x: 350, y: 200 },
            { x: 550, y: 200 },
          ]}
          path={[
            { x: 150, y: 300 },
            { x: 350, y: 200 },
            { x: 550, y: 200 },
          ]}
        />
      </div>

      {/* Right: Event Timeline */}
      <div
        style={{
          gridRow: '1 / -1',
          background: 'var(--mcos-surface)',
          border: '1px solid var(--mcos-surface-2)',
          borderRadius: 'var(--mcos-radius-lg)',
          padding: 'var(--mcos-space-3)',
          boxShadow: 'var(--mcos-shadow-md)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <h3 style={{ marginBottom: 'var(--mcos-space-2)', color: 'var(--mcos-text)' }}>
          Mission Timeline
        </h3>
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--mcos-space-2)',
          }}
        >
          {mockEvents.map((event) => (
            <EventLogItem
              key={event.id}
              timestamp={event.timestamp}
              level={event.level}
              message={event.message}
            />
          ))}
        </div>
      </div>

      {/* Bottom: Playback Controls */}
      <div
        style={{
          background: 'var(--mcos-surface)',
          border: '1px solid var(--mcos-surface-2)',
          borderRadius: 'var(--mcos-radius-lg)',
          padding: 'var(--mcos-space-3)',
          boxShadow: 'var(--mcos-shadow-md)',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--mcos-space-3)' }}>
          {/* Progress Bar */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span className="mcos-mono" style={{ fontSize: '12px', color: 'var(--mcos-muted)' }}>
                00:00:00
              </span>
              <span className="mcos-mono" style={{ fontSize: '12px', color: 'var(--mcos-muted)' }}>
                00:10:00
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={progress}
              onChange={handleProgressChange}
              style={{
                width: '100%',
                height: '6px',
                borderRadius: '3px',
                background: `linear-gradient(to right, var(--mcos-primary) 0%, var(--mcos-primary) ${progress}%, var(--mcos-surface-2) ${progress}%, var(--mcos-surface-2) 100%)`,
                appearance: 'none',
                outline: 'none',
                cursor: 'pointer',
              }}
              aria-label="Mission playback progress"
            />
          </div>

          {/* Playback Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--mcos-space-2)' }}>
            <MCOSButton
              variant="secondary"
              size="sm"
              icon={SkipBack}
              onClick={() => setProgress(Math.max(0, progress - 10))}
              ariaLabel="Skip backward 10 seconds"
            >
              -10s
            </MCOSButton>

            <MCOSButton
              variant="primary"
              icon={isPlaying ? Pause : Play}
              onClick={handlePlayPause}
              ariaLabel={isPlaying ? 'Pause playback' : 'Play mission replay'}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </MCOSButton>

            <MCOSButton
              variant="secondary"
              size="sm"
              icon={SkipForward}
              onClick={() => setProgress(Math.min(100, progress + 10))}
              ariaLabel="Skip forward 10 seconds"
            >
              +10s
            </MCOSButton>

            {/* Speed Controls */}
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
              <span style={{ fontSize: '12px', color: 'var(--mcos-muted)', marginRight: '8px' }}>
                Speed:
              </span>
              {[0.25, 0.5, 1, 2, 4].map((s) => (
                <button
                  key={s}
                  onClick={() => handleSpeedChange(s)}
                  style={{
                    padding: '4px 8px',
                    background: speed === s ? 'var(--mcos-primary)' : 'transparent',
                    border: `1px solid ${speed === s ? 'var(--mcos-primary)' : 'var(--mcos-surface-2)'}`,
                    borderRadius: 'var(--mcos-radius-sm)',
                    color: speed === s ? 'white' : 'var(--mcos-muted)',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all var(--transition-base)',
                  }}
                  aria-label={`Set playback speed to ${s}x`}
                  aria-pressed={speed === s}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
