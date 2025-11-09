import React, { useState } from 'react';
import { MCOSButton } from '../MCOSButton';
import { Rocket } from 'lucide-react';
import { MapCanvas } from '../MapCanvas';

interface MissionSetupProps {
  onMissionCreated: () => void;
}

export function MissionSetup({ onMissionCreated }: MissionSetupProps) {
  const [missionGoal, setMissionGoal] = useState('');
  const [safetyRadius, setSafetyRadius] = useState('50');
  const [startCoords, setStartCoords] = useState('14.62, 175.48');
  const [endCoords, setEndCoords] = useState('14.85, 175.72');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onMissionCreated();
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 600px',
        gap: 'var(--mcos-space-4)',
        height: '100%',
        padding: 'var(--mcos-space-3)',
        overflow: 'auto',
      }}
    >
      {/* Left: Form */}
      <div
        style={{
          background: 'var(--mcos-surface)',
          border: '1px solid var(--mcos-surface-2)',
          borderRadius: 'var(--mcos-radius-lg)',
          padding: 'var(--mcos-space-4)',
          boxShadow: 'var(--mcos-shadow-md)',
          height: 'fit-content',
        }}
      >
        <h2 style={{ marginBottom: 'var(--mcos-space-3)', color: 'var(--mcos-text)' }}>
          Create New Mission
        </h2>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--mcos-space-3)' }}>
          {/* Mission Goal */}
          <div>
            <label
              htmlFor="mission-goal"
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '14px',
                fontWeight: 600,
                color: 'var(--mcos-text)',
              }}
            >
              Mission Goal
            </label>
            <textarea
              id="mission-goal"
              value={missionGoal}
              onChange={(e) => setMissionGoal(e.target.value)}
              placeholder="e.g., Explore crater region and collect soil samples"
              style={{
                width: '100%',
                minHeight: '100px',
                padding: 'var(--mcos-space-2)',
                background: 'var(--mcos-bg)',
                border: '1px solid var(--mcos-surface-2)',
                borderRadius: 'var(--mcos-radius-md)',
                color: 'var(--mcos-text)',
                fontSize: '14px',
                fontFamily: 'var(--font-sans)',
                resize: 'vertical',
              }}
              required
            />
            <small style={{ color: 'var(--mcos-muted)', fontSize: '12px' }}>
              Describe the objectives and expected outcomes
            </small>
          </div>

          {/* Start Coordinates */}
          <div>
            <label
              htmlFor="start-coords"
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '14px',
                fontWeight: 600,
                color: 'var(--mcos-text)',
              }}
            >
              Start Coordinates
            </label>
            <input
              id="start-coords"
              type="text"
              value={startCoords}
              onChange={(e) => setStartCoords(e.target.value)}
              placeholder="Latitude, Longitude"
              style={{
                width: '100%',
                padding: 'var(--mcos-space-2)',
                background: 'var(--mcos-bg)',
                border: '1px solid var(--mcos-surface-2)',
                borderRadius: 'var(--mcos-radius-md)',
                color: 'var(--mcos-text)',
                fontSize: '14px',
                fontFamily: 'var(--font-mono)',
              }}
              required
            />
          </div>

          {/* End Coordinates */}
          <div>
            <label
              htmlFor="end-coords"
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '14px',
                fontWeight: 600,
                color: 'var(--mcos-text)',
              }}
            >
              End Coordinates
            </label>
            <input
              id="end-coords"
              type="text"
              value={endCoords}
              onChange={(e) => setEndCoords(e.target.value)}
              placeholder="Latitude, Longitude"
              style={{
                width: '100%',
                padding: 'var(--mcos-space-2)',
                background: 'var(--mcos-bg)',
                border: '1px solid var(--mcos-surface-2)',
                borderRadius: 'var(--mcos-radius-md)',
                color: 'var(--mcos-text)',
                fontSize: '14px',
                fontFamily: 'var(--font-mono)',
              }}
              required
            />
          </div>

          {/* Safety Radius */}
          <div>
            <label
              htmlFor="safety-radius"
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '14px',
                fontWeight: 600,
                color: 'var(--mcos-text)',
              }}
            >
              Safety Radius (meters)
            </label>
            <input
              id="safety-radius"
              type="number"
              value={safetyRadius}
              onChange={(e) => setSafetyRadius(e.target.value)}
              min="10"
              max="1000"
              step="10"
              style={{
                width: '100%',
                padding: 'var(--mcos-space-2)',
                background: 'var(--mcos-bg)',
                border: '1px solid var(--mcos-surface-2)',
                borderRadius: 'var(--mcos-radius-md)',
                color: 'var(--mcos-text)',
                fontSize: '14px',
                fontFamily: 'var(--font-mono)',
              }}
              required
            />
            <small style={{ color: 'var(--mcos-muted)', fontSize: '12px' }}>
              Minimum safe distance from hazards
            </small>
          </div>

          {/* Submit Button */}
          <MCOSButton variant="primary" icon={Rocket} type="submit">
            Create Mission
          </MCOSButton>
        </form>
      </div>

      {/* Right: Preview Map */}
      <div
        style={{
          position: 'sticky',
          top: 'var(--mcos-space-3)',
          height: 'calc(100vh - 128px)',
        }}
      >
        <div
          style={{
            background: 'var(--mcos-surface)',
            border: '1px solid var(--mcos-surface-2)',
            borderRadius: 'var(--mcos-radius-lg)',
            padding: 'var(--mcos-space-3)',
            boxShadow: 'var(--mcos-shadow-md)',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <h3 style={{ marginBottom: 'var(--mcos-space-2)', color: 'var(--mcos-text)' }}>
            Mission Preview
          </h3>
          <div style={{ flex: 1, minHeight: 0 }}>
            <MapCanvas
              roverPosition={{ x: 150, y: 300 }}
              waypoints={[
                { x: 350, y: 200 },
                { x: 550, y: 350 },
              ]}
              path={[]}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
