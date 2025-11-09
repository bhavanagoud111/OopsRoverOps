import React from 'react';
import { Card } from './Card';
import { Button } from './Button';
import { Play, StopCircle } from 'lucide-react';

interface MissionInputProps {
  missionGoal: string;
  onMissionGoalChange: (goal: string) => void;
  onStartMission: () => void;
  onStopMission: () => void;
  isRunning: boolean;
  disabled?: boolean;
}

export function MissionInput({
  missionGoal,
  onMissionGoalChange,
  onStartMission,
  onStopMission,
  isRunning,
  disabled
}: MissionInputProps) {
  return (
    <Card>
      <h2 className="mb-4">Mission Control</h2>
      
      <div className="space-y-4">
        <div>
          <label 
            htmlFor="mission-goal" 
            className="block mb-2 text-[var(--color-text-primary)]"
          >
            Enter Mission Goal
          </label>
          <textarea
            id="mission-goal"
            value={missionGoal}
            onChange={(e) => onMissionGoalChange(e.target.value)}
            disabled={isRunning || disabled}
            placeholder="Example: Navigate to coordinates (15, 15) and collect rock sample"
            className="w-full h-32 px-4 py-3 bg-[var(--color-background)] border border-[var(--color-card)] rounded-lg text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent disabled:opacity-50 resize-none"
            aria-describedby="mission-goal-description"
          />
          <p id="mission-goal-description" className="mt-2 text-[var(--color-text-secondary)]">
            <small>Describe what you want the rover to accomplish</small>
          </p>
        </div>

        <div className="flex gap-3">
          {!isRunning ? (
            <Button
              onClick={onStartMission}
              disabled={!missionGoal.trim() || disabled}
              variant="primary"
              size="lg"
              className="flex-1"
              aria-label="Start mission"
            >
              <Play size={20} aria-hidden="true" />
              Start Mission
            </Button>
          ) : (
            <Button
              onClick={onStopMission}
              variant="outline"
              size="lg"
              className="flex-1"
              aria-label="Stop mission"
            >
              <StopCircle size={20} aria-hidden="true" />
              Stop Mission
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
