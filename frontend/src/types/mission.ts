export enum AgentType {
  PLANNER = "planner",
  ROVER = "rover",
  SAFETY = "safety",
  REPORTER = "reporter",
  SUPERVISOR = "supervisor",
  SYSTEM = "system"
}

export enum AgentStatus {
  IDLE = "idle",
  PLANNING = "planning",
  EXECUTING = "executing",
  VALIDATING = "validating",
  REPORTING = "reporting",
  COMPLETE = "complete",
  ERROR = "error"
}

export enum MissionStatus {
  PENDING = "pending",
  PLANNING = "planning",
  EXECUTING = "executing",
  COMPLETE = "complete",
  ABORTED = "aborted",
  ERROR = "error"
}

export interface RoverPosition {
  x: number;
  y: number;
}

export interface MissionStep {
  step_number: number;
  action: string;
  target_position: RoverPosition | null;
  description: string;
  completed: boolean;
  nasa_image_url?: string;
}

export interface MissionLog {
  mission_id: string;
  timestamp: string;
  agent_type: AgentType;
  message: string;
  level: "info" | "warning" | "error" | "success";
  data?: Record<string, any>;
}

export interface MissionState {
  mission_id: string;
  goal: string;
  status: MissionStatus;
  current_step: number;
  rover_position: RoverPosition;
  obstacles: RoverPosition[];
  goal_positions: RoverPosition[];
  steps: MissionStep[];
  logs: MissionLog[];
  agent_states: Record<AgentType, AgentStatus>;
  nasa_images: string[];
  weather_data?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface MissionReport {
  mission_id: string;
  goal: string;
  status: MissionStatus;
  start_time: string;
  end_time?: string;
  duration_seconds?: number;
  steps_completed: number;
  total_steps: number;
  steps: MissionStep[];
  logs: MissionLog[];
  nasa_images: string[];
  weather_data?: Record<string, any>;
  summary: string;
  outcome: string;
}

export interface WebSocketMessage {
  type: "status" | "update" | "log" | "complete" | "error" | "pong" | "mission_status";
  mission_id: string;
  timestamp?: string;
  status?: MissionStatus | string;
  message?: string;
  data?: {
    rover_position?: RoverPosition;
    current_step?: number;
    total_steps?: number;
    agent_states?: Record<string, AgentStatus>;
    logs?: MissionLog[];
    status?: string;
    steps_completed?: number;
  };
}

export interface StartMissionRequest {
  goal: string;
}

export interface StartMissionResponse {
  mission_id: string;
  status: string;
  message: string;
}

