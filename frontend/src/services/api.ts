import type { StartMissionRequest, StartMissionResponse, MissionState } from '../types/mission';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function startMission(goal: string): Promise<StartMissionResponse> {
  const response = await fetch(`${API_URL}/api/mission/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ goal } as StartMissionRequest),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to start mission: ${response.statusText}`);
  }
  
  return response.json();
}

export async function getMissionStatus(missionId: string): Promise<MissionState> {
  const response = await fetch(`${API_URL}/api/mission/${missionId}`);
  
  if (!response.ok) {
    throw new Error(`Failed to get mission status: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data.state;
}

export async function getMissionReport(missionId: string): Promise<any> {
  const response = await fetch(`${API_URL}/api/mission/${missionId}/report`);
  
  if (!response.ok) {
    throw new Error(`Failed to get mission report: ${response.statusText}`);
  }
  
  return response.json();
}

export async function getAPOD(): Promise<any> {
  const response = await fetch(`${API_URL}/api/apod`);
  
  if (!response.ok) {
    throw new Error(`Failed to get APOD: ${response.statusText}`);
  }
  
  return response.json();
}

export async function abortMission(missionId: string): Promise<any> {
  const response = await fetch(`${API_URL}/mission/abort`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ mission_id: missionId }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to abort mission: ${response.statusText}`);
  }
  
  return response.json();
}

