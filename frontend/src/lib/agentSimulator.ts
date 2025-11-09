import type { LogEntry } from '../components/AgentLogs';

interface Position {
  x: number;
  y: number;
}

export class AgentSimulator {
  private logs: LogEntry[] = [];
  private logCallback: (log: LogEntry) => void;
  private generatePath: (goal: string) => Position[];

  constructor(logCallback: (log: LogEntry) => void, pathGenerator: (goal: string) => Position[]) {
    this.logCallback = logCallback;
    this.generatePath = pathGenerator;
  }

  private addLog(agent: LogEntry['agent'], message: string, type: LogEntry['type'] = 'info') {
    const log: LogEntry = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date().toLocaleTimeString(),
      agent,
      message,
      type
    };
    this.logs.push(log);
    this.logCallback(log);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async runMission(goal: string): Promise<Position[]> {
    this.logs = [];
    
    // Planner phase
    this.addLog('Planner', 'Mission received. Analyzing goal...', 'info');
    await this.sleep(800);
    
    this.addLog('Planner', `Goal: ${goal}`, 'info');
    await this.sleep(600);
    
    const path = this.generatePath(goal);
    this.addLog('Planner', `Route calculated. ${path.length} waypoints identified.`, 'success');
    await this.sleep(800);
    
    // Safety phase
    this.addLog('Safety', 'Performing pre-mission safety checks...', 'info');
    await this.sleep(700);
    
    this.addLog('Safety', 'Terrain analysis: Clear', 'success');
    await this.sleep(500);
    
    this.addLog('Safety', 'Power levels: Optimal', 'success');
    await this.sleep(500);
    
    this.addLog('Safety', 'Communication link: Strong', 'success');
    await this.sleep(700);
    
    // Navigator phase
    this.addLog('Navigator', 'Initializing navigation system...', 'info');
    await this.sleep(600);
    
    this.addLog('Navigator', `Starting from position (${path[0].x}, ${path[0].y})`, 'info');
    await this.sleep(500);
    
    this.addLog('Navigator', `Target destination: (${path[path.length - 1].x}, ${path[path.length - 1].y})`, 'info');
    await this.sleep(700);
    
    this.addLog('Navigator', 'Mission execution started. Rover is moving...', 'success');
    
    return path;
  }

  async onTargetReached() {
    await this.sleep(500);
    this.addLog('Navigator', 'Target destination reached!', 'success');
    await this.sleep(600);
    
    this.addLog('Safety', 'Post-mission diagnostics running...', 'info');
    await this.sleep(800);
    
    this.addLog('Safety', 'All systems nominal. Mission successful.', 'success');
    await this.sleep(700);
    
    this.addLog('Reporter', 'Collecting mission data...', 'info');
    await this.sleep(600);
    
    this.addLog('Reporter', 'Requesting NASA rover imagery...', 'info');
  }

  getLogs(): LogEntry[] {
    return this.logs;
  }
}

export function generateRoverPath(goal: string): Position[] {
  // Extract coordinates from goal if present, or generate random target
  const coordMatch = goal.match(/\((\d+),\s*(\d+)\)/);
  let targetX: number, targetY: number;

  if (coordMatch) {
    targetX = Math.min(19, Math.max(0, parseInt(coordMatch[1])));
    targetY = Math.min(19, Math.max(0, parseInt(coordMatch[2])));
  } else {
    targetX = Math.floor(Math.random() * 15) + 5;
    targetY = Math.floor(Math.random() * 15) + 5;
  }

  const path: Position[] = [];
  const startX = 2;
  const startY = 2;

  // Simple pathfinding: move horizontally then vertically
  let currentX = startX;
  let currentY = startY;

  path.push({ x: currentX, y: currentY });

  // Move horizontally
  while (currentX !== targetX) {
    currentX += currentX < targetX ? 1 : -1;
    path.push({ x: currentX, y: currentY });
  }

  // Move vertically
  while (currentY !== targetY) {
    currentY += currentY < targetY ? 1 : -1;
    path.push({ x: currentX, y: currentY });
  }

  return path;
}
