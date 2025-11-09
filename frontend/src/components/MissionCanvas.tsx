import { useRef, useEffect } from 'react';
import type { RoverPosition } from '../types/mission';

interface MissionCanvasProps {
  roverPosition: RoverPosition;
  obstacles: RoverPosition[];
  goalPositions: RoverPosition[];
  path: RoverPosition[];
  currentStep?: number;
}

const GRID_SIZE = 10;
const CANVAS_SIZE = 600;
const LABEL_SIZE = 30;

export function MissionCanvas({
  roverPosition,
  obstacles,
  goalPositions,
  path,
  currentStep = 0
}: MissionCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const totalWidth = CANVAS_SIZE + LABEL_SIZE;
    const totalHeight = CANVAS_SIZE + LABEL_SIZE;
    const cellSize = CANVAS_SIZE / GRID_SIZE;

    // Set canvas size
    canvas.width = totalWidth;
    canvas.height = totalHeight;

    // Clear entire canvas
    ctx.fillStyle = '#0A0A0F';
    ctx.fillRect(0, 0, totalWidth, totalHeight);

    // Draw label backgrounds
    ctx.fillStyle = '#0A0A0F';
    ctx.fillRect(0, 0, LABEL_SIZE, totalHeight); // Left column
    ctx.fillRect(0, 0, totalWidth, LABEL_SIZE); // Top row

    // Draw X-axis labels (0-9) at top
    ctx.fillStyle = '#a0aec0';
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < GRID_SIZE; i++) {
      ctx.fillText(
        i.toString(),
        LABEL_SIZE + i * cellSize + cellSize / 2,
        LABEL_SIZE / 2
      );
    }

    // Draw Y-axis labels (0-9) on left
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < GRID_SIZE; i++) {
      ctx.fillText(
        i.toString(),
        LABEL_SIZE - 8,
        LABEL_SIZE + i * cellSize + cellSize / 2
      );
    }

    // Draw grid (offset by label size)
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID_SIZE; i++) {
      const x = LABEL_SIZE + i * cellSize;
      ctx.beginPath();
      ctx.moveTo(x, LABEL_SIZE);
      ctx.lineTo(x, LABEL_SIZE + CANVAS_SIZE);
      ctx.stroke();
      
      const y = LABEL_SIZE + i * cellSize;
      ctx.beginPath();
      ctx.moveTo(LABEL_SIZE, y);
      ctx.lineTo(LABEL_SIZE + CANVAS_SIZE, y);
      ctx.stroke();
    }

    // Draw obstacles (red squares)
    obstacles.forEach(obstacle => {
      const x = LABEL_SIZE + obstacle.x * cellSize + 2;
      const y = LABEL_SIZE + obstacle.y * cellSize + 2;
      ctx.fillStyle = '#FF4D4D';
      ctx.fillRect(x, y, cellSize - 4, cellSize - 4);
    });

    // Draw goal positions (cyan circles)
    goalPositions.forEach(goal => {
      const x = LABEL_SIZE + goal.x * cellSize + cellSize / 2;
      const y = LABEL_SIZE + goal.y * cellSize + cellSize / 2;
      ctx.strokeStyle = '#00CEC9';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(x, y, cellSize / 2 - 6, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Draw path (purple line)
    if (path.length > 1) {
      ctx.strokeStyle = '#6C5CE7';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const startX = LABEL_SIZE + path[0].x * cellSize + cellSize / 2;
      const startY = LABEL_SIZE + path[0].y * cellSize + cellSize / 2;
      ctx.moveTo(startX, startY);
      
      for (let i = 1; i <= Math.min(currentStep, path.length - 1); i++) {
        const x = LABEL_SIZE + path[i].x * cellSize + cellSize / 2;
        const y = LABEL_SIZE + path[i].y * cellSize + cellSize / 2;
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Draw start position (cyan dot)
    if (path.length > 0) {
      const x = LABEL_SIZE + path[0].x * cellSize + cellSize / 2;
      const y = LABEL_SIZE + path[0].y * cellSize + cellSize / 2;
      ctx.fillStyle = '#00CEC9';
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw rover (white square with green dot)
    const roverX = LABEL_SIZE + roverPosition.x * cellSize + cellSize / 2;
    const roverY = LABEL_SIZE + roverPosition.y * cellSize + cellSize / 2;
    
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(roverX - 12, roverY - 12, 24, 24);
    
    ctx.fillStyle = '#55EFC4';
    ctx.beginPath();
    ctx.arc(roverX, roverY, 5, 0, Math.PI * 2);
    ctx.fill();
  }, [roverPosition, obstacles, goalPositions, path, currentStep]);

  const totalSize = CANVAS_SIZE + LABEL_SIZE;

  return (
    <div 
      className="flex items-center justify-center w-full h-full" 
      style={{ 
        overflow: 'hidden',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0
      }}
    >
      <canvas
        ref={canvasRef}
        width={totalSize}
        height={totalSize}
        className="border-2 border-[var(--color-card)] rounded-lg"
        style={{ 
          display: 'block',
          maxWidth: '100%',
          maxHeight: '100%',
          objectFit: 'contain'
        }}
        role="img"
        aria-label="10x10 grid with coordinates, rover position, obstacles, and goal positions"
      />
    </div>
  );
}
