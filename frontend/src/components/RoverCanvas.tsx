import React, { useRef, useEffect, useState } from 'react';

interface Position {
  x: number;
  y: number;
}

interface RoverCanvasProps {
  isRunning: boolean;
  onTargetReached: () => void;
  path: Position[];
}

export function RoverCanvas({ isRunning, onTargetReached, path }: RoverCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const animationFrameRef = useRef<number>();
  const lastUpdateRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const gridSize = 20;
    const cellSize = canvas.width / gridSize;

    // Clear canvas
    ctx.fillStyle = '#0A0A0F';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= gridSize; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, canvas.height);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(canvas.width, i * cellSize);
      ctx.stroke();
    }

    // Draw path
    if (path.length > 0 && currentStep > 0) {
      ctx.strokeStyle = '#6C5CE7';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(path[0].x * cellSize + cellSize / 2, path[0].y * cellSize + cellSize / 2);
      
      for (let i = 1; i <= Math.min(currentStep, path.length - 1); i++) {
        ctx.lineTo(path[i].x * cellSize + cellSize / 2, path[i].y * cellSize + cellSize / 2);
      }
      ctx.stroke();
    }

    // Draw start position
    if (path.length > 0) {
      ctx.fillStyle = '#00CEC9';
      ctx.beginPath();
      ctx.arc(
        path[0].x * cellSize + cellSize / 2,
        path[0].y * cellSize + cellSize / 2,
        8,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }

    // Draw target position
    if (path.length > 0) {
      const target = path[path.length - 1];
      ctx.strokeStyle = '#FD79A8';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(
        target.x * cellSize + cellSize / 2,
        target.y * cellSize + cellSize / 2,
        12,
        0,
        Math.PI * 2
      );
      ctx.stroke();
    }

    // Draw rover (current position)
    if (path.length > 0 && currentStep < path.length) {
      const currentPos = path[currentStep];
      
      // Rover body
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(
        currentPos.x * cellSize + cellSize / 2 - 10,
        currentPos.y * cellSize + cellSize / 2 - 10,
        20,
        20
      );
      
      // Rover indicator
      ctx.fillStyle = '#55EFC4';
      ctx.beginPath();
      ctx.arc(
        currentPos.x * cellSize + cellSize / 2,
        currentPos.y * cellSize + cellSize / 2,
        4,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
  }, [currentStep, path]);

  useEffect(() => {
    if (!isRunning || currentStep >= path.length - 1) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (currentStep >= path.length - 1 && path.length > 0) {
        onTargetReached();
      }
      return;
    }

    const animate = (timestamp: number) => {
      if (timestamp - lastUpdateRef.current >= 500) {
        setCurrentStep(prev => prev + 1);
        lastUpdateRef.current = timestamp;
      }
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRunning, currentStep, path.length, onTargetReached]);

  return (
    <div className="w-full h-full flex items-center justify-center">
      <canvas
        ref={canvasRef}
        width={600}
        height={600}
        className="border border-[var(--color-card)] rounded-lg"
        role="img"
        aria-label="Rover simulation canvas showing rover movement on grid"
      />
    </div>
  );
}
