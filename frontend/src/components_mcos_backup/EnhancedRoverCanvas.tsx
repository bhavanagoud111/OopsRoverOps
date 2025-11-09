import React, { useRef, useEffect, useState } from 'react';
import { RoverAnimated } from './animated/RoverAnimated';
import { CompanionBot } from './animated/CompanionBot';
import { DestinationMarker } from './animated/DestinationMarker';

interface Position {
  x: number;
  y: number;
}

interface EnhancedRoverCanvasProps {
  isRunning: boolean;
  onTargetReached: () => void;
  path: Position[];
}

export function EnhancedRoverCanvas({ isRunning, onTargetReached, path }: EnhancedRoverCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [companionMessage, setCompanionMessage] = useState('');
  const hasReachedTargetRef = useRef(false);
  const animationFrameRef = useRef<number>();
  const lastUpdateRef = useRef<number>(0);

  const messages = [
    "Calculating trajectory…",
    "Oops… I'm navigating!",
    "Avoiding craters…",
    "Trajectory locked!",
    "Systems optimal!",
    "Almost there!"
  ];

  // Reset currentStep when path changes (new mission starts)
  useEffect(() => {
    setCurrentStep(0);
    setCompanionMessage('');
    hasReachedTargetRef.current = false;
  }, [path]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const gridSize = 20;
    const cellSize = canvas.width / gridSize;

    // Clear canvas with dark background
    ctx.fillStyle = '#0A0A0F';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid with subtle purple glow
    ctx.strokeStyle = 'rgba(108, 92, 231, 0.15)';
    ctx.lineWidth = 1;
    ctx.shadowColor = 'rgba(108, 92, 231, 0.3)';
    ctx.shadowBlur = 2;
    
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

    ctx.shadowBlur = 0;

    // Draw path with glowing effect
    if (path.length > 0 && currentStep > 0) {
      ctx.strokeStyle = '#6C5CE7';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#6C5CE7';
      ctx.shadowBlur = 10;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
      ctx.beginPath();
      ctx.moveTo(path[0].x * cellSize + cellSize / 2, path[0].y * cellSize + cellSize / 2);
      
      for (let i = 1; i <= Math.min(currentStep, path.length - 1); i++) {
        ctx.lineTo(path[i].x * cellSize + cellSize / 2, path[i].y * cellSize + cellSize / 2);
      }
      ctx.stroke();
      
      // Add waypoint dots
      for (let i = 0; i <= Math.min(currentStep, path.length - 1); i++) {
        ctx.fillStyle = '#6C5CE7';
        ctx.shadowBlur = 5;
        ctx.beginPath();
        ctx.arc(
          path[i].x * cellSize + cellSize / 2,
          path[i].y * cellSize + cellSize / 2,
          3,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
    }

    ctx.shadowBlur = 0;

    // Draw start position with glow
    if (path.length > 0) {
      ctx.fillStyle = '#00CEC9';
      ctx.shadowColor = '#00CEC9';
      ctx.shadowBlur = 15;
      ctx.beginPath();
      ctx.arc(
        path[0].x * cellSize + cellSize / 2,
        path[0].y * cellSize + cellSize / 2,
        10,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }

    ctx.shadowBlur = 0;
  }, [currentStep, path]);

  useEffect(() => {
    if (!isRunning || currentStep >= path.length - 1) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      // Only call onTargetReached once
      if (currentStep >= path.length - 1 && path.length > 0 && !hasReachedTargetRef.current) {
        hasReachedTargetRef.current = true;
        setCompanionMessage("Target reached!");
        setTimeout(() => onTargetReached(), 1000);
      }
      return;
    }

    // Update companion message periodically
    const messageInterval = setInterval(() => {
      const randomMessage = messages[Math.floor(Math.random() * messages.length)];
      setCompanionMessage(randomMessage);
    }, 3000);

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
      clearInterval(messageInterval);
    };
  }, [isRunning, currentStep, path.length]);

  // Calculate pixel positions for overlay components
  const getPixelPosition = (gridPos: Position) => {
    const cellSize = 600 / 20; // canvas width / grid size
    return {
      x: gridPos.x * cellSize + cellSize / 2,
      y: gridPos.y * cellSize + cellSize / 2
    };
  };

  const roverState = currentStep >= path.length - 1 ? 'reachedTarget' : isRunning ? 'moving' : 'idle';
  const companionState = isRunning ? 'talking' : currentStep >= path.length - 1 ? 'excited' : 'idle';
  const destinationVariant = currentStep >= path.length - 1 ? 'glow' : 'default';

  return (
    <div 
      ref={containerRef}
      className="w-full h-full flex items-center justify-center p-8"
    >
      <div className="relative" style={{ width: 600, height: 600 }}>
        {/* Canvas with vignette effect */}
        <div className="relative rounded-xl overflow-hidden" style={{
          boxShadow: '0 0 60px rgba(108, 92, 231, 0.2), inset 0 0 100px rgba(0, 0, 0, 0.5)'
        }}>
          <canvas
            id="roverCanvas"
            ref={canvasRef}
            width={600}
            height={600}
            className="block"
            role="img"
            aria-label="Interactive rover simulation canvas showing rover movement on grid"
          />
          
          {/* Vignette overlay */}
          <div 
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'radial-gradient(circle at center, transparent 40%, rgba(10, 10, 15, 0.7) 100%)'
            }}
            aria-hidden="true"
          />
        </div>

        {/* Rover overlay */}
        {path.length > 0 && currentStep < path.length && (
          <div
            id="roverLayer"
            className="absolute pointer-events-none"
            style={{
              left: getPixelPosition(path[currentStep]).x - 30,
              top: getPixelPosition(path[currentStep]).y - 30,
              transform: 'translate(-50%, -50%)'
            }}
          >
            <RoverAnimated state={roverState} size={60} />
          </div>
        )}

        {/* Destination marker overlay */}
        {path.length > 0 && (
          <div
            id="targetLayer"
            className="absolute pointer-events-none"
            style={{
              left: getPixelPosition(path[path.length - 1]).x - 25,
              top: getPixelPosition(path[path.length - 1]).y - 60,
              transform: 'translate(-50%, -50%)'
            }}
          >
            <DestinationMarker variant={destinationVariant} size={50} />
          </div>
        )}

        {/* Companion bot overlay */}
        <div
          id="companionLayer"
          className="absolute pointer-events-none"
          style={{
            right: -100,
            top: 20
          }}
        >
          <CompanionBot 
            state={companionState} 
            message={isRunning ? companionMessage : currentStep >= path.length - 1 ? "Mission complete!" : ""}
            size={80}
          />
        </div>
      </div>
    </div>
  );
}