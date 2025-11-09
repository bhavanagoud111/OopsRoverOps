import React from 'react';

interface RoverSVGProps {
  size?: number;
  className?: string;
}

export function RoverSVG({ size = 60, className = '' }: RoverSVGProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 60 60"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="Rover icon"
    >
      {/* Main rover body - rounded square */}
      <rect
        x="10"
        y="15"
        width="40"
        height="30"
        rx="6"
        fill="url(#roverGradient)"
        stroke="#00CEC9"
        strokeWidth="2"
      />
      
      {/* Wheels */}
      <rect x="8" y="42" width="8" height="6" rx="2" fill="#555" />
      <rect x="22" y="42" width="8" height="6" rx="2" fill="#555" />
      <rect x="30" y="42" width="8" height="6" rx="2" fill="#555" />
      <rect x="44" y="42" width="8" height="6" rx="2" fill="#555" />
      
      {/* Antenna */}
      <line x1="30" y1="15" x2="30" y2="8" stroke="#00CEC9" strokeWidth="2" />
      
      {/* Camera/sensor panels */}
      <rect x="15" y="22" width="8" height="6" rx="1" fill="#333" />
      <rect x="37" y="22" width="8" height="6" rx="1" fill="#333" />
      
      {/* Center panel */}
      <rect x="25" y="30" width="10" height="8" rx="2" fill="#222" />
      
      {/* LED indicator on top */}
      <circle
        cx="30"
        cy="6"
        r="3"
        fill="#FD79A8"
        filter="url(#ledGlow)"
      />
      
      {/* Gradients and filters */}
      <defs>
        <linearGradient id="roverGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#2a2a3e" />
          <stop offset="100%" stopColor="#1a1a2e" />
        </linearGradient>
        
        <filter id="ledGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
    </svg>
  );
}
