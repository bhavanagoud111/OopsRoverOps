import React from 'react';

interface SpaceshipSVGProps {
  size?: number;
  className?: string;
  variant?: 'default' | 'glow';
}

export function SpaceshipSVG({ size = 50, className = '', variant = 'default' }: SpaceshipSVGProps) {
  return (
    <svg
      width={size}
      height={size * 1.2}
      viewBox="0 0 50 60"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="Spaceship destination marker"
    >
      <g filter={variant === 'glow' ? 'url(#spaceshipGlowFilter)' : 'url(#spaceshipDefaultFilter)'}>
        {/* Nose cone */}
        <path
          d="M 25 5 L 18 20 L 32 20 Z"
          fill="url(#spaceshipGradient)"
          stroke="#A47CFF"
          strokeWidth="1.5"
        />
        
        {/* Main body */}
        <rect
          x="18"
          y="20"
          width="14"
          height="20"
          fill="url(#spaceshipGradient)"
          stroke="#A47CFF"
          strokeWidth="1.5"
        />
        
        {/* Windows */}
        <circle cx="25" cy="25" r="2.5" fill="#00CEC9" opacity="0.8" />
        <circle cx="25" cy="32" r="2" fill="#00CEC9" opacity="0.6" />
        
        {/* Fins */}
        <path
          d="M 18 35 L 10 45 L 18 40 Z"
          fill="url(#finGradient)"
          stroke="#6C5CE7"
          strokeWidth="1.5"
        />
        <path
          d="M 32 35 L 40 45 L 32 40 Z"
          fill="url(#finGradient)"
          stroke="#6C5CE7"
          strokeWidth="1.5"
        />
        
        {/* Engine exhaust */}
        <rect
          x="20"
          y="40"
          width="10"
          height="8"
          rx="2"
          fill="#FD79A8"
          opacity="0.7"
        />
        
        {/* Flame effect (only for glow variant) */}
        {variant === 'glow' && (
          <path
            d="M 22 48 Q 25 55 25 58 Q 25 55 28 48"
            fill="url(#flameGradient)"
            opacity="0.8"
          />
        )}
      </g>
      
      <defs>
        {/* Spaceship gradient */}
        <linearGradient id="spaceshipGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6C5CE7" />
          <stop offset="50%" stopColor="#8B7CFF" />
          <stop offset="100%" stopColor="#A47CFF" />
        </linearGradient>
        
        {/* Fin gradient */}
        <linearGradient id="finGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6C5CE7" />
          <stop offset="100%" stopColor="#4a3cc7" />
        </linearGradient>
        
        {/* Flame gradient */}
        <linearGradient id="flameGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FD79A8" />
          <stop offset="50%" stopColor="#FF9F6E" />
          <stop offset="100%" stopColor="#FFD93D" />
        </linearGradient>
        
        {/* Glow filter */}
        <filter id="spaceshipGlowFilter" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        
        <filter id="spaceshipDefaultFilter" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="1" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
    </svg>
  );
}
