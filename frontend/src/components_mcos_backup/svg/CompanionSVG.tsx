import React from 'react';

interface CompanionSVGProps {
  size?: number;
  className?: string;
  eyeState?: 'open' | 'blink' | 'happy';
}

export function CompanionSVG({ size = 80, className = '', eyeState = 'open' }: CompanionSVGProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="AI companion robot"
    >
      {/* Bot head - round */}
      <circle
        cx="40"
        cy="40"
        r="30"
        fill="url(#botGradient)"
        stroke="#6C5CE7"
        strokeWidth="2"
      />
      
      {/* Antenna */}
      <line x1="40" y1="10" x2="40" y2="4" stroke="#6C5CE7" strokeWidth="2" strokeLinecap="round" />
      <circle cx="40" cy="2" r="2" fill="#FD79A8" filter="url(#antennaGlow)" />
      
      {/* Eyes */}
      <g>
        {/* Left eye */}
        {eyeState === 'blink' ? (
          <line x1="25" y1="35" x2="35" y2="35" stroke="#00CEC9" strokeWidth="2" strokeLinecap="round" />
        ) : (
          <>
            <ellipse cx="30" cy="35" rx="5" ry="6" fill="#00CEC9" />
            <circle cx="30" cy="34" r="2" fill="#ffffff" />
          </>
        )}
        
        {/* Right eye */}
        {eyeState === 'blink' ? (
          <line x1="45" y1="35" x2="55" y2="35" stroke="#00CEC9" strokeWidth="2" strokeLinecap="round" />
        ) : (
          <>
            <ellipse cx="50" cy="35" rx="5" ry="6" fill="#00CEC9" />
            <circle cx="50" cy="34" r="2" fill="#ffffff" />
          </>
        )}
      </g>
      
      {/* Mouth - changes based on state */}
      {eyeState === 'happy' ? (
        <path d="M 28 48 Q 40 56 52 48" stroke="#FD79A8" strokeWidth="2" strokeLinecap="round" fill="none" />
      ) : (
        <path d="M 28 50 Q 40 52 52 50" stroke="#6C5CE7" strokeWidth="2" strokeLinecap="round" fill="none" />
      )}
      
      {/* Cheek blush marks */}
      <circle cx="20" cy="42" r="3" fill="#FD79A8" opacity="0.4" />
      <circle cx="60" cy="42" r="3" fill="#FD79A8" opacity="0.4" />
      
      <defs>
        <linearGradient id="botGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#2a2a4e" />
          <stop offset="100%" stopColor="#1a1a3e" />
        </linearGradient>
        
        <filter id="antennaGlow" x="-100%" y="-100%" width="300%" height="300%">
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
