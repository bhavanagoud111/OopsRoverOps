import React from 'react';

interface SpeechBubbleSVGProps {
  width?: number;
  height?: number;
  className?: string;
  tailDirection?: 'left' | 'right' | 'top' | 'bottom';
}

export function SpeechBubbleSVG({ 
  width = 120, 
  height = 60, 
  className = '',
  tailDirection = 'left'
}: SpeechBubbleSVGProps) {
  const getTailPath = () => {
    switch (tailDirection) {
      case 'left':
        return 'M 0 30 L -10 25 L 0 35 Z';
      case 'right':
        return `M ${width} 30 L ${width + 10} 25 L ${width} 35 Z`;
      case 'top':
        return `M ${width / 2} 0 L ${width / 2 - 5} -10 L ${width / 2 + 5} 0 Z`;
      case 'bottom':
        return `M ${width / 2} ${height} L ${width / 2 - 5} ${height + 10} L ${width / 2 + 5} ${height} Z`;
      default:
        return 'M 0 30 L -10 25 L 0 35 Z';
    }
  };

  return (
    <svg
      width={width + 20}
      height={height + 20}
      viewBox={`-10 -10 ${width + 20} ${height + 20}`}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="Speech bubble"
    >
      {/* Main bubble */}
      <rect
        x="0"
        y="0"
        width={width}
        height={height}
        rx="8"
        fill="#111120"
        stroke="#6C5CE7"
        strokeWidth="2"
      />
      
      {/* Tail */}
      <path
        d={getTailPath()}
        fill="#111120"
        stroke="#6C5CE7"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );
}
