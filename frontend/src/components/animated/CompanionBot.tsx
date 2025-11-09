import React from 'react';
import { motion } from 'motion/react';

interface CompanionBotProps {
  state?: 'idle' | 'talking' | 'excited';
  message?: string;
  size?: number;
}

export function CompanionBot({ state = 'idle', message, size = 80 }: CompanionBotProps) {
  const floatVariants = {
    idle: {
      y: [0, -5, 0],
      transition: {
        duration: 3,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    },
    talking: {
      y: [0, -3, 0],
      transition: {
        duration: 0.8,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    },
    excited: {
      y: [0, -8, 0],
      rotate: [0, 5, -5, 0],
      transition: {
        duration: 0.6,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    }
  };

  const eyeVariants = {
    idle: {
      scaleY: [1, 0.1, 1],
      transition: {
        duration: 3,
        repeat: Infinity,
        repeatDelay: 2,
        ease: 'easeInOut'
      }
    },
    talking: {
      scaleY: [1, 0.8, 1],
      transition: {
        duration: 0.5,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    },
    excited: {
      scaleY: 1,
      scale: [1, 1.2, 1],
      transition: {
        duration: 0.4,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    }
  };

  return (
    <div className="relative" style={{ width: size, height: size + 40 }}>
      <motion.div
        variants={floatVariants}
        animate={state}
        role="img"
        aria-label={`AI companion bot in ${state} state`}
      >
        <svg
          width={size}
          height={size}
          viewBox="0 0 80 80"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
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
          
          {/* Eyes container */}
          <g>
            {/* Left eye */}
            <ellipse cx="30" cy="35" rx="5" ry="6" fill="#00CEC9" />
            <motion.ellipse
              cx="30"
              cy="35"
              rx="5"
              ry="6"
              fill="#00CEC9"
              variants={eyeVariants}
              animate={state}
            />
            <circle cx="30" cy="34" r="2" fill="#ffffff" />
            
            {/* Right eye */}
            <ellipse cx="50" cy="35" rx="5" ry="6" fill="#00CEC9" />
            <motion.ellipse
              cx="50"
              cy="35"
              rx="5"
              ry="6"
              fill="#00CEC9"
              variants={eyeVariants}
              animate={state}
            />
            <circle cx="50" cy="34" r="2" fill="#ffffff" />
          </g>
          
          {/* Mouth - changes based on state */}
          {state === 'idle' && (
            <path d="M 28 50 Q 40 52 52 50" stroke="#6C5CE7" strokeWidth="2" strokeLinecap="round" fill="none" />
          )}
          {state === 'talking' && (
            <circle cx="40" cy="50" r="4" fill="#6C5CE7" />
          )}
          {state === 'excited' && (
            <path d="M 28 48 Q 40 56 52 48" stroke="#FD79A8" strokeWidth="2" strokeLinecap="round" fill="none" />
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
      </motion.div>
      
      {/* Speech bubble */}
      {message && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className="absolute top-0 left-full ml-2 whitespace-nowrap"
        >
          <div className="relative bg-[var(--color-card)] border border-[var(--color-primary)] rounded-lg px-3 py-2 shadow-lg">
            <p className="text-[var(--color-text-primary)]">
              <small>{message}</small>
            </p>
            {/* Speech bubble tail */}
            <div className="absolute left-0 top-1/2 -translate-x-1/2 -translate-y-1/2">
              <div className="w-0 h-0 border-t-[6px] border-t-transparent border-r-[8px] border-r-[var(--color-primary)] border-b-[6px] border-b-transparent" />
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
