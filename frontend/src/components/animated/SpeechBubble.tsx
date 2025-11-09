import React from 'react';
import { motion } from 'motion/react';

interface SpeechBubbleProps {
  message: string;
  variant?: 'left' | 'right' | 'top' | 'bottom';
  visible?: boolean;
}

export function SpeechBubble({ message, variant = 'right', visible = true }: SpeechBubbleProps) {
  if (!visible) return null;

  const getTailPosition = () => {
    switch (variant) {
      case 'left':
        return 'left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 border-t-[6px] border-t-transparent border-r-[8px] border-r-[var(--color-primary)] border-b-[6px] border-b-transparent';
      case 'right':
        return 'right-0 top-1/2 translate-x-1/2 -translate-y-1/2 border-t-[6px] border-t-transparent border-l-[8px] border-l-[var(--color-primary)] border-b-[6px] border-b-transparent';
      case 'top':
        return 'top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 border-l-[6px] border-l-transparent border-b-[8px] border-b-[var(--color-primary)] border-r-[6px] border-r-transparent';
      case 'bottom':
        return 'bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 border-l-[6px] border-l-transparent border-t-[8px] border-t-[var(--color-primary)] border-r-[6px] border-r-transparent';
      default:
        return 'right-0 top-1/2 translate-x-1/2 -translate-y-1/2 border-t-[6px] border-t-transparent border-l-[8px] border-l-[var(--color-primary)] border-b-[6px] border-b-transparent';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ duration: 0.2 }}
      className="relative"
    >
      <div className="bg-[var(--color-card)] border border-[var(--color-primary)] rounded-lg px-4 py-2 shadow-lg">
        <p className="text-[var(--color-text-primary)] whitespace-nowrap">
          <small>{message}</small>
        </p>
        {/* Speech bubble tail */}
        <div className={`absolute ${getTailPosition()}`} />
      </div>
    </motion.div>
  );
}
