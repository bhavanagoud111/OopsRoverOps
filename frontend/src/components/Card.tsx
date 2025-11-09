import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export function Card({ children, className = '', style }: CardProps) {
  return (
    <div 
      className={`bg-[var(--color-card)] rounded-lg p-6 shadow-[var(--shadow-subtle)] ${className}`}
      style={style}
    >
      {children}
    </div>
  );
}
