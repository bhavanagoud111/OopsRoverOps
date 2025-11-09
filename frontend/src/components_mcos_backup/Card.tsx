import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export function Card({ children, className = '' }: CardProps) {
  return (
    <div 
      className={`bg-[var(--color-card)] rounded-lg p-6 shadow-[var(--shadow-subtle)] ${className}`}
    >
      {children}
    </div>
  );
}
