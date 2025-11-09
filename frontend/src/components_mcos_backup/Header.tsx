import React from 'react';
import { Rocket } from 'lucide-react';

interface HeaderProps {
  onNavigate?: (page: string) => void;
  currentPage?: string;
}

export function Header({ onNavigate, currentPage }: HeaderProps) {
  return (
    <header className="border-b border-[var(--color-card)]">
      <div className="max-w-[1440px] mx-auto px-6 py-4">
        <nav className="flex items-center justify-between" aria-label="Main navigation">
          <div className="flex items-center gap-4">
            <button
              onClick={() => onNavigate?.('landing')}
              className="flex items-center gap-2 text-[var(--color-text-primary)] hover:text-[var(--color-primary)] transition-colors focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] rounded"
              aria-label="Go to home page"
            >
              <Rocket size={28} aria-hidden="true" />
              <span>OopsRoverOps</span>
            </button>
          </div>
          
          <ul className="flex items-center gap-6" role="list">
            <li>
              <button
                onClick={() => onNavigate?.('dashboard')}
                className={`hover:text-[var(--color-primary)] transition-colors focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] rounded px-2 py-1 ${
                  currentPage === 'dashboard' ? 'text-[var(--color-primary)]' : 'text-[var(--color-text-secondary)]'
                }`}
                aria-current={currentPage === 'dashboard' ? 'page' : undefined}
              >
                Dashboard
              </button>
            </li>
            <li>
              <button
                onClick={() => onNavigate?.('showcase')}
                className={`hover:text-[var(--color-primary)] transition-colors focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] rounded px-2 py-1 ${
                  currentPage === 'showcase' ? 'text-[var(--color-primary)]' : 'text-[var(--color-text-secondary)]'
                }`}
                aria-current={currentPage === 'showcase' ? 'page' : undefined}
              >
                Components
              </button>
            </li>
            <li>
              <button
                onClick={() => onNavigate?.('about')}
                className={`hover:text-[var(--color-primary)] transition-colors focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] rounded px-2 py-1 ${
                  currentPage === 'about' ? 'text-[var(--color-primary)]' : 'text-[var(--color-text-secondary)]'
                }`}
                aria-current={currentPage === 'about' ? 'page' : undefined}
              >
                About
              </button>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}