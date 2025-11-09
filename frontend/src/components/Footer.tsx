import React from 'react';

interface FooterProps {
  onNavigate?: (page: string) => void;
}

export function Footer({ onNavigate }: FooterProps) {
  return (
    <footer className="border-t border-[var(--color-card)] mt-auto">
      <div className="max-w-[1440px] mx-auto px-6 py-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-[var(--color-text-secondary)]">
            <small>© 2025 OopsRoverOps. Built with NASA APIs.</small>
          </p>
          <nav aria-label="Footer navigation">
            <ul className="flex items-center gap-4" role="list">
              <li>
                <button
                  onClick={() => onNavigate?.('about')}
                  className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] rounded px-2 py-1"
                >
                  <small>About</small>
                </button>
              </li>
              <li>
                <a
                  href="https://api.nasa.gov"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] transition-colors focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] rounded px-2 py-1"
                >
                  <small>NASA API</small>
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </div>
    </footer>
  );
}