import React from 'react';
import { Download, FileText, Calendar, Clock } from 'lucide-react';
import { MCOSButton } from '../MCOSButton';

interface Report {
  id: string;
  title: string;
  date: string;
  duration: string;
  status: 'complete' | 'partial' | 'failed';
}

const mockReports: Report[] = [
  {
    id: '1',
    title: 'Mars Crater Exploration - Nov 8, 2025',
    date: '2025-11-08',
    duration: '2h 34m',
    status: 'complete',
  },
  {
    id: '2',
    title: 'Soil Sample Collection - Nov 7, 2025',
    date: '2025-11-07',
    duration: '1h 45m',
    status: 'complete',
  },
  {
    id: '3',
    title: 'North Ridge Survey - Nov 6, 2025',
    date: '2025-11-06',
    duration: '3h 12m',
    status: 'partial',
  },
];

const statusColors = {
  complete: { bg: 'rgba(29, 185, 84, 0.15)', text: '#1DB954', border: 'rgba(29, 185, 84, 0.4)' },
  partial: { bg: 'rgba(255, 176, 32, 0.15)', text: '#FFB020', border: 'rgba(255, 176, 32, 0.4)' },
  failed: { bg: 'rgba(255, 77, 77, 0.15)', text: '#FF4D4D', border: 'rgba(255, 77, 77, 0.4)' },
};

export function Reports() {
  const handleDownload = (reportId: string) => {
    console.log('Downloading report:', reportId);
    // Trigger download action
  };

  return (
    <div
      style={{
        height: '100%',
        padding: 'var(--mcos-space-4)',
        overflow: 'auto',
      }}
    >
      <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
        <div style={{ marginBottom: 'var(--mcos-space-4)' }}>
          <h2 style={{ marginBottom: '8px', color: 'var(--mcos-text)' }}>
            Mission Reports
          </h2>
          <p style={{ color: 'var(--mcos-muted)', fontSize: '14px' }}>
            View and download detailed reports from past missions
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--mcos-space-3)' }}>
          {mockReports.map((report) => {
            const statusColor = statusColors[report.status];

            return (
              <div
                key={report.id}
                style={{
                  background: 'var(--mcos-surface)',
                  border: '1px solid var(--mcos-surface-2)',
                  borderRadius: 'var(--mcos-radius-lg)',
                  padding: 'var(--mcos-space-3)',
                  boxShadow: 'var(--mcos-shadow-md)',
                  display: 'grid',
                  gridTemplateColumns: 'auto 1fr auto',
                  gap: 'var(--mcos-space-3)',
                  alignItems: 'center',
                  transition: 'all var(--transition-base)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--mcos-primary)';
                  e.currentTarget.style.boxShadow = 'var(--mcos-shadow-glow)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--mcos-surface-2)';
                  e.currentTarget.style.boxShadow = 'var(--mcos-shadow-md)';
                }}
              >
                {/* Icon */}
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    background: 'rgba(79, 127, 255, 0.15)',
                    border: '1px solid rgba(79, 127, 255, 0.3)',
                    borderRadius: 'var(--mcos-radius-md)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <FileText size={24} color="var(--mcos-primary)" aria-hidden="true" />
                </div>

                {/* Content */}
                <div>
                  <h4 style={{ marginBottom: '8px', color: 'var(--mcos-text)' }}>
                    {report.title}
                  </h4>
                  <div style={{ display: 'flex', gap: 'var(--mcos-space-3)', flexWrap: 'wrap' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Calendar size={14} color="var(--mcos-muted)" aria-hidden="true" />
                      <span style={{ fontSize: '12px', color: 'var(--mcos-muted)' }}>
                        {report.date}
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Clock size={14} color="var(--mcos-muted)" aria-hidden="true" />
                      <span style={{ fontSize: '12px', color: 'var(--mcos-muted)' }}>
                        {report.duration}
                      </span>
                    </div>
                    <div
                      style={{
                        padding: '2px 8px',
                        background: statusColor.bg,
                        border: `1px solid ${statusColor.border}`,
                        borderRadius: 'var(--mcos-radius-sm)',
                        fontSize: '11px',
                        fontWeight: 600,
                        color: statusColor.text,
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                      }}
                    >
                      {report.status}
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <MCOSButton
                  variant="secondary"
                  size="sm"
                  icon={Download}
                  onClick={() => handleDownload(report.id)}
                  ariaLabel={`Download ${report.title}`}
                >
                  Download
                </MCOSButton>
              </div>
            );
          })}
        </div>

        {/* Empty State */}
        {mockReports.length === 0 && (
          <div
            style={{
              background: 'var(--mcos-surface)',
              border: '1px solid var(--mcos-surface-2)',
              borderRadius: 'var(--mcos-radius-lg)',
              padding: 'var(--mcos-space-6)',
              textAlign: 'center',
            }}
          >
            <FileText size={48} color="var(--mcos-muted)" style={{ marginBottom: '16px' }} />
            <h3 style={{ marginBottom: '8px', color: 'var(--mcos-text)' }}>
              No Reports Yet
            </h3>
            <p style={{ color: 'var(--mcos-muted)', fontSize: '14px' }}>
              Complete a mission to generate your first report
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
