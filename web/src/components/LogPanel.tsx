import type { IconType } from 'react-icons';
import { MdHourglassEmpty, MdCheckCircle, MdBlock } from 'react-icons/md';

export interface LogEntry {
  id: string;
  stage: 'input_check' | 'generating' | 'output_check';
  status: 'running' | 'ok' | 'blocked';
  detail?: string;
  timestamp: Date;
}

export interface LogPanelProps {
  entries: LogEntry[];
}

const STAGE_LABELS: Record<LogEntry['stage'], string> = {
  input_check: 'Input check (Layer 1)',
  generating: 'Generating (Layer 2-3)',
  output_check: 'Output check (Layer 4)',
};

const STATUS_ICONS: Record<LogEntry['status'], IconType> = {
  running: MdHourglassEmpty,
  ok: MdCheckCircle,
  blocked: MdBlock,
};

const STATUS_COLORS: Record<LogEntry['status'], string> = {
  running: 'var(--astryx-color-text-subtle, #6b7280)',
  ok: 'var(--astryx-color-success, #16a34a)',
  blocked: 'var(--astryx-color-danger, #dc2626)',
};

function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

export function LogPanel({ entries }: LogPanelProps) {
  return (
    <div
      style={{
        border: '1px solid var(--astryx-color-border, #e5e7eb)',
        borderRadius: '8px',
        padding: '1rem',
        background: 'var(--astryx-color-surface, #fff)',
      }}
    >
      <p
        style={{
          margin: '0 0 0.75rem',
          fontSize: '0.75rem',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: 'var(--astryx-color-text-subtle, #6b7280)',
        }}
      >
        Activity Log
      </p>

      {entries.length === 0 ? (
        <div
          style={{
            fontSize: '0.8125rem',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
            fontStyle: 'italic',
            padding: '0.5rem 0',
          }}
        >
          No activity yet
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {entries.map((entry) => {
            const StatusIcon = STATUS_ICONS[entry.status];
            return (
              <div
                key={entry.id}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.125rem',
                  padding: '0.5rem 0.625rem',
                  borderRadius: '6px',
                  background: 'var(--astryx-color-surface-raised, #f9fafb)',
                  borderLeft: `3px solid ${STATUS_COLORS[entry.status]}`,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '0.5rem',
                  }}
                >
                  <span
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.375rem',
                      fontSize: '0.8125rem',
                      fontWeight: 500,
                      color: STATUS_COLORS[entry.status],
                    }}
                  >
                    <StatusIcon
                      aria-label={entry.status}
                      size={14}
                      color={STATUS_COLORS[entry.status]}
                    />
                    {STAGE_LABELS[entry.stage]}
                  </span>
                  <span
                    style={{
                      fontSize: '0.6875rem',
                      color: 'var(--astryx-color-text-subtle, #9ca3af)',
                      fontVariantNumeric: 'tabular-nums',
                      flexShrink: 0,
                    }}
                  >
                    {formatTime(entry.timestamp)}
                  </span>
                </div>
                {entry.detail && (
                  <span
                    style={{
                      fontSize: '0.75rem',
                      color: 'var(--astryx-color-text-subtle, #6b7280)',
                      paddingLeft: '1.375rem',
                    }}
                  >
                    {entry.detail}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
