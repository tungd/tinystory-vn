import { useState } from 'react';

export interface FableMeta {
  model_id: string;
  model_name: string;
  kind?: string;
  temperature: number;
  top_p: number;
  repetition_penalty: number;
  num_predict: number;
  seed?: number;
  prompt_sent: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  tokens_per_sec: number;
}

export interface ObservabilityPanelProps {
  meta: FableMeta | null;
}

function MetaRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        gap: '0.5rem',
        fontSize: '0.8125rem',
        padding: '0.25rem 0',
        borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
      }}
    >
      <span style={{ color: 'var(--astryx-color-text-subtle, #6b7280)', flexShrink: 0 }}>
        {label}
      </span>
      <span
        style={{
          fontWeight: 500,
          color: 'var(--astryx-color-text, #111827)',
          textAlign: 'right',
          fontVariantNumeric: 'tabular-nums',
          wordBreak: 'break-all',
        }}
      >
        {value}
      </span>
    </div>
  );
}

export function ObservabilityPanel({ meta }: ObservabilityPanelProps) {
  const [promptOpen, setPromptOpen] = useState(false);

  const containerStyle: React.CSSProperties = {
    border: '1px solid var(--astryx-color-border, #e5e7eb)',
    borderRadius: '8px',
    padding: '1rem',
    background: 'var(--astryx-color-surface, #fff)',
  };

  const headingStyle: React.CSSProperties = {
    margin: '0 0 0.75rem',
    fontSize: '0.75rem',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'var(--astryx-color-text-subtle, #6b7280)',
  };

  if (!meta) {
    return (
      <div style={containerStyle}>
        <p style={headingStyle}>Observability</p>
        <div
          style={{
            fontSize: '0.8125rem',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
            fontStyle: 'italic',
            padding: '0.5rem 0',
          }}
        >
          Run a generation to see metrics
        </div>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <p style={headingStyle}>Observability</p>

      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {/* Model info */}
        <MetaRow
          label="Model"
          value={meta.kind ? `${meta.model_name} (${meta.kind})` : meta.model_name}
        />

        {/* Params section */}
        <div
          style={{
            margin: '0.5rem 0 0.25rem',
            fontSize: '0.6875rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--astryx-color-text-subtle, #9ca3af)',
          }}
        >
          Parameters
        </div>
        <MetaRow label="Temperature" value={meta.temperature} />
        <MetaRow label="Top P" value={meta.top_p} />
        <MetaRow label="Repetition penalty" value={meta.repetition_penalty} />
        <MetaRow label="Max tokens" value={meta.num_predict} />
        {meta.seed !== undefined && meta.seed !== null && (
          <MetaRow label="Seed" value={meta.seed} />
        )}

        {/* Usage section */}
        <div
          style={{
            margin: '0.5rem 0 0.25rem',
            fontSize: '0.6875rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--astryx-color-text-subtle, #9ca3af)',
          }}
        >
          Usage
        </div>
        <MetaRow label="Input tokens" value={meta.input_tokens} />
        <MetaRow label="Output tokens" value={meta.output_tokens} />
        <MetaRow label="Latency" value={`${meta.latency_ms.toFixed(0)} ms`} />
        <MetaRow label="Tokens / sec" value={meta.tokens_per_sec.toFixed(1)} />
      </div>

      {/* Collapsible prompt */}
      <div style={{ marginTop: '0.75rem' }}>
        <button
          onClick={() => setPromptOpen((o) => !o)}
          aria-expanded={promptOpen}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.375rem',
            width: '100%',
            padding: '0.375rem 0',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '0.8125rem',
            fontWeight: 500,
            color: 'var(--astryx-color-text, #111827)',
            textAlign: 'left',
          }}
        >
          <span
            style={{
              display: 'inline-block',
              transition: 'transform 0.15s',
              transform: promptOpen ? 'rotate(90deg)' : 'rotate(0deg)',
              fontSize: '0.625rem',
            }}
          >
            ▶
          </span>
          Prompt sent
        </button>

        {promptOpen && (
          <pre
            style={{
              margin: '0.375rem 0 0',
              padding: '0.75rem',
              borderRadius: '6px',
              background: 'var(--astryx-color-surface-raised, #f9fafb)',
              border: '1px solid var(--astryx-color-border, #e5e7eb)',
              fontSize: '0.75rem',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              color: 'var(--astryx-color-text, #111827)',
              maxHeight: '200px',
              overflowY: 'auto',
            }}
          >
            {meta.prompt_sent}
          </pre>
        )}
      </div>
    </div>
  );
}
