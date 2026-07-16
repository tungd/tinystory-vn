import { MdInfoOutline } from 'react-icons/md';
import { EvalRadar } from './EvalRadar';
import type { EvalAxes, EvalResult } from '../api';

export type EvalState = 'idle' | 'loading' | 'done' | 'error';

/** (i) icon với tooltip native giải thích một chỉ số. */
function InfoIcon({ text }: { text: string }) {
  return (
    <MdInfoOutline
      size={13}
      title={text}
      aria-label={text}
      tabIndex={0}
      style={{
        marginLeft: '0.25rem',
        color: 'var(--astryx-color-text-subtle, #9ca3af)',
        cursor: 'help',
        flexShrink: 0,
        verticalAlign: 'text-top',
      }}
    />
  );
}

export interface EvalPanelProps {
  state: EvalState;
  scores: EvalResult | null;
  errorMsg?: string;
}

const AXIS_LABELS: Record<keyof EvalAxes, string> = {
  grammar: 'Grammar',
  creativity: 'Creativity',
  moral_clarity: 'Moral Clarity',
  prompt_adherence: 'Prompt Adherence',
};

export function EvalPanel({ state, scores, errorMsg }: EvalPanelProps) {
  if (state === 'idle') {
    return (
      <p
        style={{
          margin: '0.75rem 0 0',
          fontSize: '0.8125rem',
          color: 'var(--astryx-color-text-subtle, #9ca3af)',
          fontStyle: 'italic',
        }}
      >
        Evaluation will run after generation
      </p>
    );
  }

  if (state === 'loading') {
    return (
      <div
        role="status"
        aria-live="polite"
        aria-label="Evaluating story"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          margin: '0.75rem 0 0',
          padding: '0.75rem 1rem',
          borderRadius: '6px',
          background: 'var(--astryx-color-surface-raised, #f9fafb)',
          border: '1px solid var(--astryx-color-border, #e5e7eb)',
          fontSize: '0.875rem',
          color: 'var(--astryx-color-text-subtle, #6b7280)',
        }}
      >
        <span
          style={{
            display: 'inline-block',
            width: '1rem',
            height: '1rem',
            borderRadius: '50%',
            border: '2px solid var(--astryx-color-border, #e5e7eb)',
            borderTopColor: '#2563eb',
            animation: 'spin 0.75s linear infinite',
          }}
        />
        Evaluating…
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (state === 'error') {
    return (
      <div
        role="alert"
        aria-label="Evaluation error"
        style={{
          margin: '0.75rem 0 0',
          padding: '0.75rem 1rem',
          borderRadius: '6px',
          background: '#fef2f2',
          border: '1px solid #fecaca',
          fontSize: '0.875rem',
          color: '#dc2626',
        }}
      >
        Evaluation error: {errorMsg ?? 'Unknown error'}
      </div>
    );
  }

  if (state === 'done' && scores) {
    const axes = Object.keys(AXIS_LABELS) as (keyof typeof AXIS_LABELS)[];
    const radarScores = {
      grammar: scores.grammar,
      creativity: scores.creativity,
      moral_clarity: scores.moral_clarity,
      prompt_adherence: scores.prompt_adherence,
    };

    return (
      <div
        style={{
          margin: '0.75rem 0 0',
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
          Quick Evaluation
        </p>

        <EvalRadar series={[{ name: 'Scores', scores: radarScores }]} />

        <table
          role="table"
          aria-label="Evaluation score breakdown"
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.8125rem',
            marginTop: '0.75rem',
          }}
        >
          <thead>
            <tr>
              <th
                style={{
                  textAlign: 'left',
                  padding: '0.375rem 0.5rem',
                  borderBottom: '2px solid var(--astryx-color-border, #e5e7eb)',
                  fontWeight: 600,
                  color: 'var(--astryx-color-text-subtle, #6b7280)',
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                }}
              >
                Axis
              </th>
              <th
                style={{
                  textAlign: 'right',
                  padding: '0.375rem 0.5rem',
                  borderBottom: '2px solid var(--astryx-color-border, #e5e7eb)',
                  fontWeight: 600,
                  color: 'var(--astryx-color-text-subtle, #6b7280)',
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                }}
              >
                Score / 10
              </th>
            </tr>
          </thead>
          <tbody>
            {axes.map((axis) => {
              const reason = scores.rationale?.[axis];
              return (
                <tr key={axis}>
                  <td
                    style={{
                      padding: '0.375rem 0.5rem',
                      borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                      color: 'var(--astryx-color-text, #111827)',
                      verticalAlign: 'top',
                    }}
                  >
                    <div>
                      {AXIS_LABELS[axis]}
                      {scores.method?.axes?.[axis] && (
                        <InfoIcon text={scores.method.axes[axis]} />
                      )}
                    </div>
                    {reason && (
                      <div
                        style={{
                          marginTop: '0.2rem',
                          fontSize: '0.75rem',
                          lineHeight: 1.4,
                          color: 'var(--astryx-color-text-subtle, #6b7280)',
                          fontWeight: 400,
                        }}
                      >
                        {reason}
                      </div>
                    )}
                  </td>
                  <td
                    style={{
                      textAlign: 'right',
                      padding: '0.375rem 0.5rem',
                      borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                      fontVariantNumeric: 'tabular-nums',
                      fontWeight: 500,
                      color: '#2563eb',
                      verticalAlign: 'top',
                    }}
                  >
                    {scores[axis].toFixed(1)}
                  </td>
                </tr>
              );
            })}
            <tr>
              <td style={{ padding: '0.375rem 0.5rem', fontWeight: 700 }}>
                Overall
                {scores.method?.overall_formula && (
                  <InfoIcon text={scores.method.overall_formula} />
                )}
              </td>
              <td
                style={{
                  textAlign: 'right',
                  padding: '0.375rem 0.5rem',
                  fontVariantNumeric: 'tabular-nums',
                  fontWeight: 700,
                  color: 'var(--astryx-color-text, #111827)',
                }}
              >
                {scores.overall.toFixed(2)}
              </td>
            </tr>
          </tbody>
        </table>

        {/* Objective metrics (reference-free, deterministic) - phần tự động của paper */}
        {scores.objective && (
          <div style={{ marginTop: '1rem' }}>
            <p
              style={{
                margin: '0 0 0.4rem',
                fontSize: '0.75rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: 'var(--astryx-color-text-subtle, #6b7280)',
              }}
            >
              Objective metrics (auto, reference-free)
            </p>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
              <tbody>
                {(
                  [
                    ['distinct_1', 'Distinct-1', scores.objective.distinct_1.toFixed(3)],
                    ['distinct_2', 'Distinct-2', scores.objective.distinct_2.toFixed(3)],
                    ['flesch_reading_ease', 'Flesch Reading Ease', scores.objective.flesch_reading_ease.toFixed(1)],
                  ] as const
                ).map(([key, label, val]) => (
                  <tr key={key}>
                    <td
                      style={{
                        padding: '0.3rem 0.5rem',
                        borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                        color: 'var(--astryx-color-text, #111827)',
                      }}
                    >
                      {label}
                      {scores.method?.objective_defs?.[key] && (
                        <InfoIcon text={scores.method.objective_defs[key]} />
                      )}
                    </td>
                    <td
                      style={{
                        textAlign: 'right',
                        padding: '0.3rem 0.5rem',
                        borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                        fontVariantNumeric: 'tabular-nums',
                        fontWeight: 500,
                        color: '#0891b2',
                      }}
                    >
                      {val}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Method + citation (thay footnote 'See Results tab' cũ) */}
        <p
          style={{
            margin: '0.85rem 0 0',
            fontSize: '0.75rem',
            lineHeight: 1.5,
            color: 'var(--astryx-color-text-subtle, #9ca3af)',
            fontStyle: 'italic',
          }}
        >
          {scores.method ? (
            <>
              Judge: <b>{scores.method.judge_model}</b> ({scores.method.scale}), 1 judge = chỉ báo nhanh.
              {' '}{scores.method.citation}{' '}{scores.method.note}
            </>
          ) : (
            'Scores from a single LLM judge (quick indicator).'
          )}
        </p>
      </div>
    );
  }

  return null;
}
