import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { MdBarChart } from 'react-icons/md';
import { fetchResults } from '../api';
import { EvalRadar } from './EvalRadar';

// Internal schema - typed defensively
interface ObjectiveMetrics {
  perplexity?: number;
  distinct_1?: number;
  distinct_2?: number;
  self_bleu?: number;
  flesch_reading_ease?: number;
}

interface JudgeScores {
  grammar?: number;
  creativity?: number;
  moral_clarity?: number;
  prompt_adherence?: number;
  overall?: number;
}

interface EvalSummary {
  objective?: { base?: ObjectiveMetrics; finetuned?: ObjectiveMetrics };
  judge_panel?: { judges?: string[]; base?: JudgeScores; finetuned?: JudgeScores };
  agreement?: { cohen_kappa?: number; kendall_tau?: number };
  conclusion?: { winner?: string; by_rank?: string; notes?: string };
  loss_curve?: { step: number; loss: number }[];
}

// Type-guard for EvalSummary shape
function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function toEvalSummary(data: unknown): EvalSummary {
  if (!isRecord(data)) return {};
  return data as EvalSummary;
}

function fmt(v: number | undefined, digits = 3): string {
  return v !== undefined && v !== null ? v.toFixed(digits) : '-';
}

type ResultsStatus = 'loading' | 'done';

export function ResultsPanel() {
  const [status, setStatus] = useState<ResultsStatus>('loading');
  const [available, setAvailable] = useState(false);
  const [summary, setSummary] = useState<EvalSummary>({});

  useEffect(() => {
    fetchResults()
      .then(({ available: avail, data }) => {
        setAvailable(avail);
        if (avail) setSummary(toEvalSummary(data));
      })
      .catch(() => {
        setAvailable(false);
      })
      .finally(() => setStatus('done'));
  }, []);

  if (status === 'loading') {
    return (
      <div
        role="status"
        aria-live="polite"
        aria-label="Loading evaluation results"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          padding: '3rem 1rem',
          color: 'var(--astryx-color-text-subtle, #6b7280)',
          fontSize: '0.9375rem',
        }}
      >
        <span
          style={{
            display: 'inline-block',
            width: '1.25rem',
            height: '1.25rem',
            borderRadius: '50%',
            border: '2px solid var(--astryx-color-border, #e5e7eb)',
            borderTopColor: '#2563eb',
            animation: 'spin 0.75s linear infinite',
            flexShrink: 0,
          }}
        />
        Loading results…
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (!available) {
    return (
      <div
        style={{
          padding: '3rem 1rem',
          textAlign: 'center',
          color: 'var(--astryx-color-text-subtle, #6b7280)',
        }}
      >
        <p style={{ marginBottom: '0.75rem' }}>
          <MdBarChart size={36} aria-label="Bar chart" color="var(--astryx-color-text-subtle, #6b7280)" />
        </p>
        <p
          style={{
            fontWeight: 600,
            fontSize: '1rem',
            color: 'var(--astryx-color-text, #111827)',
            margin: '0 0 0.5rem',
          }}
        >
          No batch evaluation yet
        </p>
        <p style={{ margin: 0, fontSize: '0.875rem' }}>
          Run{' '}
          <code
            style={{
              background: 'var(--astryx-color-surface-raised, #f9fafb)',
              border: '1px solid var(--astryx-color-border, #e5e7eb)',
              borderRadius: '4px',
              padding: '0.125rem 0.375rem',
              fontSize: '0.8125rem',
            }}
          >
            scripts/eval_tf1.py
          </code>{' '}
          to populate{' '}
          <code
            style={{
              background: 'var(--astryx-color-surface-raised, #f9fafb)',
              border: '1px solid var(--astryx-color-border, #e5e7eb)',
              borderRadius: '4px',
              padding: '0.125rem 0.375rem',
              fontSize: '0.8125rem',
            }}
          >
            results/eval_summary.json
          </code>
          .
        </p>
      </div>
    );
  }

  const { objective, judge_panel, agreement, conclusion, loss_curve } = summary;

  return (
    <div
      style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}
      aria-label="Evaluation results"
    >
      <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 700 }}>Evaluation Results</h2>

      {/* Objective Metrics */}
      {objective && (
        <section aria-labelledby="obj-metrics-heading">
          <h3
            id="obj-metrics-heading"
            style={{ margin: '0 0 0.75rem', fontSize: '1rem', fontWeight: 600 }}
          >
            Objective Metrics
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table
              role="table"
              aria-label="Objective metrics comparison"
              style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}
            >
              <thead>
                <tr>
                  {['Metric', 'Base', 'Fine-tuned', 'Δ (FT − Base)'].map((h) => (
                    <th
                      key={h}
                      style={{
                        textAlign: h === 'Metric' ? 'left' : 'right',
                        padding: '0.5rem 0.75rem',
                        borderBottom: '2px solid var(--astryx-color-border, #e5e7eb)',
                        fontWeight: 600,
                        color: 'var(--astryx-color-text-subtle, #6b7280)',
                        fontSize: '0.75rem',
                        textTransform: 'uppercase',
                        letterSpacing: '0.04em',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ['Perplexity', 'perplexity'],
                    ['Distinct-1', 'distinct_1'],
                    ['Distinct-2', 'distinct_2'],
                    ['Self-BLEU', 'self_bleu'],
                    ['Flesch Reading Ease', 'flesch_reading_ease'],
                  ] as [string, keyof ObjectiveMetrics][]
                ).map(([label, key]) => {
                  const base = objective.base?.[key];
                  const ft = objective.finetuned?.[key];
                  const delta =
                    base !== undefined && ft !== undefined ? ft - base : undefined;
                  return (
                    <tr key={key}>
                      <td
                        style={{
                          padding: '0.5rem 0.75rem',
                          borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                        }}
                      >
                        {label}
                      </td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '0.5rem 0.75rem',
                          borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                          fontVariantNumeric: 'tabular-nums',
                        }}
                      >
                        {fmt(base)}
                      </td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '0.5rem 0.75rem',
                          borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                          fontVariantNumeric: 'tabular-nums',
                        }}
                      >
                        {fmt(ft)}
                      </td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '0.5rem 0.75rem',
                          borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                          fontVariantNumeric: 'tabular-nums',
                          color:
                            delta !== undefined
                              ? delta > 0
                                ? '#16a34a'
                                : delta < 0
                                ? '#dc2626'
                                : 'inherit'
                              : 'inherit',
                        }}
                      >
                        {delta !== undefined ? `${delta > 0 ? '+' : ''}${delta.toFixed(3)}` : '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Judge Panel */}
      {judge_panel && (
        <section aria-labelledby="judge-panel-heading">
          <h3
            id="judge-panel-heading"
            style={{ margin: '0 0 0.75rem', fontSize: '1rem', fontWeight: 600 }}
          >
            Judge Panel
          </h3>
          {judge_panel.judges && judge_panel.judges.length > 0 && (
            <p
              style={{
                margin: '0 0 0.75rem',
                fontSize: '0.8125rem',
                color: 'var(--astryx-color-text-subtle, #6b7280)',
              }}
            >
              Panel of judges: {judge_panel.judges.join(', ')}
            </p>
          )}
          {judge_panel.base && judge_panel.finetuned && (
            <EvalRadar
              series={[
                {
                  name: 'Base',
                  scores: {
                    grammar: judge_panel.base.grammar ?? 0,
                    creativity: judge_panel.base.creativity ?? 0,
                    moral_clarity: judge_panel.base.moral_clarity ?? 0,
                    prompt_adherence: judge_panel.base.prompt_adherence ?? 0,
                  },
                },
                {
                  name: 'Fine-tuned',
                  scores: {
                    grammar: judge_panel.finetuned.grammar ?? 0,
                    creativity: judge_panel.finetuned.creativity ?? 0,
                    moral_clarity: judge_panel.finetuned.moral_clarity ?? 0,
                    prompt_adherence: judge_panel.finetuned.prompt_adherence ?? 0,
                  },
                },
              ]}
            />
          )}
        </section>
      )}

      {/* Agreement */}
      {agreement && (
        <section aria-labelledby="agreement-heading">
          <h3
            id="agreement-heading"
            style={{ margin: '0 0 0.75rem', fontSize: '1rem', fontWeight: 600 }}
          >
            Inter-Judge Agreement
          </h3>
          <p
            style={{
              margin: '0 0 0.5rem',
              fontSize: '0.8125rem',
              color: 'var(--astryx-color-text-subtle, #6b7280)',
            }}
          >
            Cohen's κ measures inter-judge agreement (&gt;0.6 = substantial); Kendall's τ measures
            rank correlation.
          </p>
          <div
            style={{
              display: 'flex',
              gap: '1.5rem',
              flexWrap: 'wrap',
              fontSize: '0.9375rem',
            }}
          >
            {agreement.cohen_kappa !== undefined && (
              <div>
                <span style={{ color: 'var(--astryx-color-text-subtle, #6b7280)', fontSize: '0.8125rem' }}>
                  Cohen's κ{' '}
                </span>
                <strong style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {agreement.cohen_kappa.toFixed(3)}
                </strong>
              </div>
            )}
            {agreement.kendall_tau !== undefined && (
              <div>
                <span style={{ color: 'var(--astryx-color-text-subtle, #6b7280)', fontSize: '0.8125rem' }}>
                  Kendall's τ{' '}
                </span>
                <strong style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {agreement.kendall_tau.toFixed(3)}
                </strong>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Loss Curve */}
      {loss_curve && loss_curve.length > 0 && (
        <section aria-labelledby="loss-curve-heading">
          <h3
            id="loss-curve-heading"
            style={{ margin: '0 0 0.75rem', fontSize: '1rem', fontWeight: 600 }}
          >
            Training Loss Curve
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={loss_curve} aria-label="Training loss curve">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" tick={{ fontSize: 11 }} label={{ value: 'Step', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#2563eb"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      {/* Conclusion */}
      {conclusion && (
        <section aria-labelledby="conclusion-heading">
          <h3
            id="conclusion-heading"
            style={{ margin: '0 0 0.75rem', fontSize: '1rem', fontWeight: 600 }}
          >
            Conclusion
          </h3>
          <div
            style={{
              padding: '1.25rem',
              borderRadius: '8px',
              background: 'var(--astryx-color-surface-raised, #f9fafb)',
              border: '1px solid var(--astryx-color-border, #e5e7eb)',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
            }}
          >
            {conclusion.winner && (
              <p style={{ margin: 0, fontWeight: 700, fontSize: '1.0625rem' }}>
                Winner: {conclusion.winner}
              </p>
            )}
            {conclusion.by_rank && (
              <p style={{ margin: 0, fontSize: '0.9375rem' }}>{conclusion.by_rank}</p>
            )}
            {conclusion.notes && (
              <p
                style={{
                  margin: 0,
                  fontSize: '0.875rem',
                  color: 'var(--astryx-color-text-subtle, #6b7280)',
                }}
              >
                {conclusion.notes}
              </p>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
