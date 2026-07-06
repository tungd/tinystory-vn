import { useState, useEffect, useRef } from 'react';
import { fetchModels, streamFable, evaluate } from '../api';
import type { SSEEvent } from '../api';
import { StoryStream } from './StoryStream';
import type { StoryStreamState } from './StoryStream';
import type { FableMeta } from './ObservabilityPanel';
import type { FablePayload } from './InputPanel';
import { EvalRadar } from './EvalRadar';

interface ModelInfo {
  model_id: string;
  name: string;
  kind?: string;
  desc?: string;
}

interface SlotState {
  streamState: StoryStreamState;
  tokens: string;
  finalStory?: string;
  reason?: string;
  meta?: FableMeta;
}

const EMPTY_SLOT: SlotState = {
  streamState: 'empty',
  tokens: '',
};

type EvalState = 'idle' | 'loading' | 'done' | 'error';

interface EvalScores {
  grammar: number;
  creativity: number;
  moral_clarity: number;
  prompt_adherence: number;
  overall: number;
}

interface SlotEval {
  state: EvalState;
  scores: EvalScores | null;
  errorMsg?: string;
}

const EMPTY_EVAL: SlotEval = { state: 'idle', scores: null };

interface CompactObsProps {
  meta?: FableMeta;
  streamState: StoryStreamState;
}

function CompactObs({ meta, streamState }: CompactObsProps) {
  if (!meta) {
    if (streamState === 'generating') {
      return (
        <div
          style={{
            fontSize: '0.75rem',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
            fontStyle: 'italic',
          }}
        >
          Generating…
        </div>
      );
    }
    return null;
  }

  return (
    <div
      style={{
        display: 'flex',
        gap: '1rem',
        fontSize: '0.75rem',
        color: 'var(--astryx-color-text-subtle, #6b7280)',
        flexWrap: 'wrap',
      }}
    >
      <span>
        <strong style={{ color: 'var(--astryx-color-text, #111827)' }}>
          {meta.tokens_per_sec.toFixed(1)}
        </strong>{' '}
        tok/s
      </span>
      <span>
        <strong style={{ color: 'var(--astryx-color-text, #111827)' }}>
          {meta.latency_ms.toFixed(0)}
        </strong>{' '}
        ms
      </span>
      <span>
        <strong style={{ color: 'var(--astryx-color-text, #111827)' }}>
          {meta.output_tokens}
        </strong>{' '}
        tokens
      </span>
    </div>
  );
}

interface ModelDropdownProps {
  label: string;
  models: ModelInfo[];
  value: string;
  onChange: (v: string) => void;
  disabledValue?: string;
}

function ModelDropdown({ label, models, value, onChange, disabledValue }: ModelDropdownProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
      <label
        style={{
          fontSize: '0.75rem',
          fontWeight: 600,
          color: 'var(--astryx-color-text-subtle, #6b7280)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}
      >
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          padding: '0.375rem 0.625rem',
          borderRadius: '6px',
          border: '1px solid var(--astryx-color-border, #e5e7eb)',
          background: 'var(--astryx-color-surface, #fff)',
          fontSize: '0.875rem',
          color: 'var(--astryx-color-text, #111827)',
          cursor: 'pointer',
          width: '100%',
        }}
      >
        {models.map((m) => (
          <option key={m.model_id} value={m.model_id} disabled={m.model_id === disabledValue}>
            {m.name}
            {m.kind ? ` (${m.kind})` : ''}
          </option>
        ))}
      </select>
    </div>
  );
}

export interface CompareModeProps {
  /**
   * Narrative fields from InputPanel — passed in when user hits Generate.
   * null = no generation triggered yet.
   */
  pendingPayload: Omit<FablePayload, 'model_id'> | null;
  /** Called after both streams have been kicked off, to reset pending state */
  onGenerationStarted: () => void;
}

export function CompareMode({ pendingPayload, onGenerationStarted }: CompareModeProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelA, setModelA] = useState('');
  const [modelB, setModelB] = useState('');
  const [slotA, setSlotA] = useState<SlotState>(EMPTY_SLOT);
  const [slotB, setSlotB] = useState<SlotState>(EMPTY_SLOT);
  const [evalA, setEvalA] = useState<SlotEval>(EMPTY_EVAL);
  const [evalB, setEvalB] = useState<SlotEval>(EMPTY_EVAL);

  // Monotonic generation counter — guards against double-firing the same generation
  const generationCounterRef = useRef<number>(0);
  const lastFiredCounterRef = useRef<number>(-1);

  useEffect(() => {
    fetchModels()
      .then((data: ModelInfo[]) => {
        setModels(data);
        if (data.length >= 1) setModelA(data[0].model_id);
        if (data.length >= 2) setModelB(data[1].model_id);
      })
      .catch(() => {
        // Graceful fallback
      })
      .finally(() => setModelsLoading(false));
  }, []);

  const canCompare = models.length >= 2;

  // Auto-eval for slot A when it finishes
  useEffect(() => {
    if (slotA.streamState === 'done' && slotA.finalStory) {
      setEvalA({ state: 'loading', scores: null });
      const prompt = slotA.meta?.prompt_sent ?? '';
      evaluate(slotA.finalStory, prompt).then((result) => {
        if ('error' in result) {
          setEvalA({ state: 'error', scores: null, errorMsg: result.error as string });
        } else {
          setEvalA({ state: 'done', scores: result as EvalScores });
        }
      }).catch((err: unknown) => {
        setEvalA({ state: 'error', scores: null, errorMsg: err instanceof Error ? err.message : String(err) });
      });
    } else if (slotA.streamState !== 'done') {
      setEvalA(EMPTY_EVAL);
    }
  }, [slotA.streamState, slotA.finalStory, slotA.meta?.prompt_sent]);

  // Auto-eval for slot B when it finishes
  useEffect(() => {
    if (slotB.streamState === 'done' && slotB.finalStory) {
      setEvalB({ state: 'loading', scores: null });
      const prompt = slotB.meta?.prompt_sent ?? '';
      evaluate(slotB.finalStory, prompt).then((result) => {
        if ('error' in result) {
          setEvalB({ state: 'error', scores: null, errorMsg: result.error as string });
        } else {
          setEvalB({ state: 'done', scores: result as EvalScores });
        }
      }).catch((err: unknown) => {
        setEvalB({ state: 'error', scores: null, errorMsg: err instanceof Error ? err.message : String(err) });
      });
    } else if (slotB.streamState !== 'done') {
      setEvalB(EMPTY_EVAL);
    }
  }, [slotB.streamState, slotB.finalStory, slotB.meta?.prompt_sent]);

  // Fire off both streams when pendingPayload arrives
  useEffect(() => {
    if (!pendingPayload || !canCompare || !modelA || !modelB) return;

    // Increment counter for each new generation trigger
    const currentGen = ++generationCounterRef.current;
    // Guard: if this generation was already fired, skip
    if (lastFiredCounterRef.current === currentGen) return;
    lastFiredCounterRef.current = currentGen;

    onGenerationStarted();

    const payloadA: FablePayload = { ...pendingPayload, model_id: modelA };
    const payloadB: FablePayload = { ...pendingPayload, model_id: modelB };

    // Reset both slots
    setSlotA({ streamState: 'generating', tokens: '' });
    setSlotB({ streamState: 'generating', tokens: '' });

    // Stream A
    streamFable(payloadA, (e: SSEEvent) => {
      if (e.type === 'token') {
        setSlotA((prev) => ({ ...prev, tokens: prev.tokens + e.text }));
      } else if (e.type === 'done') {
        if (e.status === 'refused') {
          setSlotA({ streamState: 'refused', tokens: '', reason: e.reason });
        } else {
          setSlotA({
            streamState: 'done',
            tokens: '',
            finalStory: e.story,
            meta: e.meta as FableMeta | undefined,
          });
        }
      } else if (e.type === 'error') {
        setSlotA({ streamState: 'error', tokens: '', reason: e.reason });
      }
    }).catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      setSlotA({ streamState: 'error', tokens: '', reason: msg });
    });

    // Stream B
    streamFable(payloadB, (e: SSEEvent) => {
      if (e.type === 'token') {
        setSlotB((prev) => ({ ...prev, tokens: prev.tokens + e.text }));
      } else if (e.type === 'done') {
        if (e.status === 'refused') {
          setSlotB({ streamState: 'refused', tokens: '', reason: e.reason });
        } else {
          setSlotB({
            streamState: 'done',
            tokens: '',
            finalStory: e.story,
            meta: e.meta as FableMeta | undefined,
          });
        }
      } else if (e.type === 'error') {
        setSlotB({ streamState: 'error', tokens: '', reason: e.reason });
      }
    }).catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      setSlotB({ streamState: 'error', tokens: '', reason: msg });
    });
  }, [pendingPayload, canCompare, modelA, modelB, onGenerationStarted]);

  if (modelsLoading) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '3rem',
          color: 'var(--astryx-color-text-subtle, #6b7280)',
        }}
      >
        Loading models…
      </div>
    );
  }

  if (!canCompare) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '3rem',
          gap: '0.75rem',
          color: 'var(--astryx-color-text-subtle, #6b7280)',
          textAlign: 'center',
        }}
      >
        <span style={{ fontSize: '2rem' }}>⚖️</span>
        <p style={{ margin: 0, fontWeight: 600, color: 'var(--astryx-color-text, #111827)' }}>
          Compare mode unavailable
        </p>
        <p style={{ margin: 0, fontSize: '0.875rem', maxWidth: '320px' }}>
          Add a fine-tuned model to compare. At least 2 models are required.
        </p>
      </div>
    );
  }

  const bothDone =
    (slotA.streamState === 'done' || slotA.streamState === 'refused' || slotA.streamState === 'error') &&
    (slotB.streamState === 'done' || slotB.streamState === 'refused' || slotB.streamState === 'error');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', flex: 1 }}>
      {/* Model selectors row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '1rem',
        }}
      >
        <ModelDropdown
          label="Model A"
          models={models}
          value={modelA}
          onChange={setModelA}
          disabledValue={modelB}
        />
        <ModelDropdown
          label="Model B"
          models={models}
          value={modelB}
          onChange={setModelB}
          disabledValue={modelA}
        />
      </div>

      {/* Side-by-side story columns */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '1rem',
          flex: 1,
          minHeight: '400px',
        }}
      >
        {/* Column A */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--astryx-color-text-subtle, #6b7280)',
            }}
          >
            Model A — {models.find((m) => m.model_id === modelA)?.name ?? modelA}
          </div>
          <CompactObs meta={slotA.meta} streamState={slotA.streamState} />
          <div style={{ flex: 1 }}>
            <StoryStream
              state={slotA.streamState}
              tokens={slotA.tokens}
              finalStory={slotA.finalStory}
              reason={slotA.reason}
            />
          </div>
        </div>

        {/* Column B */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--astryx-color-text-subtle, #6b7280)',
            }}
          >
            Model B — {models.find((m) => m.model_id === modelB)?.name ?? modelB}
          </div>
          <CompactObs meta={slotB.meta} streamState={slotB.streamState} />
          <div style={{ flex: 1 }}>
            <StoryStream
              state={slotB.streamState}
              tokens={slotB.tokens}
              finalStory={slotB.finalStory}
              reason={slotB.reason}
            />
          </div>
        </div>
      </div>

      {/* Verdict — shown once both slots have finished */}
      {bothDone && (
        <div
          style={{
            border: '1px solid var(--astryx-color-border, #e5e7eb)',
            borderRadius: '8px',
            padding: '1.5rem',
          }}
        >
          <p style={{ margin: '0 0 1rem', fontWeight: 700, fontSize: '1rem' }}>⚖️ Verdict</p>
          <p
            style={{
              margin: '0 0 0.75rem',
              fontSize: '0.75rem',
              color: 'var(--astryx-color-text-subtle, #6b7280)',
            }}
          >
            Quick 1-judge indicator — see Results tab for canonical metrics.
          </p>

          {/* Loading state */}
          {(evalA.state === 'loading' || evalB.state === 'loading') && (
            <div
              role="status"
              aria-live="polite"
              aria-label="Evaluating both models"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
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
                  flexShrink: 0,
                }}
              />
              Evaluating both models…
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
          )}

          {/* Radar + table when both done */}
          {evalA.state === 'done' && evalB.state === 'done' && evalA.scores && evalB.scores && (() => {
            const modelAName = models.find((m) => m.model_id === modelA)?.name ?? 'Model A';
            const modelBName = models.find((m) => m.model_id === modelB)?.name ?? 'Model B';
            const axes = ['grammar', 'creativity', 'moral_clarity', 'prompt_adherence'] as const;
            const axisLabels: Record<string, string> = {
              grammar: 'Grammar',
              creativity: 'Creativity',
              moral_clarity: 'Moral Clarity',
              prompt_adherence: 'Prompt Adherence',
            };
            const aWins = axes.filter((ax) => evalA.scores![ax] > evalB.scores![ax]).length;
            const bWins = axes.filter((ax) => evalB.scores![ax] > evalA.scores![ax]).length;
            const overallDelta = evalB.scores.overall - evalA.scores.overall;
            let verdictText: string;
            if (bWins > aWins) {
              verdictText = `Model B ranks higher on ${bWins}/4 axes (overall ${overallDelta > 0 ? '+' : ''}${overallDelta.toFixed(2)})`;
            } else if (aWins > bWins) {
              verdictText = `Model A ranks higher on ${aWins}/4 axes (overall ${(-overallDelta) > 0 ? '+' : ''}${(-overallDelta).toFixed(2)} for A)`;
            } else {
              verdictText = `Tie — each wins ${aWins}/4 axes (overall Δ ${overallDelta > 0 ? '+' : ''}${overallDelta.toFixed(2)})`;
            }

            return (
              <>
                <EvalRadar
                  series={[
                    { name: `Model A — ${modelAName}`, scores: evalA.scores },
                    { name: `Model B — ${modelBName}`, scores: evalB.scores },
                  ]}
                />

                <table
                  role="table"
                  aria-label="Per-axis score comparison"
                  style={{
                    width: '100%',
                    borderCollapse: 'collapse',
                    fontSize: '0.8125rem',
                    marginTop: '1rem',
                  }}
                >
                  <thead>
                    <tr>
                      <th
                        style={{
                          textAlign: 'left',
                          padding: '0.375rem 0.5rem',
                          borderBottom: '2px solid var(--astryx-color-border, #e5e7eb)',
                        }}
                      >
                        Axis
                      </th>
                      <th
                        style={{
                          textAlign: 'right',
                          padding: '0.375rem 0.5rem',
                          borderBottom: '2px solid var(--astryx-color-border, #e5e7eb)',
                          color: '#2563eb',
                        }}
                      >
                        A
                      </th>
                      <th
                        style={{
                          textAlign: 'right',
                          padding: '0.375rem 0.5rem',
                          borderBottom: '2px solid var(--astryx-color-border, #e5e7eb)',
                          color: '#16a34a',
                        }}
                      >
                        B
                      </th>
                      <th
                        style={{
                          textAlign: 'right',
                          padding: '0.375rem 0.5rem',
                          borderBottom: '2px solid var(--astryx-color-border, #e5e7eb)',
                        }}
                      >
                        Δ (B−A)
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {axes.map((axis) => {
                      const a = evalA.scores![axis];
                      const b = evalB.scores![axis];
                      const delta = b - a;
                      return (
                        <tr key={axis}>
                          <td
                            style={{
                              padding: '0.375rem 0.5rem',
                              borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                            }}
                          >
                            {axisLabels[axis]}
                          </td>
                          <td
                            style={{
                              textAlign: 'right',
                              padding: '0.375rem 0.5rem',
                              borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                              color: '#2563eb',
                              fontVariantNumeric: 'tabular-nums',
                            }}
                          >
                            {a.toFixed(1)}
                          </td>
                          <td
                            style={{
                              textAlign: 'right',
                              padding: '0.375rem 0.5rem',
                              borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                              color: '#16a34a',
                              fontVariantNumeric: 'tabular-nums',
                            }}
                          >
                            {b.toFixed(1)}
                          </td>
                          <td
                            style={{
                              textAlign: 'right',
                              padding: '0.375rem 0.5rem',
                              borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
                              fontVariantNumeric: 'tabular-nums',
                              color: delta > 0 ? '#16a34a' : delta < 0 ? '#dc2626' : 'inherit',
                            }}
                          >
                            {delta > 0 ? '+' : ''}
                            {delta.toFixed(1)}
                          </td>
                        </tr>
                      );
                    })}
                    <tr>
                      <td style={{ padding: '0.375rem 0.5rem', fontWeight: 600 }}>Overall</td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '0.375rem 0.5rem',
                          color: '#2563eb',
                          fontWeight: 600,
                          fontVariantNumeric: 'tabular-nums',
                        }}
                      >
                        {evalA.scores.overall.toFixed(2)}
                      </td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '0.375rem 0.5rem',
                          color: '#16a34a',
                          fontWeight: 600,
                          fontVariantNumeric: 'tabular-nums',
                        }}
                      >
                        {evalB.scores.overall.toFixed(2)}
                      </td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '0.375rem 0.5rem',
                          fontWeight: 600,
                          fontVariantNumeric: 'tabular-nums',
                          color:
                            overallDelta > 0
                              ? '#16a34a'
                              : overallDelta < 0
                              ? '#dc2626'
                              : 'inherit',
                        }}
                      >
                        {overallDelta > 0 ? '+' : ''}
                        {overallDelta.toFixed(2)}
                      </td>
                    </tr>
                  </tbody>
                </table>

                <p
                  aria-live="polite"
                  style={{
                    margin: '1rem 0 0',
                    padding: '0.75rem',
                    background: 'var(--astryx-color-surface-raised, #f9fafb)',
                    borderRadius: '6px',
                    fontWeight: 600,
                    fontSize: '0.9375rem',
                  }}
                >
                  {verdictText} — {modelAName} vs {modelBName}
                </p>
              </>
            );
          })()}

          {/* Partial errors */}
          {(evalA.state === 'error' || evalB.state === 'error') && (
            <p
              role="alert"
              style={{ color: '#dc2626', fontSize: '0.8125rem', margin: '0.5rem 0 0' }}
            >
              {evalA.state === 'error' ? `Model A eval error: ${evalA.errorMsg ?? 'Unknown'}` : ''}
              {evalA.state === 'error' && evalB.state === 'error' ? ' ' : ''}
              {evalB.state === 'error' ? `Model B eval error: ${evalB.errorMsg ?? 'Unknown'}` : ''}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
