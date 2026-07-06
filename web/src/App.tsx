import { useState, useCallback } from 'react';
import { TabList } from '@astryxdesign/core/TabList';
import { Tab } from '@astryxdesign/core/TabList';
import { InputPanel } from './components/InputPanel';
import type { FablePayload } from './components/InputPanel';
import { StoryStream } from './components/StoryStream';
import type { StoryStreamState } from './components/StoryStream';
import { LogPanel } from './components/LogPanel';
import type { LogEntry } from './components/LogPanel';
import { ObservabilityPanel } from './components/ObservabilityPanel';
import type { FableMeta } from './components/ObservabilityPanel';
import { CompareMode } from './components/CompareMode';
import { streamFable } from './api';
import type { SSEEvent } from './api';

type AppTab = 'playground' | 'results';
type PlaygroundMode = 'single' | 'compare';

let logIdCounter = 0;
function nextLogId(): string {
  return `log-${++logIdCounter}`;
}

function App() {
  const [activeTab, setActiveTab] = useState<AppTab>('playground');
  const [playgroundMode, setPlaygroundMode] = useState<PlaygroundMode>('single');

  // Single mode state
  const [streamState, setStreamState] = useState<StoryStreamState>('empty');
  const [tokens, setTokens] = useState('');
  const [finalStory, setFinalStory] = useState<string | undefined>();
  const [reason, setReason] = useState<string | undefined>();
  const [meta, setMeta] = useState<FableMeta | null>(null);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);

  // Compare mode: InputPanel fires a payload; CompareMode picks it up
  const [comparePayload, setComparePayload] = useState<Omit<FablePayload, 'model_id'> | null>(
    null,
  );

  function handleSingleSubmit(payload: FablePayload) {
    // Reset all state for a new generation
    setStreamState('generating');
    setTokens('');
    setFinalStory(undefined);
    setReason(undefined);
    setMeta(null);
    setLogEntries([]);

    streamFable(payload, (e: SSEEvent) => {
      if (e.type === 'step') {
        const entry: LogEntry = {
          id: nextLogId(),
          stage: e.stage,
          status: e.status,
          detail: e.detail,
          timestamp: new Date(),
        };
        setLogEntries((prev) => {
          // Replace a running entry for the same stage when it transitions to ok/blocked
          let idx = -1;
          for (let i = prev.length - 1; i >= 0; i--) {
            if (prev[i].stage === e.stage) { idx = i; break; }
          }
          if (idx >= 0 && prev[idx].status === 'running' && e.status !== 'running') {
            const updated = [...prev];
            updated[idx] = entry;
            return updated;
          }
          return [...prev, entry];
        });
      } else if (e.type === 'token') {
        setTokens((prev) => prev + e.text);
      } else if (e.type === 'done') {
        if (e.status === 'refused') {
          setStreamState('refused');
          setReason(e.reason);
        } else {
          setStreamState('done');
          setFinalStory(e.story);
          if (e.meta) setMeta(e.meta as FableMeta);
        }
      } else if (e.type === 'error') {
        setStreamState('error');
        setReason(e.reason);
      }
    }).catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      setStreamState('error');
      setReason(msg);
    });
  }

  function handleCompareSubmit(payload: FablePayload) {
    // Strip model_id — CompareMode manages its own model selections
    const { model_id: _ignored, ...narrativeFields } = payload;
    void _ignored;
    setComparePayload(narrativeFields);
  }

  const handleCompareGenerationStarted = useCallback(() => {
    setComparePayload(null);
  }, []);

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header
        style={{
          padding: '1.25rem 2rem 0',
          borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
        }}
      >
        <div style={{ marginBottom: '0.75rem' }}>
          <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>
            English Fable Generator
          </h1>
          <p
            style={{
              margin: '0.25rem 0 0',
              color: 'var(--astryx-color-text-subtle, #6b7280)',
              fontSize: '0.875rem',
            }}
          >
            Generate creative fables using fine-tuned language models
          </p>
        </div>
        <TabList value={activeTab} onChange={(v) => setActiveTab(v as AppTab)}>
          <Tab value="playground" label="Playground" />
          <Tab value="results" label="Results" />
        </TabList>
      </header>

      {/* ── Tab: Results ── */}
      {activeTab === 'results' && (
        <main
          style={{
            padding: '3rem 2rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
            textAlign: 'center',
          }}
        >
          <div>
            <p style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>📊</p>
            <p
              style={{
                margin: 0,
                fontWeight: 600,
                fontSize: '1rem',
                color: 'var(--astryx-color-text, #111827)',
              }}
            >
              Results
            </p>
            <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem' }}>
              Results — coming after evaluation
            </p>
          </div>
        </main>
      )}

      {/* ── Tab: Playground ── */}
      {activeTab === 'playground' && (
        <main
          style={{
            padding: '1.5rem 2rem',
            maxWidth: '1600px',
            margin: '0 auto',
            width: '100%',
            boxSizing: 'border-box',
          }}
        >
          {/* Mode toggle */}
          <div
            role="group"
            aria-label="Playground mode"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginBottom: '1.25rem',
            }}
          >
            {(['single', 'compare'] as PlaygroundMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setPlaygroundMode(mode)}
                aria-pressed={playgroundMode === mode}
                style={{
                  padding: '0.375rem 0.875rem',
                  borderRadius: '9999px',
                  border: '1px solid',
                  borderColor:
                    playgroundMode === mode
                      ? 'var(--astryx-color-primary, #2563eb)'
                      : 'var(--astryx-color-border, #e5e7eb)',
                  background:
                    playgroundMode === mode ? 'var(--astryx-color-primary, #2563eb)' : 'transparent',
                  color: playgroundMode === mode ? '#fff' : 'var(--astryx-color-text, #111827)',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
              >
                {mode === 'single' ? 'Single' : 'Compare'}
              </button>
            ))}
          </div>

          {/* ── Single mode: 3-column layout ── */}
          {playgroundMode === 'single' && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 3fr 1fr',
                gap: '1.5rem',
                alignItems: 'start',
              }}
            >
              {/* Left: Input panel */}
              <div>
                <InputPanel onSubmit={handleSingleSubmit} />
              </div>

              {/* Center: Story stream */}
              <div style={{ minHeight: '400px' }}>
                <StoryStream
                  state={streamState}
                  tokens={tokens}
                  finalStory={finalStory}
                  reason={reason}
                />
              </div>

              {/* Right: Log + Observability stacked */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <LogPanel entries={logEntries} />
                <ObservabilityPanel meta={meta} />
              </div>
            </div>
          )}

          {/* ── Compare mode ── */}
          {playgroundMode === 'compare' && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '280px 1fr',
                gap: '1.5rem',
                alignItems: 'start',
              }}
            >
              {/* Left: Input panel */}
              <div>
                <InputPanel onSubmit={handleCompareSubmit} />
              </div>

              {/* Right: Compare columns */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <CompareMode
                  pendingPayload={comparePayload}
                  onGenerationStarted={handleCompareGenerationStarted}
                />
              </div>
            </div>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
