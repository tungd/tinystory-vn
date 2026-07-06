import { useState } from 'react';
import { InputPanel, FablePayload } from './components/InputPanel';

function App() {
  const [lastPayload, setLastPayload] = useState<FablePayload | null>(null);

  function handleSubmit(payload: FablePayload) {
    console.log('[InputPanel] submitted payload:', payload);
    setLastPayload(payload);
  }

  return (
    <div className="app-shell">
      <header
        style={{
          padding: '1.5rem 2rem',
          borderBottom: '1px solid var(--astryx-color-border, #e5e7eb)',
        }}
      >
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
      </header>

      <main
        style={{
          padding: '2rem',
          display: 'grid',
          gridTemplateColumns: '1fr 3fr 1fr',
          gap: '1.5rem',
          maxWidth: '1400px',
          margin: '0 auto',
        }}
      >
        {/* Left column: Input panel */}
        <div>
          <InputPanel onSubmit={handleSubmit} />
        </div>

        {/* Center column: Story output (wired in next task) */}
        <div
          style={{
            border: '1px dashed var(--astryx-color-border, #e5e7eb)',
            borderRadius: '8px',
            padding: '1.5rem',
            minHeight: '300px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
          }}
        >
          {lastPayload ? (
            <pre
              style={{
                fontSize: '0.75rem',
                textAlign: 'left',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                width: '100%',
              }}
            >
              {JSON.stringify(lastPayload, null, 2)}
            </pre>
          ) : (
            <span>Story output — coming in Task 8</span>
          )}
        </div>

        {/* Right column: Log (wired in next task) */}
        <div
          style={{
            border: '1px dashed var(--astryx-color-border, #e5e7eb)',
            borderRadius: '8px',
            padding: '1.5rem',
            minHeight: '300px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
            fontSize: '0.875rem',
          }}
        >
          Log — coming in Task 8
        </div>
      </main>
    </div>
  );
}

export default App;
