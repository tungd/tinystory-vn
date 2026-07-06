import { useEffect, useRef } from 'react';
import { MdAutoStories, MdCheckCircle, MdBlock, MdErrorOutline } from 'react-icons/md';

export type StoryStreamState = 'empty' | 'generating' | 'done' | 'refused' | 'error';

export interface StoryStreamProps {
  state: StoryStreamState;
  tokens: string;       // accumulated token string during streaming
  finalStory?: string;  // full story from done event
  reason?: string;      // reason for refused or error
}

/**
 * Escapes HTML special characters to prevent XSS.
 */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Simple markdown-safe renderer: escape first, then minimal formatting.
 * Supports: **bold**, blank-line paragraphs.
 * No external libs needed.
 */
export function renderMarkdown(text: string): string {
  const escaped = escapeHtml(text);

  // Split on blank lines into paragraphs
  const paragraphs = escaped.split(/\n\s*\n/).filter((p) => p.trim().length > 0);

  const rendered = paragraphs
    .map((paragraph) => {
      // Replace **bold** with <strong> (already escaped so ** is literal)
      const withBold = paragraph.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      // Replace single newlines within a paragraph with <br>
      const withLineBreaks = withBold.replace(/\n/g, '<br>');
      return `<p style="margin:0 0 0.875rem;">${withLineBreaks}</p>`;
    })
    .join('');

  return rendered;
}

export function StoryStream({ state, tokens, finalStory, reason }: StoryStreamProps) {
  const streamRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom while streaming
  useEffect(() => {
    if (state === 'generating' && streamRef.current) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight;
    }
  }, [tokens, state]);

  const containerStyle: React.CSSProperties = {
    border: '1px solid var(--astryx-color-border, #e5e7eb)',
    borderRadius: '8px',
    padding: '1.5rem',
    minHeight: '300px',
    background: 'var(--astryx-color-surface, #fff)',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    height: '100%',
    boxSizing: 'border-box',
  };

  if (state === 'empty') {
    return (
      <div style={{ ...containerStyle, alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', color: 'var(--astryx-color-text-subtle, #6b7280)' }}>
          <div style={{ marginBottom: '0.75rem' }}>
            <MdAutoStories size={40} aria-label="Book" color="var(--astryx-color-text-subtle, #6b7280)" />
          </div>
          <p style={{ margin: 0, fontSize: '0.9375rem', fontWeight: 500 }}>
            Your fable will appear here
          </p>
          <p style={{ margin: '0.25rem 0 0', fontSize: '0.8125rem' }}>
            Fill in the details and click Generate
          </p>
        </div>
      </div>
    );
  }

  if (state === 'generating') {
    const hasTokens = tokens.length > 0;
    return (
      <div style={containerStyle}>
        <style>{`
          @keyframes ss-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
          }
          @keyframes ss-blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
          }
        `}</style>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
            fontSize: '0.8125rem',
          }}
        >
          <span
            style={{
              display: 'inline-block',
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: 'var(--astryx-color-primary, #2563eb)',
              animation: 'ss-pulse 1s ease-in-out infinite',
            }}
          />
          Generating…
        </div>
        <div
          ref={streamRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            lineHeight: 1.7,
            fontSize: '0.9375rem',
            color: 'var(--astryx-color-text, #111827)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {hasTokens ? (
            <>
              {tokens}
              <span
                style={{
                  display: 'inline-block',
                  width: '2px',
                  height: '1em',
                  background: 'var(--astryx-color-primary, #2563eb)',
                  marginLeft: '2px',
                  verticalAlign: 'text-bottom',
                  animation: 'ss-blink 1s step-end infinite',
                }}
              />
            </>
          ) : (
            <div
              style={{
                color: 'var(--astryx-color-text-subtle, #6b7280)',
                fontStyle: 'italic',
                fontSize: '0.875rem',
              }}
            >
              Processing your request…
            </div>
          )}
        </div>
      </div>
    );
  }

  if (state === 'done' && finalStory) {
    return (
      <div style={containerStyle}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: 'var(--astryx-color-success, #16a34a)',
            fontSize: '0.8125rem',
            fontWeight: 500,
          }}
        >
          <MdCheckCircle size={16} aria-hidden="true" color="var(--astryx-color-success, #16a34a)" />
          Story generated
        </div>
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            lineHeight: 1.75,
            fontSize: '0.9375rem',
            color: 'var(--astryx-color-text, #111827)',
          }}
          dangerouslySetInnerHTML={{ __html: renderMarkdown(finalStory) }}
        />
      </div>
    );
  }

  if (state === 'refused') {
    return (
      <div style={containerStyle}>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            flex: 1,
            gap: '0.75rem',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
          }}
        >
          <MdBlock size={32} aria-label="Blocked" color="var(--astryx-color-danger, #dc2626)" />
          <p
            style={{
              margin: 0,
              fontWeight: 600,
              fontSize: '0.9375rem',
              color: 'var(--astryx-color-text, #111827)',
            }}
          >
            Request refused by guardrail
          </p>
          {reason && (
            <p
              style={{
                margin: 0,
                fontSize: '0.875rem',
                textAlign: 'center',
                maxWidth: '400px',
              }}
            >
              {reason}
            </p>
          )}
        </div>
      </div>
    );
  }

  if (state === 'error') {
    return (
      <div style={containerStyle}>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            flex: 1,
            gap: '0.75rem',
          }}
        >
          <MdErrorOutline size={32} aria-label="Error" color="#dc2626" />
          <p
            style={{
              margin: 0,
              fontWeight: 600,
              fontSize: '0.9375rem',
              color: 'var(--astryx-color-text, #111827)',
            }}
          >
            An error occurred
          </p>
          {reason && (
            <p
              style={{
                margin: 0,
                fontSize: '0.875rem',
                color: 'var(--astryx-color-text-subtle, #6b7280)',
                textAlign: 'center',
                maxWidth: '400px',
              }}
            >
              {reason}
            </p>
          )}
        </div>
      </div>
    );
  }

  // fallback for done with no story
  return (
    <div style={{ ...containerStyle, alignItems: 'center', justifyContent: 'center' }}>
      <span style={{ color: 'var(--astryx-color-text-subtle, #6b7280)' }}>No content</span>
    </div>
  );
}
