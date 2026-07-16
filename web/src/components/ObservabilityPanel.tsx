import { useState } from 'react';
import { MdChevronRight, MdExpandMore, MdInfoOutline } from 'react-icons/md';

// Giải thích từng tham số: ý nghĩa, khoảng giá trị (boundary), ảnh hưởng chất lượng.
const PARAM_INFO: Record<string, string> = {
  temperature:
    'Độ ngẫu nhiên khi chọn token. Khoảng 0-2; app dùng 0.7. Cao (>1.0) đa dạng hơn nhưng dễ lỗi ngữ pháp và lan man (không hội tụ kết); thấp (<0.5) an toàn nhưng lặp và đơn điệu.',
  top_p:
    'Nucleus sampling: chỉ lấy nhóm token có tổng xác suất <= P. Khoảng 0-1; app dùng 0.85. Thấp hơn = hội tụ, ít lạc đề; cao hơn = đa dạng nhưng rủi ro lan man.',
  repetition_penalty:
    'Phạt token đã xuất hiện để tránh lặp. Khoảng 1.0-1.3; app dùng 1.1. Quá cao (>=1.3) phạt cả tên nhân vật -> model đổi nhân vật giữa truyện (entity drift); 1.0 = không phạt, dễ lặp cụm.',
  num_predict:
    'Trần số token sinh ra. Model train seq_len 512 nên trần thực chỉ ~400-460 token (sau khi trừ prompt); đặt cao hơn vô nghĩa. Nếu chạm trần trước khi model phát <|end|> thì truyện bị cắt.',
  seed:
    'Hạt giống ngẫu nhiên. Cùng seed + cùng prompt = cùng truyện (tái lập được để so sánh); "random" = mỗi lần một truyện khác.',
  input_tokens:
    'Số token của prompt gửi vào model. Ăn vào ngân sách context 512, càng nhiều thì chỗ còn lại cho truyện càng ít.',
  output_tokens: 'Số token truyện model sinh ra.',
  latency: 'Thời gian sinh truyện (mili-giây).',
  tokens_per_sec:
    'Tốc độ sinh = output tokens / thời gian. SLM 30M đạt ~950 tok/s, nhanh hơn LLM lớn (Qwen-4B) nhiều lần - đây là lợi thế then chốt của model nhỏ.',
};

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

function MetaRow({
  label,
  value,
  info,
}: {
  label: string;
  value: string | number;
  info?: string;
}) {
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
      <span
        style={{
          color: 'var(--astryx-color-text-subtle, #6b7280)',
          flexShrink: 0,
          display: 'inline-flex',
          alignItems: 'center',
        }}
      >
        {label}
        {info && (
          <MdInfoOutline
            size={13}
            title={info}
            aria-label={info}
            tabIndex={0}
            style={{ marginLeft: '0.25rem', color: 'var(--astryx-color-text-subtle, #9ca3af)', cursor: 'help' }}
          />
        )}
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
        <MetaRow label="Temperature" value={meta.temperature} info={PARAM_INFO.temperature} />
        <MetaRow label="Top P" value={meta.top_p} info={PARAM_INFO.top_p} />
        <MetaRow label="Repetition penalty" value={meta.repetition_penalty} info={PARAM_INFO.repetition_penalty} />
        <MetaRow label="Max tokens" value={meta.num_predict} info={PARAM_INFO.num_predict} />
        {meta.seed !== undefined && meta.seed !== null && (
          <MetaRow label="Seed" value={meta.seed} info={PARAM_INFO.seed} />
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
        <MetaRow label="Input tokens" value={meta.input_tokens} info={PARAM_INFO.input_tokens} />
        <MetaRow label="Output tokens" value={meta.output_tokens} info={PARAM_INFO.output_tokens} />
        <MetaRow label="Latency" value={`${meta.latency_ms.toFixed(0)} ms`} info={PARAM_INFO.latency} />
        <MetaRow label="Tokens / sec" value={meta.tokens_per_sec.toFixed(1)} info={PARAM_INFO.tokens_per_sec} />
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
          {promptOpen ? (
            <MdExpandMore size={16} aria-hidden="true" />
          ) : (
            <MdChevronRight size={16} aria-hidden="true" />
          )}
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
