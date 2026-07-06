import { useState } from 'react';
import { MdCasino } from 'react-icons/md';
import { TextArea } from '@astryxdesign/core/TextArea';
import { SegmentedControl } from '@astryxdesign/core/SegmentedControl';
import { SegmentedControlItem } from '@astryxdesign/core/SegmentedControl';
import { Switch } from '@astryxdesign/core/Switch';
import { Button } from '@astryxdesign/core/Button';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { ModelSelect } from './ModelSelect';
import { PRESETS } from '../presets';

export type FableLength = 'short' | 'medium' | 'long';

export interface FablePayload {
  character: string;
  setting: string;
  challenge: string;
  outcome: string;
  teaching: string;
  length: FableLength;
  model_id: string;
  guardrail_enabled: boolean;
}

export interface InputPanelProps {
  onSubmit: (payload: FablePayload) => void;
  /** When true, generation is in progress: lock all inputs and show a busy button. */
  busy?: boolean;
}

export function InputPanel({ onSubmit, busy = false }: InputPanelProps) {
  const [character, setCharacter] = useState('');
  const [setting, setSetting] = useState('');
  const [challenge, setChallenge] = useState('');
  const [outcome, setOutcome] = useState('');
  const [teaching, setTeaching] = useState('');
  const [length, setLength] = useState<FableLength>('medium');
  const [modelId, setModelId] = useState('');
  const [guardrail, setGuardrail] = useState(true);

  function applyPreset(presetId: string) {
    const preset = PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setCharacter(preset.character);
    setSetting(preset.setting);
    setChallenge(preset.challenge);
    setOutcome(preset.outcome);
    setTeaching(preset.teaching);
  }

  function handleSurpriseMe() {
    const randomPreset = PRESETS[Math.floor(Math.random() * PRESETS.length)];
    applyPreset(randomPreset.id);
  }

  function handleSubmit() {
    onSubmit({
      character,
      setting,
      challenge,
      outcome,
      teaching,
      length,
      model_id: modelId,
      guardrail_enabled: guardrail,
    });
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem',
        padding: '1.5rem',
        background: 'var(--astryx-color-surface, #fff)',
        border: '1px solid var(--astryx-color-border, #e5e7eb)',
        borderRadius: '8px',
      }}
    >
      {/* Preset chips */}
      <div>
        <p
          style={{
            margin: '0 0 0.5rem',
            fontSize: '0.75rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--astryx-color-text-subtle, #6b7280)',
          }}
        >
          Presets
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => applyPreset(preset.id)}
              disabled={busy}
              aria-label={`Use "${preset.label}" preset`}
              style={{
                padding: '0.25rem 0.75rem',
                fontSize: '0.8125rem',
                borderRadius: '9999px',
                border: '1px solid var(--astryx-color-border, #e5e7eb)',
                background: 'var(--astryx-color-surface-raised, #f9fafb)',
                cursor: busy ? 'not-allowed' : 'pointer',
                opacity: busy ? 0.5 : 1,
                color: 'var(--astryx-color-text, #111827)',
                transition: 'background 0.15s',
              }}
            >
              {preset.label}
            </button>
          ))}
          <button
            onClick={handleSurpriseMe}
            disabled={busy}
            aria-label="Fill with a random preset"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.375rem',
              padding: '0.25rem 0.75rem',
              fontSize: '0.8125rem',
              borderRadius: '9999px',
              border: '1px dashed var(--astryx-color-border-emphasis, #9ca3af)',
              background: 'transparent',
              cursor: busy ? 'not-allowed' : 'pointer',
              opacity: busy ? 0.5 : 1,
              color: 'var(--astryx-color-text-subtle, #6b7280)',
              transition: 'background 0.15s',
            }}
          >
            <MdCasino size={14} aria-hidden="true" style={{ flexShrink: 0 }} /> Surprise me
          </button>
        </div>
      </div>

      {/* Fable fields */}
      <TextArea
        label="Main character"
        placeholder="e.g. a clever fox"
        value={character}
        onChange={(v) => setCharacter(v)}
        rows={2}
        isDisabled={busy}
        isOptional
      />
      <TextArea
        label="Setting"
        placeholder="e.g. a foggy marsh"
        value={setting}
        onChange={(v) => setSetting(v)}
        rows={2}
        isDisabled={busy}
        isOptional
      />
      <TextArea
        label="Challenge"
        placeholder="e.g. everyone is faster than him"
        value={challenge}
        onChange={(v) => setChallenge(v)}
        rows={2}
        isDisabled={busy}
        isOptional
      />
      <TextArea
        label="Outcome"
        placeholder="e.g. he wins by being patient"
        value={outcome}
        onChange={(v) => setOutcome(v)}
        rows={2}
        isDisabled={busy}
        isOptional
      />
      <TextArea
        label="Teaching"
        placeholder="e.g. patience pays off"
        value={teaching}
        onChange={(v) => setTeaching(v)}
        rows={2}
        isDisabled={busy}
        isOptional
      />

      {/* Length selector */}
      <div>
        <p
          style={{
            margin: '0 0 0.5rem',
            fontSize: '0.875rem',
            fontWeight: 500,
            color: 'var(--astryx-color-text, #111827)',
          }}
        >
          Length
        </p>
        <SegmentedControl
          value={length}
          onChange={(v) => setLength(v as FableLength)}
          label="Fable length"
          layout="fill"
          isDisabled={busy}
        >
          <SegmentedControlItem value="short" label="Short" />
          <SegmentedControlItem value="medium" label="Medium" />
          <SegmentedControlItem value="long" label="Long" />
        </SegmentedControl>
      </div>

      {/* Model selector */}
      <ModelSelect value={modelId} onChange={setModelId} disabled={busy} />

      {/* Guardrail toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Switch
          label="Guardrail"
          value={guardrail}
          onChange={(checked) => setGuardrail(checked)}
          isDisabled={busy}
        />
        <Tooltip
          content="Multi-layer safety: filters unsafe requests and outputs so only wholesome children's fables are produced. Turn off to observe the model without safety."
          placement="above"
        >
          <span
            aria-label="Guardrail info"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '16px',
              height: '16px',
              borderRadius: '50%',
              border: '1px solid var(--astryx-color-border, #e5e7eb)',
              fontSize: '0.625rem',
              fontWeight: 700,
              color: 'var(--astryx-color-text-subtle, #6b7280)',
              cursor: 'default',
              flexShrink: 0,
            }}
          >
            i
          </span>
        </Tooltip>
      </div>

      {/* Generate button */}
      <Button
        label={busy ? 'Generating…' : 'Generate fable'}
        variant="primary"
        onClick={handleSubmit}
        isDisabled={busy}
      >
        {busy ? 'Generating…' : 'Generate fable'}
      </Button>
    </div>
  );
}
