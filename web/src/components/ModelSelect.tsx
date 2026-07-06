import { useEffect, useState } from 'react';
import { Selector } from '@astryxdesign/core/Selector';
import { fetchModels } from '../api';
import type { ModelInfo } from '../api';

export interface ModelSelectProps {
  value: string;
  onChange: (model_id: string) => void;
  disabled?: boolean;
}

export function ModelSelect({ value, onChange, disabled = false }: ModelSelectProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels()
      .then((data: ModelInfo[]) => {
        setModels(data);
        if (data.length > 0 && !value) {
          onChange(data[0].model_id);
        }
      })
      .catch(() => {
        // Graceful fallback: show empty list, no crash
      })
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const options = models.map((m) => ({
    value: m.model_id,
    label: m.name,
    ...(m.kind || m.desc
      ? { description: [m.kind, m.desc].filter(Boolean).join(' - ') }
      : {}),
  }));

  return (
    <Selector
      label="Model"
      options={options}
      value={value}
      onChange={onChange}
      isLoading={loading}
      isDisabled={disabled}
      placeholder="Select a model…"
    />
  );
}
