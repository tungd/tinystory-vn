export interface ModelInfo {
  model_id: string;
  name: string;
  ollama: string;
  kind: string;
  desc: string;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const r = await fetch("/models");
  const raw: { id: string; name: string; ollama: string; kind: string; desc: string }[] = await r.json();
  return raw.map((m) => ({ model_id: m.id, name: m.name, ollama: m.ollama, kind: m.kind, desc: m.desc }));
}

// -- Eval types (canonical, single source of truth) --------------------------

/** The 4 scored axes returned by the judge. */
export interface EvalAxes {
  grammar: number;
  creativity: number;
  moral_clarity: number;
  prompt_adherence: number;
}

/** Full evaluation result: 4 axes + weighted overall + optional per-axis rationale. */
export interface EvalResult extends EvalAxes {
  overall: number;
  /** Judge's short justification per axis (evidence quoted from the story). May be absent. */
  rationale?: Partial<Record<keyof EvalAxes, string>>;
}

/** Runtime type guard - verifies all 5 fields are finite numbers. */
export function isEvalResult(x: unknown): x is EvalResult {
  if (typeof x !== 'object' || x === null) return false;
  const obj = x as Record<string, unknown>;
  return (
    Number.isFinite(obj.grammar) &&
    Number.isFinite(obj.creativity) &&
    Number.isFinite(obj.moral_clarity) &&
    Number.isFinite(obj.prompt_adherence) &&
    Number.isFinite(obj.overall)
  );
}

export async function evaluate(story: string, prompt: string, judge_model_id?: string) {
  const r = await fetch("/evaluate", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({story, prompt, judge_model_id})}); return r.json();
}

export interface GenMeta {
  model_id: string;
  model_name: string;
  kind?: string;
  temperature: number;
  top_p: number;
  repetition_penalty: number;
  generation_mode?: "raw" | "postprocess" | "repair";
  enhancement?: {
    actions?: string[];
    initial_validation?: {
      fixable_reasons?: string[];
      severe_reasons?: string[];
    };
    final_validation?: {
      fixable_reasons?: string[];
      severe_reasons?: string[];
    };
  } | null;
  num_predict: number;
  seed?: number;
  prompt_sent: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  tokens_per_sec: number;
}

export type SSEEvent =
  | { type: "step"; stage: "prepare" | "model" | "input_check" | "generating" | "output_check"; status: "running" | "ok" | "blocked"; detail: string }
  | { type: "token"; text: string }
  | { type: "done"; status: "success" | "refused"; story?: string; reason?: string; meta?: GenMeta }
  | { type: "error"; reason: string };

export async function streamFable(payload: unknown, onEvent: (e: SSEEvent)=>void) {
  const res = await fetch("/generate/stream", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
  const reader = res.body!.getReader(); const dec = new TextDecoder(); let buf = "";
  while (true) { const {done,value} = await reader.read(); if (done) break;
    buf += dec.decode(value,{stream:true}); const parts = buf.split("\n\n"); buf = parts.pop()!;
    for (const p of parts) { const t=p.trim(); if (t.startsWith("data:")) onEvent(JSON.parse(t.slice(5).trim()) as SSEEvent); } }
}

export async function fetchResults(): Promise<{ available: boolean; data: unknown }> {
  const r = await fetch("/results");
  return r.json() as Promise<{ available: boolean; data: unknown }>;
}
