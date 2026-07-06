export async function fetchModels() { const r = await fetch("/models"); return r.json(); }
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
  num_predict: number;
  seed?: number;
  prompt_sent: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  tokens_per_sec: number;
}

export type SSEEvent =
  | { type: "step"; stage: "input_check" | "generating" | "output_check"; status: "running" | "ok" | "blocked"; detail: string }
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
