export async function fetchModels() { const r = await fetch("/models"); return r.json(); }
export async function evaluate(story: string, prompt: string, judge_model_id?: string) {
  const r = await fetch("/evaluate", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({story, prompt, judge_model_id})}); return r.json();
}
export async function streamFable(payload: unknown, onEvent: (e: unknown)=>void) {
  const res = await fetch("/generate/stream", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
  const reader = res.body!.getReader(); const dec = new TextDecoder(); let buf = "";
  while (true) { const {done,value} = await reader.read(); if (done) break;
    buf += dec.decode(value,{stream:true}); const parts = buf.split("\n\n"); buf = parts.pop()!;
    for (const p of parts) { const t=p.trim(); if (t.startsWith("data:")) onEvent(JSON.parse(t.slice(5).trim())); } }
}
