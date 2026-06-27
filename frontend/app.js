const form = document.getElementById("fable-form");
const submitBtn = document.getElementById("submit-btn");
const logCard = document.getElementById("log-card");
const logList = document.getElementById("log-list");
const states = {
  empty: document.getElementById("empty"),
  loading: document.getElementById("loading"),
  error: document.getElementById("error"),
  refused: document.getElementById("refused"),
  story: document.getElementById("story"),
};

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Render markdown an toàn: escape HTML TRƯỚC, rồi mới áp một tập markdown giới hạn.
function renderMarkdown(md) {
  const esc = escapeHtml(md);
  const inline = (t) =>
    t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
     .replace(/\*([^*]+)\*/g, "<em>$1</em>");
  const out = [];
  let listType = null; // "ul" | "ol" | null
  const closeList = () => { if (listType) { out.push(`</${listType}>`); listType = null; } };
  for (const raw of esc.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) { closeList(); continue; }
    let m;
    if ((m = line.match(/^(#{1,3})\s+(.*)$/))) {
      closeList();
      const lvl = m[1].length;
      out.push(`<h${lvl}>${inline(m[2])}</h${lvl}>`);
    } else if ((m = line.match(/^[-*]\s+(.*)$/))) {
      if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\d+\.\s+(.*)$/))) {
      if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return out.join("");
}

// Hiển thị truyện hoàn chỉnh ở chế độ markdown.
function showStory(md) {
  for (const key of Object.keys(states)) states[key].hidden = key !== "story";
  states.story.innerHTML = renderMarkdown(md);  // an toàn: nội dung đã được escape trong renderMarkdown
}

function showState(name, text) {
  for (const key of Object.keys(states)) {
    states[key].hidden = key !== name;
    if (key === name && text !== undefined) states[key].textContent = text;
  }
}

function clearLog() { logList.innerHTML = ""; logCard.hidden = false; }

const STAGE_LABEL = {
  input_check: "Kiểm tra đầu vào (Lớp 1)",
  generating: "Sinh truyện (Lớp 2-3)",
  output_check: "Kiểm tra đầu ra (Lớp 4)",
};
const STATUS_ICON = { running: "⏳", ok: "✅", blocked: "⛔" };

function nowStr() {
  const d = new Date();
  return d.toLocaleTimeString("vi-VN");
}

function addLog(ev) {
  const li = document.createElement("li");
  li.className = "log-entry " + ev.status;
  const label = STAGE_LABEL[ev.stage] || ev.stage;

  const iconSpan = document.createElement("span");
  iconSpan.className = "icon";
  iconSpan.textContent = STATUS_ICON[ev.status] || "•";

  const bodySpan = document.createElement("span");
  bodySpan.className = "body";

  const strong = document.createElement("strong");
  strong.textContent = label;

  const br = document.createElement("br");

  const detailText = document.createTextNode(ev.detail || "");

  const tsDiv = document.createElement("div");
  tsDiv.className = "ts";
  tsDiv.textContent = nowStr();

  bodySpan.appendChild(strong);
  bodySpan.appendChild(br);
  bodySpan.appendChild(detailText);
  bodySpan.appendChild(tsDiv);

  li.appendChild(iconSpan);
  li.appendChild(bodySpan);
  logList.appendChild(li);
}

async function generate(payload) {
  showState("loading");
  clearLog();
  let story = "";
  let streaming = false;
  try {
    const res = await fetch("/generate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok || !res.body) { showState("error", "Không kết nối được máy chủ."); return; }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5).trim());
        if (ev.type === "step") {
          addLog(ev);
        } else if (ev.type === "token") {
          if (!streaming) { showState("story", ""); streaming = true; }
          story += ev.text;
          states.story.textContent = story;
        } else if (ev.type === "done") {
          if (ev.status === "success") showStory(ev.story);
          else if (ev.status === "refused") showState("refused", ev.reason);
        } else if (ev.type === "error") {
          showState("error", ev.reason || "Đã có lỗi xảy ra.");
        }
      }
    }
  } catch (err) {
    showState("error", "Không kết nối được máy chủ.");
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  const length = document.querySelector('input[name="length"]:checked').value;
  await generate({
    topic: document.getElementById("topic").value,
    moral: document.getElementById("moral").value,
    age_range: document.getElementById("age_range").value,
    length,
    model_choice: document.getElementById("model_choice").value,
    guardrail_enabled: document.getElementById("guardrail").checked,
  });
  submitBtn.disabled = false;
});

const modelSelect = document.getElementById("model_choice");
const modelInfoEl = document.getElementById("model-info");
let MODEL_INFO = null;

function updateModelInfo() {
  if (!MODEL_INFO) return;
  const info = MODEL_INFO[modelSelect.value];
  if (info) modelInfoEl.textContent = `${info.name} — ${info.desc}`;
}

async function loadModelInfo() {
  try {
    const res = await fetch("/models");
    if (res.ok) { MODEL_INFO = await res.json(); updateModelInfo(); }
  } catch (e) { /* bỏ qua: chú thích chỉ là phụ trợ */ }
}

modelSelect.addEventListener("change", updateModelInfo);
loadModelInfo();
