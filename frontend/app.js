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
          if (ev.status === "success") showState("story", ev.story);
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
