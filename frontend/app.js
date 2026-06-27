const form = document.getElementById("fable-form");
const els = {
  loading: document.getElementById("loading"),
  empty: document.getElementById("empty"),
  error: document.getElementById("error"),
  refused: document.getElementById("refused"),
  story: document.getElementById("story"),
};

function show(state, text) {
  for (const key of Object.keys(els)) {
    els[key].hidden = key !== state;
    if (key === state && text !== undefined) els[key].textContent = text;
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  show("loading");
  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic: document.getElementById("topic").value,
        moral: document.getElementById("moral").value,
        age_range: document.getElementById("age_range").value,
        guardrail_enabled: document.getElementById("guardrail").checked,
        model_choice: document.getElementById("model_choice").value,
      }),
    });
    const data = await res.json();
    if (data.status === "success") show("story", data.story);
    else if (data.status === "refused") show("refused", data.reason);
    else show("error", data.reason || "Đã có lỗi xảy ra.");
  } catch (err) {
    show("error", "Không kết nối được máy chủ.");
  }
});
