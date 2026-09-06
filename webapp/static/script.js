const CLASS_COLORS = {
  venous: "#7c3aed",
  diabetic: "#ea580c",
  pressure: "#0891b2",
  surgical: "#16a34a",
};

let currentFile = null;
let modelInfo = null;
let classInfoMap = {};
let history = [];

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const classifyBtn = document.getElementById("classifyBtn");
const classifyBtnText = document.getElementById("classifyBtnText");
const spinner = document.getElementById("spinner");
const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");
const samplesRow = document.getElementById("samplesRow");
const historyList = document.getElementById("historyList");
const camToggle = document.getElementById("camToggle");

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }

async function loadModelInfo() {
  const [infoRes, classInfoRes] = await Promise.all([
    fetch("/api/model_info").then(r => r.json()),
    fetch("/api/class_info").then(r => r.json()),
  ]);
  modelInfo = infoRes;
  classInfoMap = classInfoRes;

  const badgeRow = document.getElementById("badgeRow");
  badgeRow.innerHTML = "";
  const classesBadge = document.createElement("div");
  classesBadge.className = "badge";
  classesBadge.textContent = "Classes: " + modelInfo.classes.join(", ");
  badgeRow.appendChild(classesBadge);

  if (modelInfo.test_accuracy != null) {
    const accBadge = document.createElement("div");
    accBadge.className = "badge";
    accBadge.textContent = "Held-out test accuracy: " + fmtPct(modelInfo.test_accuracy);
    badgeRow.appendChild(accBadge);
  }
  const scopeBadge = document.createElement("div");
  scopeBadge.className = "badge";
  scopeBadge.textContent = "Arterial ulcers: not in scope";
  badgeRow.appendChild(scopeBadge);

  document.getElementById("disclaimerCard").innerHTML =
    "<strong>Important:</strong> " + modelInfo.disclaimer;

  samplesRow.innerHTML = "";
  modelInfo.classes.forEach(cls => {
    const img = document.createElement("img");
    img.src = "/api/sample_image/" + cls;
    img.title = "Try a sample " + cls + " image";
    img.onerror = () => img.remove();
    img.onclick = () => loadSampleAsFile(cls, img.src);
    samplesRow.appendChild(img);
  });
}

async function loadSampleAsFile(cls, url) {
  const resp = await fetch(url);
  const blob = await resp.blob();
  const file = new File([blob], cls + "_sample.jpg", { type: blob.type });
  setCurrentFile(file);
}

function setCurrentFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewWrap.style.display = "block";
  classifyBtn.disabled = false;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) setCurrentFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setCurrentFile(fileInput.files[0]);
});

classifyBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  classifyBtn.disabled = true;
  spinner.style.display = "inline-block";
  classifyBtnText.textContent = "Classifying...";

  try {
    const form = new FormData();
    form.append("image", currentFile);
    const resp = await fetch("/api/predict", { method: "POST", body: form });
    const data = await resp.json();
    if (data.error) {
      alert("Error: " + data.error);
    } else {
      renderResult(data);
      addToHistory(data);
    }
  } catch (err) {
    alert("Request failed: " + err);
  } finally {
    classifyBtn.disabled = false;
    spinner.style.display = "none";
    classifyBtnText.textContent = "Classify";
  }
});

camToggle.addEventListener("change", () => {
  const camFigure = document.getElementById("camFigure");
  if (camFigure) camFigure.style.display = camToggle.checked ? "block" : "none";
});

function renderResult(data) {
  resultEmpty.style.display = "none";
  resultContent.style.display = "block";

  const color = CLASS_COLORS[data.pred_class] || "#334155";
  const info = data.class_info || {};

  let html = "";
  html += `<div class="pred-header">
      <div class="pred-dot" style="background:${color}"></div>
      <div>
        <div class="pred-title">${data.pred_label}</div>
        <div class="pred-confidence">${fmtPct(data.top_confidence)} confidence</div>
      </div>
    </div>`;

  if (data.is_uncertain) {
    html += `<div class="uncertain-banner">
      The top two predictions are close (${fmtPct(data.top_confidence)} vs
      ${fmtPct(data.runner_up_confidence)}). Treat this prediction as uncertain and consider
      it inconclusive rather than a confident call.
    </div>`;
  }

  html += `<div class="bars">`;
  const sorted = Object.entries(data.confidences).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([cls, conf]) => {
    const c = CLASS_COLORS[cls] || "#334155";
    html += `<div class="bar-row">
      <div class="bar-label"><span>${cls}</span><span>${fmtPct(conf)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${(conf*100).toFixed(1)}%;background:${c}"></div></div>
    </div>`;
  });
  html += `</div>`;

  html += `<div class="images-row">
    <figure>
      <img src="${data.display_image}" alt="input">
      <figcaption>Input (resized)</figcaption>
    </figure>
    <figure id="camFigure" style="display:${camToggle.checked ? "block" : "none"}">
      <img src="${data.cam_image}" alt="grad-cam">
      <figcaption>Grad-CAM: what the model focused on</figcaption>
    </figure>
  </div>`;

  html += `<div class="info-box">
    <div><strong>What this is:</strong> ${info.summary || "No description available."}</div>
    <ul>${(info.typical_features || []).map(f => `<li>${f}</li>`).join("")}</ul>
  </div>`;

  html += `<div class="actions-row">
    <button class="btn btn-ghost btn-sm" onclick="downloadReport()">Download report (JSON)</button>
  </div>`;

  resultContent.innerHTML = html;
  resultContent._lastData = data;
}

function addToHistory(data) {
  const entry = {
    time: new Date().toLocaleTimeString(),
    pred_label: data.pred_label,
    pred_class: data.pred_class,
    top_confidence: data.top_confidence,
    display_image: data.display_image,
    full: data,
  };
  history.unshift(entry);
  renderHistory();
}

function renderHistory() {
  if (!history.length) {
    historyList.innerHTML = `<div class="history-empty">No classifications yet this session.</div>`;
    return;
  }
  historyList.innerHTML = "";
  history.forEach((entry, idx) => {
    const div = document.createElement("div");
    div.className = "history-item";
    div.innerHTML = `
      <img src="${entry.display_image}" alt="thumb">
      <div class="meta">
        <div class="name">${entry.pred_label}</div>
        <div class="sub">${fmtPct(entry.top_confidence)} &middot; ${entry.time}</div>
      </div>`;
    div.onclick = () => renderResult(entry.full);
    historyList.appendChild(div);
  });
}

function downloadReport() {
  const data = resultContent._lastData;
  if (!data) return;
  const report = {
    predicted_class: data.pred_class,
    predicted_label: data.pred_label,
    confidences: data.confidences,
    uncertain: data.is_uncertain,
    model_classes: modelInfo.classes,
    model_test_accuracy: modelInfo.test_accuracy,
    disclaimer: modelInfo.disclaimer,
    generated_at: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ulcer_classification_report_" + Date.now() + ".json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

loadModelInfo();
