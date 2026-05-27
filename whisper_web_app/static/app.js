const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#fileInput");
const fileName = document.querySelector("#fileName");
const model = document.querySelector("#model");
const organizerEnabled = document.querySelector("#organizerEnabled");
const organizerTemplate = document.querySelector("#organizerTemplate");
const organizerProvider = document.querySelector("#organizerProvider");
const organizerModel = document.querySelector("#organizerModel");
const organizerContext = document.querySelector("#organizerContext");
const statusEl = document.querySelector("#status");
const progress = document.querySelector("#progress");
const progressBar = document.querySelector(".bar");
const progressText = document.querySelector("#progressText");
const result = document.querySelector("#result");
const meta = document.querySelector("#meta");
const copyBtn = document.querySelector("#copyBtn");
const downloadBtn = document.querySelector("#downloadBtn");

let lastText = "";
let lastFileStem = "transcript";
let pollTimer = null;

function setStatus(text, type = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${type}`.trim();
}

function setProgress(percent, text) {
  progress.hidden = false;
  const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
  progressBar.style.width = `${safePercent}%`;
  progressText.textContent = `${safePercent}% · ${text}`;
}

function setBusy(isBusy) {
  fileInput.disabled = isBusy;
  model.disabled = isBusy;
  organizerEnabled.disabled = isBusy;
  organizerTemplate.disabled = isBusy;
  organizerProvider.disabled = isBusy;
  organizerModel.disabled = isBusy;
  organizerContext.disabled = isBusy;
  if (isBusy) {
    setStatus("处理中", "active");
  }
}

function formatSeconds(seconds) {
  const total = Math.round(seconds || 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) {
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function safeStem(name) {
  return (name || "transcript").replace(/\.[^.]+$/, "").replace(/[^\w\u4e00-\u9fa5-]+/g, "_");
}

function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const body = new FormData();
    body.append("audio", file);
    body.append("model", model.value);
    body.append("organizer_enabled", organizerEnabled.checked ? "1" : "0");
    body.append("organizer_template", organizerTemplate.value);
    body.append("organizer_provider", organizerProvider.value);
    body.append("organizer_model", organizerModel.value.trim());
    body.append("organizer_context", organizerContext.value);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/transcribe");
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        const percent = (event.loaded / event.total) * 8;
        setProgress(percent, `正在上传：${file.name}`);
      }
    });
    xhr.addEventListener("load", () => {
      try {
        const data = JSON.parse(xhr.responseText || "{}");
        if (xhr.status >= 400 || data.error) {
          reject(new Error(data.error || "上传失败"));
          return;
        }
        resolve(data.job_id);
      } catch (error) {
        reject(error);
      }
    });
    xhr.addEventListener("error", () => reject(new Error("上传失败")));
    xhr.send(body);
  });
}

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || "读取进度失败");
  }

  const percent = (data.progress || 0) * 100;
  setProgress(percent, data.message || "正在处理");

  if (data.partial_text) {
    result.value = data.partial_text;
    result.scrollTop = result.scrollHeight;
  }
  if (data.duration) {
    meta.textContent = `${data.model} | 已到 ${formatSeconds(data.current_time)} / ${formatSeconds(data.duration)}`;
  } else if (data.status === "organizing") {
    meta.textContent = `${data.model} | 整理模型处理中`;
  }

  if (data.status === "done") {
    lastText = data.result?.text || data.partial_text || "";
    result.value = lastText;
    const organizerNote = data.result?.organized_text
      ? ` | 已整理：${data.result?.organizer_provider}/${data.result?.organizer_model}`
      : data.result?.organizer_error
        ? ` | 整理失败：${data.result.organizer_error}`
        : "";
    meta.textContent = `${data.model} | 音频 ${formatSeconds(data.result?.duration)} | 用时 ${formatSeconds(data.result?.elapsed)}${organizerNote}`;
    copyBtn.disabled = !lastText;
    downloadBtn.disabled = !lastText;
    setStatus("已完成", "active");
    setBusy(false);
    return;
  }

  if (data.status === "error") {
    throw new Error(data.message || "转写失败");
  }

  pollTimer = window.setTimeout(() => {
    pollJob(jobId).catch(handleError);
  }, 1000);
}

function handleError(error) {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
  setBusy(false);
  setStatus("出错", "error");
  progress.hidden = true;
  result.value = `转写失败：${error.message}`;
}

async function transcribe(file) {
  if (!file) return;
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
  fileName.textContent = file.name;
  lastFileStem = safeStem(file.name);
  lastText = "";
  result.value = "";
  meta.textContent = "";
  copyBtn.disabled = true;
  downloadBtn.disabled = true;
  setBusy(true);
  setProgress(0, "准备上传");

  try {
    const jobId = await uploadFile(file);
    setProgress(8, "已上传，等待模型开始识别");
    await pollJob(jobId);
  } catch (error) {
    handleError(error);
  }
}

dropzone.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", (event) => {
  if (!dropzone.contains(event.relatedTarget)) {
    dropzone.classList.remove("dragover");
  }
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files?.[0];
  transcribe(file);
});

fileInput.addEventListener("change", () => {
  transcribe(fileInput.files?.[0]);
});

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(lastText);
  setStatus("已复制", "active");
});

downloadBtn.addEventListener("click", () => {
  const blob = new Blob([lastText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${lastFileStem}_dialogue.txt`;
  link.click();
  URL.revokeObjectURL(url);
});

organizerProvider.addEventListener("change", () => {
  if (organizerProvider.value === "ollama") {
    organizerModel.value = "qwen2.5:7b-instruct";
  } else {
    organizerModel.value = "gpt-4o-mini";
  }
});
