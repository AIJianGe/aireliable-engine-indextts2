/* ============================================================
   IndexTTS2 TTS WebUI 逻辑（原生 JS，直连引擎 API）
   三区联动：设置区（模式/变体/加载） ↔ 参数区（音色参考/模式专属/文本/高级） ↔ 结果区（SSE + 播放器）
   ============================================================ */
"use strict";

const API = "";  // 同源直连（引擎自身挂载 /ui）

// ---------- 状态 ----------
const state = {
  mode: "clone",
  speaker: null,    // { name, path, previewUrl }（path 为引擎侧绝对路径）
  emotion: null,    // { name, path, localUrl }（emo_clone 模式）
  voices: [],       // 平台音色库列表（引擎 /voices 代理）
  variants: [],     // capabilities.variants
  generating: false,
  loading: false,
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

// SSE 阶段 → 中文（stage 定义见 progress_reporter.py ProgressStage）
const STAGE_TEXT = {
  "indextts2.start": "开始推理",
  "indextts2.model_ready": "模型就绪",
  "indextts2.text_processing": "文本处理",
  "indextts2.generating": "生成音频",
  "indextts2.saving": "保存产物",
  "completed": "完成",
};

// 模式提示（对应 Gradio ui.py 单 Tab 化设计）
const MODE_HINTS = {
  clone: "克隆（clone）：仅音色参考音频，零样本音色复刻",
  emo_clone: "情感克隆（emo_clone）：音色参考 + 情感参考音频 + 情感权重",
  emo_vector: "情感向量（emo_vector）：音色参考 + 8 维情感向量控制",
};

// 8 维情感标签（顺序与 IndexTTS2.infer() 一致，schemas.py EMOTION_LABELS）
const EMOTION_LABELS = [
  "开心 (happy)", "愤怒 (angry)", "悲伤 (sad)", "害怕 (afraid)",
  "厌恶 (disgusted)", "忧郁 (melanchoric)", "惊讶 (surprised)", "平静 (calm)",
];

// ============================================================
// 初始化
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  buildEmoGrid();
  bindModeSeg();
  bindSliders();
  bindVoiceSelect();
  bindUploads();
  bindButtons();
  refreshStatus();
  refreshCapabilities();
  refreshVoices();
  setInterval(refreshStatus, 15000);  // 状态轮询（空闲时）
});

// ============================================================
// 设置区：模式切换
// ============================================================
function bindModeSeg() {
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      $("mode-hint").textContent = MODE_HINTS[state.mode];
      // 模式专属区块联动显隐（对应 Gradio _on_mode_change）
      $("emo-clone-field").style.display = state.mode === "emo_clone" ? "" : "none";
      $("emo-vector-field").style.display = state.mode === "emo_vector" ? "" : "none";
      $("footer-mode").textContent = `模式：${btn.textContent}`;
    });
  });
  $("footer-mode").textContent = "模式：克隆";
}

// ============================================================
// 参数区
// ============================================================
function bindSliders() {
  const pairs = [
    ["emotion-alpha", "emotion-alpha-val", (v) => parseFloat(v).toFixed(2)],
    ["top-p", "top-p-val", (v) => parseFloat(v).toFixed(2)],
    ["top-k", "top-k-val", (v) => v],
    ["temperature", "temperature-val", (v) => parseFloat(v).toFixed(2)],
    ["rep-penalty", "rep-penalty-val", (v) => parseFloat(v).toFixed(1)],
    ["max-mel", "max-mel-val", (v) => v],
    ["max-text", "max-text-val", (v) => v],
  ];
  for (const [id, valId, fmt] of pairs) {
    $(id).addEventListener("input", (e) => { $(valId).textContent = fmt(e.target.value); });
  }
}

// 8 维情感滑块组（动态生成，2 列网格）
function buildEmoGrid() {
  const grid = $("emo-grid");
  grid.innerHTML = "";
  EMOTION_LABELS.forEach((label, i) => {
    const item = document.createElement("div");
    item.className = "emo-item";
    item.innerHTML = `
      <label>${label}<span class="val" id="emo-val-${i}">0.00</span></label>
      <input type="range" id="emo-${i}" min="0" max="1" step="0.05" value="0">`;
    grid.appendChild(item);
    $(`emo-${i}`).addEventListener("input", (e) => {
      $(`emo-val-${i}`).textContent = parseFloat(e.target.value).toFixed(2);
    });
  });
}

// ---------- 音色库选择（引擎 /voices 代理 → gateway /v1/voices） ----------
function bindVoiceSelect() {
  $("btn-refresh-voices").addEventListener("click", refreshVoices);
  $("voice-select").addEventListener("change", (e) => {
    const id = e.target.value;
    if (!id) { clearSpeakerPreview(); return; }
    const card = state.voices.find((v) => v.id === id);
    if (!card || !card.path) {
      clearSpeakerPreview();
      showError("音色本地文件不可用（请到平台音色库管理页检查）");
      return;
    }
    // 选中平台音色：speaker.path = 引擎侧绝对路径，预览用 gateway audio_url
    state.speaker = { name: card.name, path: card.path, previewUrl: card.audio_url || "" };
    renderVoicePreview(card);
    clearUploadChip("speaker");
  });
}

function renderVoicePreview(card) {
  const box = $("voice-preview");
  const meta = [
    card.name ? `「${card.name}」` : "",
    card.description ? ` ${card.description}` : "",
    card.duration ? ` · ${card.duration.toFixed(1)}s` : "",
  ].join("");
  box.innerHTML = `
    <div class="audio-chip">
      <span class="fc-name" title="${card.name || ""}">🎤 平台音色${meta}</span>
    </div>
    ${card.audio_url ? `<audio src="${card.audio_url}" controls preload="none"></audio>` : ""}`;
}

function clearSpeakerPreview() {
  $("voice-preview").innerHTML = "";
  if (state.speaker?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(state.speaker.previewUrl);
  state.speaker = null;
}

// ---------- 参考音频上传（音色 / 情感） ----------
function bindUploads() {
  bindAudioDropzone("speaker-dropzone", "speaker-file-input", uploadSpeaker);
  bindAudioDropzone("emotion-dropzone", "emotion-file-input", uploadEmotion);
}

function bindAudioDropzone(dzId, fiId, handler) {
  const dz = $(dzId);
  const fi = $(fiId);
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    handler(e.dataTransfer.files[0]);
  });
  fi.addEventListener("change", () => handler(fi.files[0]));
}

async function uploadSpeaker(file) {
  if (!file) return;
  if (state.speaker) { showError("已有音色参考，请先移除后再上传"); return; }
  // 本地预览 + 上传拿引擎侧 path
  const localUrl = URL.createObjectURL(file);
  state.speaker = { name: file.name, path: "", previewUrl: localUrl };
  renderUploadChip("speaker", file.name, "上传中…");
  try {
    const data = await uploadFile(file);
    state.speaker.path = data.path;
    renderUploadChip("speaker", file.name, fmtSize(data.size ?? file.size));
    // 上传自定义音频时清除平台音色选择
    $("voice-select").value = "";
    $("voice-preview").innerHTML = "";
  } catch (err) {
    removeUpload("speaker");
    showError(`参考音频上传失败：${err.message}`);
  }
}

async function uploadEmotion(file) {
  if (!file) return;
  if (state.emotion) { showError("已有情感参考音频，请先移除后再上传"); return; }
  const localUrl = URL.createObjectURL(file);
  state.emotion = { name: file.name, path: "", localUrl };
  renderUploadChip("emotion", file.name, "上传中…");
  try {
    const data = await uploadFile(file);
    state.emotion.path = data.path;
    renderUploadChip("emotion", file.name, fmtSize(data.size ?? file.size));
  } catch (err) {
    removeUpload("emotion");
    showError(`情感参考音频上传失败：${err.message}`);
  }
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/upload`, { method: "POST", body: fd });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

function renderUploadChip(kind, name, sizeText) {
  const box = $(`${kind}-chip-box`);
  box.innerHTML = `
    <span class="audio-chip">
      <span class="fc-name" title="${name}">${kind === "speaker" ? "🎙" : "🎭"} ${name}</span>
      <span style="color:#1a7f37">${sizeText}</span>
      <button class="fc-rm" title="移除">×</button>
    </span>`;
  box.querySelector(".fc-rm").addEventListener("click", () => removeUpload(kind));
}

function removeUpload(kind) {
  if (kind === "speaker") {
    if (state.speaker?.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(state.speaker.previewUrl);
    state.speaker = null;
  } else {
    if (state.emotion?.localUrl) URL.revokeObjectURL(state.emotion.localUrl);
    state.emotion = null;
  }
  clearUploadChip(kind);
}

function clearUploadChip(kind) {
  $(`${kind}-chip-box`).innerHTML = "";
  $(`${kind}-file-input`).value = "";
}

function fmtSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return Math.round(bytes / 1024) + " KB";
}

// ============================================================
// 平台音色库（引擎 /voices 代理 → gateway GET /v1/voices）
// ============================================================
async function refreshVoices() {
  try {
    const r = await fetch(`${API}/voices`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const voices = await r.json();
    state.voices = Array.isArray(voices) ? voices : [];
    renderVoiceSelect();
  } catch {
    state.voices = [];
    renderVoiceSelect();
  }
}

function renderVoiceSelect() {
  const sel = $("voice-select");
  const cur = sel.value;
  sel.innerHTML = `<option value="">— 选择平台音色 —</option>`;
  for (const v of state.voices) {
    if (!v.id || !v.name) continue;
    const opt = document.createElement("option");
    opt.value = v.id;
    const dur = v.duration ? ` · ${v.duration.toFixed(1)}s` : "";
    opt.textContent = `${v.name}${dur}`;
    sel.appendChild(opt);
  }
  if (cur && [...sel.options].some((o) => o.value === cur)) sel.value = cur;
}

// ============================================================
// 设置区：变体 / 加载 / 卸载
// ============================================================
async function refreshCapabilities() {
  try {
    const r = await fetch(`${API}/capabilities`);
    const cap = await r.json();
    state.variants = cap.variants || [];
    renderVariantSelect();
  } catch {
    $("mode-hint").textContent = "capabilities 获取失败（引擎未启动？）";
  }
}

function renderVariantSelect() {
  const sel = $("variant-select");
  const cur = sel.value;
  sel.innerHTML = `<option value="">自动（默认变体）</option>`;
  for (const v of state.variants) {
    const opt = document.createElement("option");
    opt.value = v.name;
    const tag = v.available ? "" : "（未下载）";
    opt.textContent = `${v.name} · ${v.precision || "fp16"}${tag}`;
    if (!v.available) opt.disabled = true;
    sel.appendChild(opt);
  }
  if (cur && [...sel.options].some((o) => o.value === cur)) sel.value = cur;
}

function bindButtons() {
  $("btn-refresh-status").addEventListener("click", refreshStatus);
  $("btn-load").addEventListener("click", loadModel);
  $("btn-unload").addEventListener("click", unloadModel);
  $("btn-generate").addEventListener("click", generate);
}

// IndexTTS2 模型加载耗时较长（FP16 约需 6GB 显存）
async function loadModel() {
  if (state.loading) return;
  state.loading = true;
  $("btn-load").disabled = true;
  $("btn-load").textContent = "加载中…";
  try {
    const variant = $("variant-select").value || null;
    const r = await fetch(`${API}/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
  } catch (err) {
    showError(`模型加载失败：${err.message}`);
  } finally {
    state.loading = false;
    $("btn-load").disabled = false;
    $("btn-load").textContent = "加载模型";
    refreshStatus();
  }
}

async function unloadModel() {
  try {
    const r = await fetch(`${API}/unload`, { method: "POST" });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
  } catch (err) {
    showError(`模型卸载失败：${err.message}`);
  } finally {
    refreshStatus();
  }
}

// ============================================================
// 状态轮询
// ============================================================
async function refreshStatus() {
  if (state.generating) return;
  try {
    const r = await fetch(`${API}/status`);
    const st = await r.json();
    const badge = $("state-badge");
    badge.className = "badge " + (st.state || "");
    badge.textContent = {
      idle: "空闲", loading: "加载中", ready: "就绪",
      inferring: "推理中", error: "异常",
    }[st.state] || st.state;
    // IndexTTS2 状态扩展字段（extra.is_loaded / extra.loaded_variant / extra.vram_used）
    const dev = st.extra?.device_name || st.extra?.device || "—";
    const vram = st.extra?.vram_used ? `${st.extra.vram_used}MB` : (st.vram_used || "—");
    $("state-detail").textContent =
      `变体 ${st.extra?.loaded_variant || st.loaded_variant || "未加载"} · 显存 ${vram} · ${dev}`;
  } catch {
    $("state-badge").className = "badge";
    $("state-badge").textContent = "未连接";
    $("state-detail").textContent = "引擎不可达（127.0.0.1:7005）";
  }
}

// ============================================================
// 结果区：生成（SSE 流）
// ============================================================
async function generate() {
  if (state.generating) return;

  // 校验（对应 ui.py _generate 的模式校验）
  const text = $("text-input").value.trim();
  if (!state.speaker?.path) { showError("请先从平台音色库选择音色，或上传自定义参考音频"); return; }
  if (!text) { showError("请输入合成文本"); return; }
  if (state.mode === "emo_clone" && !state.emotion?.path) { showError("情感克隆模式需要上传情感参考音频"); return; }
  if (state.mode === "emo_vector") {
    const vec = getEmotionVector();
    if (vec.every((v) => v === 0)) { showError("情感向量模式需要至少一个非零情感值"); return; }
  }

  const inputs = {
    text,
    mode: state.mode,
    top_p: parseFloat($("top-p").value),
    top_k: parseInt($("top-k").value, 10),
    temperature: parseFloat($("temperature").value),
    repetition_penalty: parseFloat($("rep-penalty").value),
    max_mel_tokens: parseInt($("max-mel").value, 10),
    max_text_tokens_per_segment: parseInt($("max-text").value, 10),
    output_format: "wav",
    seed: parseInt($("seed").value, 10) || -1,
  };
  if (state.mode === "emo_clone") {
    inputs.emotion_alpha = parseFloat($("emotion-alpha").value);
  }
  if (state.mode === "emo_vector") {
    inputs.emotion_vector = getEmotionVector();
  }

  const audios = [{ path: state.speaker.path, role: "speaker" }];
  if (state.mode === "emo_clone" && state.emotion?.path) {
    audios.push({ path: state.emotion.path, role: "emotion" });
  }

  // 不携带 task_id：直连链路契约（引擎以 request.task_id 是否非空区分
  // gateway 转发/直连）。引擎会自动生成 task_id 上报任务中心全生命周期。
  const body = { inputs, audios };
  const variant = $("variant-select").value;
  if (variant) body.variant = variant;

  // UI 状态
  state.generating = true;
  $("btn-generate").disabled = true;
  $("btn-generate").textContent = "合成中…";
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = "0%";
  $("progress-text").textContent = "连接引擎…";
  $("error-box").style.display = "none";
  $("state-badge").className = "badge inferring";
  $("state-badge").textContent = "推理中";

  try {
    const resp = await fetch(`${API}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${resp.status}`);
    }

    // 解析 SSE 流（fetch ReadableStream 手动读 data: 行）
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finalResult = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();  // 末行可能不完整，留在缓冲
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        let frame;
        try { frame = JSON.parse(line.slice(5)); } catch { continue; }
        handleFrame(frame);
        if (frame.stage === "completed" && frame.result) finalResult = frame.result;
      }
    }

    if (finalResult) {
      showResult(finalResult);
    } else {
      throw new Error("SSE 流结束但未收到结果帧");
    }
  } catch (err) {
    showError(`合成失败：${err.message}`);
  } finally {
    state.generating = false;
    $("btn-generate").disabled = false;
    $("btn-generate").textContent = "生成语音";
    $("progress-wrap").style.display = "none";
    refreshStatus();
  }
}

function getEmotionVector() {
  return Array.from({ length: 8 }, (_, i) => parseFloat($(`emo-${i}`).value) || 0);
}

function handleFrame(f) {
  if (f.stage === "error") {
    throw new Error(f.error?.message || "引擎错误");
  }
  const pct = Math.round((f.progress || 0) * 100);
  $("progress-fill").style.width = `${pct}%`;
  const stageText = STAGE_TEXT[f.stage] || f.stage;
  let detail = f.message || "";
  if (f.stage === "indextts2.generating" && f.elapsed != null) {
    detail += `（${f.elapsed.toFixed(1)}s）`;
  }
  $("progress-text").textContent = `${stageText} ${pct}%${detail ? " · " + detail : ""}`;
}

function showResult(result) {
  const view = $("result-view");
  const t = Date.now();  // 防缓存
  // result: {audio_url, audio_path, duration, sample_rate, mode}
  const audioUrl = result.audio_url + "?t=" + t;
  view.innerHTML = `
    <div class="audio-result">
      <span class="media-label">合成语音（${result.mode} 模式）</span>
      <audio src="${audioUrl}" controls autoplay></audio>
    </div>`;

  const MODE_CN = { clone: "克隆", emo_clone: "情感克隆", emo_vector: "情感向量" };
  $("result-meta").innerHTML = `
    <span class="meta-chip">模式 ${MODE_CN[result.mode] || result.mode}</span>
    <span class="meta-chip">时长 ${result.duration?.toFixed(1) ?? "—"}s</span>
    <span class="meta-chip">采样率 ${result.sample_rate ?? 24000}Hz</span>
    <span class="meta-chip">seed ${$("seed").value}</span>
    <span class="meta-chip"><a href="${audioUrl}" download>下载音频</a></span>`;
}

// ---------- 错误提示（不自动消失，手动关闭） ----------
function showError(msg) {
  $("error-text").textContent = msg;
  $("error-box").style.display = "block";
}
