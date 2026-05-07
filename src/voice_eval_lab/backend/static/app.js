/**
 * voice-eval-lab frontend — vanilla JS, no framework.
 *
 * UX states:
 *   idle       — page load, no session
 *   no-livekit — POST /sessions returned livekit_token: null
 *   connecting — LiveKit SDK loaded, connecting to room
 *   connected  — room joined, audio waveform + SSE transcript active
 *   error      — unexpected failure at any stage
 *   ended      — user clicked "End session"
 */

"use strict";

const LIVEKIT_CDN =
  "https://unpkg.com/livekit-client@2.x/dist/livekit-client.umd.js";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $startBtn         = document.getElementById("start-session");
const $endBtn           = document.getElementById("end-session");
const $noBackendBanner  = document.getElementById("no-backend-banner");
const $errorBanner      = document.getElementById("error-banner");
const $errorMsg         = document.getElementById("error-message");
const $statusEl         = document.getElementById("connection-status");
const $waveform         = document.getElementById("waveform");
const $transcriptArea   = document.getElementById("transcript");
const $transcriptBox    = document.getElementById("transcript-container");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _sessionId   = null;
let _room        = null;      // LiveKit Room instance
let _eventSource = null;      // SSE connection
let _animFrame   = null;      // requestAnimationFrame handle for waveform
let _analyser    = null;      // Web Audio AnalyserNode
let _dataArr     = null;      // Uint8Array for waveform data

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function setStatus(label, cls) {
  $statusEl.textContent = label;
  $statusEl.className = "status-" + cls;
}

function showError(msg) {
  $errorBanner.hidden = false;
  $errorMsg.textContent = msg;
  setStatus("Error", "error");
}

function clearError() {
  $errorBanner.hidden = true;
  $errorMsg.textContent = "";
}

function appendTurn(role, text) {
  const div = document.createElement("div");
  div.className = "turn turn-" + role;
  const roleSpan = document.createElement("span");
  roleSpan.className = "turn-role";
  roleSpan.textContent = role === "user" ? "User" : "Agent";
  const textNode = document.createTextNode(text);
  div.appendChild(roleSpan);
  div.appendChild(textNode);
  $transcriptArea.appendChild(div);
  $transcriptArea.scrollTop = $transcriptArea.scrollHeight;
}

// ---------------------------------------------------------------------------
// Waveform via Web Audio API
// ---------------------------------------------------------------------------
function startWaveform(stream) {
  try {
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    _analyser = ctx.createAnalyser();
    _analyser.fftSize = 256;
    src.connect(_analyser);
    _dataArr = new Uint8Array(_analyser.frequencyBinCount);

    const canvas = $waveform;
    const ctx2d = canvas.getContext("2d");
    canvas.hidden = false;

    function draw() {
      _animFrame = requestAnimationFrame(draw);
      _analyser.getByteTimeDomainData(_dataArr);

      ctx2d.clearRect(0, 0, canvas.width, canvas.height);
      ctx2d.fillStyle = "#1a1d27";
      ctx2d.fillRect(0, 0, canvas.width, canvas.height);

      ctx2d.lineWidth = 2;
      ctx2d.strokeStyle = "#6366f1";
      ctx2d.beginPath();

      const sliceWidth = canvas.width / _dataArr.length;
      let x = 0;
      for (let i = 0; i < _dataArr.length; i++) {
        const v = _dataArr[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx2d.moveTo(x, y);
        else ctx2d.lineTo(x, y);
        x += sliceWidth;
      }
      ctx2d.lineTo(canvas.width, canvas.height / 2);
      ctx2d.stroke();
    }
    draw();
  } catch (err) {
    console.warn("Waveform unavailable:", err);
  }
}

function stopWaveform() {
  if (_animFrame !== null) {
    cancelAnimationFrame(_animFrame);
    _animFrame = null;
  }
  $waveform.hidden = true;
}

// ---------------------------------------------------------------------------
// SSE transcript stream
// ---------------------------------------------------------------------------
function startSSE(sessionId) {
  if (_eventSource) {
    _eventSource.close();
  }
  _eventSource = new EventSource(`/sessions/${sessionId}/events`);

  _eventSource.addEventListener("turn", (e) => {
    try {
      const data = JSON.parse(e.data);
      appendTurn(data.role, data.text);
    } catch {
      console.warn("Unparseable turn event:", e.data);
    }
  });

  _eventSource.onerror = () => {
    // SSE errors are non-fatal; the connection will retry automatically.
    console.warn("SSE connection error — will retry");
  };
}

function stopSSE() {
  if (_eventSource) {
    _eventSource.close();
    _eventSource = null;
  }
}

// ---------------------------------------------------------------------------
// LiveKit connection
// ---------------------------------------------------------------------------
async function loadLiveKitSDK() {
  return new Promise((resolve, reject) => {
    if (window.LivekitClient) {
      resolve(window.LivekitClient);
      return;
    }
    const script = document.createElement("script");
    script.src = LIVEKIT_CDN;
    script.onload = () => resolve(window.LivekitClient);
    script.onerror = () => reject(new Error("Failed to load LiveKit JS SDK from CDN"));
    document.head.appendChild(script);
  });
}

async function connectToLiveKit(token, wsUrl) {
  const LK = await loadLiveKitSDK();
  const room = new LK.Room();
  _room = room;

  room.on(LK.RoomEvent.Disconnected, () => {
    setStatus("Disconnected", "ended");
    stopWaveform();
    stopSSE();
  });

  room.on(LK.RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === LK.Track.Kind.Audio) {
      const mediaStream = new MediaStream([track.mediaStreamTrack]);
      startWaveform(mediaStream);
    }
  });

  await room.connect(wsUrl, token);
  setStatus("Connected", "connected");

  // Start local mic (publish) — request permission but don't fail hard if denied
  try {
    await room.localParticipant.setMicrophoneEnabled(true);
    const micTrack = room.localParticipant.getTrackPublication(LK.Track.Source.Microphone);
    if (micTrack && micTrack.track) {
      const stream = new MediaStream([micTrack.track.mediaStreamTrack]);
      startWaveform(stream);
    }
  } catch (micErr) {
    console.warn("Microphone unavailable:", micErr);
  }
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------
async function startSession() {
  clearError();
  $startBtn.disabled = true;
  setStatus("Connecting…", "connecting");

  try {
    const resp = await fetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: "browser-user" }),
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`POST /sessions ${resp.status}: ${txt}`);
    }

    const body = await resp.json();
    _sessionId = body.session_id;

    if (!body.livekit_token) {
      // No LiveKit credentials configured on the server.
      $noBackendBanner.hidden = false;
      setStatus("No LiveKit token — demo mode", "connecting");
      $endBtn.disabled = false;
      // Still show transcript so SSE mock works.
      $transcriptBox.hidden = false;
      startSSE(_sessionId);
      return;
    }

    // Credentials present — connect via LiveKit JS SDK.
    $noBackendBanner.hidden = true;
    const wsUrl = body.livekit_url || "wss://localhost:7880";
    await connectToLiveKit(body.livekit_token, wsUrl);

    $transcriptBox.hidden = false;
    startSSE(_sessionId);
    $endBtn.disabled = false;
  } catch (err) {
    showError(err.message || String(err));
    $startBtn.disabled = false;
  }
}

async function endSession() {
  $endBtn.disabled = true;
  stopSSE();
  stopWaveform();

  if (_room) {
    try { await _room.disconnect(); } catch { /* ignore */ }
    _room = null;
  }

  if (_sessionId) {
    try {
      await fetch(`/sessions/${_sessionId}/end`, { method: "POST" });
    } catch (err) {
      console.warn("End session request failed:", err);
    }
    _sessionId = null;
  }

  setStatus("Session ended", "ended");
  $startBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------
$startBtn.addEventListener("click", startSession);
$endBtn.addEventListener("click", endSession);
