/**
 * TwinBrain — 3D Brain Visualization
 * Three.js interactive viewer: 200 cortical regions at anatomical positions,
 * network-based activity coloring, orbit controls, timeline scrubber.
 */

// ── Constants ─────────────────────────────────────────────────────────────────
const WS_URL       = "ws://127.0.0.1:8765";
const N_REGIONS    = 200;
const RECONNECT_MS = 3000;
const SPHERE_R     = 4.0;   // base sphere radius (Three.js units ≈ mm)

// ── DOM refs ──────────────────────────────────────────────────────────────────
const canvas      = document.getElementById('brain-canvas');
const tooltip     = document.getElementById('tooltip');
const canvasWrap  = document.getElementById('canvas-wrap');
const slider      = document.getElementById('timeline-slider');
const frameLabel  = document.getElementById('frame-label');
const btnPlay     = document.getElementById('btn-play');

// ── State ─────────────────────────────────────────────────────────────────────
let ws         = null;
let connected  = false;
let demoTick   = 0;
let selected   = new Set();
let frameSeq   = [];      // [{activity:[…200 floats…]}, …]  — currently displayed
let framesFmri = [];      // fMRI-specific frames (from cache)
let framesEeg  = [];      // EEG-specific frames  (from cache)
let activeModality = 'fmri';  // 'fmri' | 'eeg'
let curFrame   = 0;
let isPlaying  = false;
let playTimer  = null;

// ── Guard: Three.js must be loaded ────────────────────────────────────────────
if (typeof THREE === 'undefined') {
  console.error('Three.js not loaded — falling back to demo-only mode');
}

// ── Three.js Scene ────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = false;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x07081a);

const camera = new THREE.PerspectiveCamera(48, 1, 1, 2000);
camera.position.set(0, 10, 320);
camera.lookAt(0, 0, 0);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dLight = new THREE.DirectionalLight(0xffffff, 0.85);
dLight.position.set(1, 2, 2);
scene.add(dLight);
const dLight2 = new THREE.DirectionalLight(0x4466ff, 0.30);
dLight2.position.set(-2, -1, -3);
scene.add(dLight2);

// ── Brain Region Positions ────────────────────────────────────────────────────
// Fibonacci-sphere distribution per hemisphere, stretched to brain proportions.
// X = left-right  (right +),  Y = inferior-superior,  Z = posterior-anterior
function makeBrainPositions() {
  const pos  = [];
  const dAz  = 2 * Math.PI * (2 - (1 + Math.sqrt(5)) / 2); // golden angle ≈ 2.40 rad

  for (let h = 0; h < 2; h++) {
    const sign = h === 0 ? -1 : 1;   // -1 = left hemisphere, +1 = right

    for (let i = 0; i < 100; i++) {
      const t   = (i + 0.5) / 100;
      // elevation: 1 (superior pole) → -0.85 (inferior, ~148°)
      const el  = 1.0 - 1.85 * t;
      const r   = Math.sqrt(Math.max(0, 1 - el * el));
      const az  = dAz * i;

      const ux  = r * Math.cos(az);   // lateral raw component
      const uz  = r * Math.sin(az);   // anterior-posterior component

      // Outer cortical surface: force lateral extent ≥ 15 %
      const lateralExtent = Math.abs(ux) * 0.85 + 0.15;

      // Temporal lobe bulge at mid-inferior height (el ≈ -0.22)
      const bulge = 9 * Math.exp(-((el + 0.22) ** 2) * 5);

      pos.push(new THREE.Vector3(
        sign * (lateralExtent * 55 + bulge + 9),  // X:  ±17 … ±73 mm
        el   * 63 - 4,                   // Y: −55 … +59 mm
        uz   * 76 - 8                    // Z: −84 … +68 mm
      ));
    }
  }
  return pos;
}

const BRAIN_POS = makeBrainPositions();

// ── Brain Outline (glass-brain effect) ───────────────────────────────────────
(function buildOutline() {
  const geo = new THREE.SphereGeometry(90, 20, 16);
  geo.applyMatrix4(new THREE.Matrix4().makeScale(0.88, 0.82, 0.97));

  // Semitransparent fill (inside faces so regions are visible through it)
  scene.add(new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
    color: 0x1a2244, transparent: true, opacity: 0.07,
    side: THREE.BackSide, depthWrite: false,
  })));

  // Wireframe edges
  scene.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: 0x2244bb, transparent: true, opacity: 0.12 })
  ));

  // Sagittal midline hint
  const div = new THREE.Mesh(
    new THREE.CircleGeometry(76, 32),
    new THREE.MeshBasicMaterial({ color: 0x3355cc, transparent: true, opacity: 0.045,
      side: THREE.DoubleSide, depthWrite: false })
  );
  div.rotation.y = Math.PI / 2;
  div.position.y = -4;
  scene.add(div);
})();

// ── Region Spheres ────────────────────────────────────────────────────────────
const regionGeo    = new THREE.SphereGeometry(SPHERE_R, 10, 8);
const regionMeshes = [];

for (let i = 0; i < N_REGIONS; i++) {
  const mat  = new THREE.MeshPhongMaterial({
    color:   new THREE.Color(0x0022cc),
    emissive:new THREE.Color(0x001066),
    shininess: 40,
  });
  const mesh = new THREE.Mesh(regionGeo, mat);
  mesh.position.copy(BRAIN_POS[i]);
  mesh.userData = { regionId: i, activity: 0.2, selected: false };
  scene.add(mesh);
  regionMeshes.push(mesh);
}

// ── Activity → Color ──────────────────────────────────────────────────────────
const COLOR_STOPS = [
  { t: 0.00, c: new THREE.Color(0x0020c8) },
  { t: 0.25, c: new THREE.Color(0x0080ff) },
  { t: 0.50, c: new THREE.Color(0x00e8cc) },
  { t: 0.75, c: new THREE.Color(0xffd800) },
  { t: 1.00, c: new THREE.Color(0xff2200) },
];
const _tc = new THREE.Color();
function activityColor(v) {
  for (let i = 1; i < COLOR_STOPS.length; i++) {
    if (v <= COLOR_STOPS[i].t) {
      const f = (v - COLOR_STOPS[i-1].t) / (COLOR_STOPS[i].t - COLOR_STOPS[i-1].t);
      return _tc.copy(COLOR_STOPS[i-1].c).lerp(COLOR_STOPS[i].c, f);
    }
  }
  return _tc.copy(COLOR_STOPS[COLOR_STOPS.length - 1].c);
}

function updateActivity(arr) {
  for (let i = 0; i < regionMeshes.length; i++) {
    const m  = regionMeshes[i];
    const v  = Math.max(0, Math.min(1, arr[i] ?? 0.2));
    m.userData.activity = v;
    const c  = activityColor(v);
    m.material.color.copy(c);
    m.material.emissive.setRGB(c.r * 0.22, c.g * 0.22, c.b * 0.22);
    m.scale.setScalar(m.userData.selected ? 1.6 : 0.65 + v * 0.60);
  }
}

// ── Orbit Controls ────────────────────────────────────────────────────────────
const orbit = { theta: 0, phi: Math.PI * 0.44, radius: 320,
                dragging: false, px: 0, py: 0 };

canvas.addEventListener('mousedown', e => {
  if (e.button === 0) { orbit.dragging = true; orbit.px = e.clientX; orbit.py = e.clientY; }
});
document.addEventListener('mouseup', () => { orbit.dragging = false; });
document.addEventListener('mousemove', e => {
  if (!orbit.dragging) return;
  orbit.theta -= (e.clientX - orbit.px) * 0.007;
  orbit.phi    = Math.max(0.15, Math.min(Math.PI - 0.15,
                   orbit.phi - (e.clientY - orbit.py) * 0.007));
  orbit.px = e.clientX; orbit.py = e.clientY;
});
canvas.addEventListener('wheel', e => {
  orbit.radius = Math.max(150, Math.min(700, orbit.radius + e.deltaY * 0.45));
}, { passive: true });

// Touch
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    orbit.dragging = true; orbit.px = e.touches[0].clientX; orbit.py = e.touches[0].clientY;
  }
}, { passive: true });
canvas.addEventListener('touchend', () => { orbit.dragging = false; }, { passive: true });
canvas.addEventListener('touchmove', e => {
  if (!orbit.dragging || e.touches.length !== 1) return;
  orbit.theta -= (e.touches[0].clientX - orbit.px) * 0.007;
  orbit.phi    = Math.max(0.15, Math.min(Math.PI - 0.15,
                   orbit.phi - (e.touches[0].clientY - orbit.py) * 0.007));
  orbit.px = e.touches[0].clientX; orbit.py = e.touches[0].clientY;
}, { passive: true });

function applyOrbit() {
  const { theta, phi, radius } = orbit;
  camera.position.set(
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.cos(theta)
  );
  camera.lookAt(0, -4, 0);
}

// ── Resize ────────────────────────────────────────────────────────────────────
function onResize() {
  const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);
onResize();

// ── Raycasting ────────────────────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
const _mouse    = new THREE.Vector2();

const NETWORK_NAMES = ['视觉','体感运动','背侧注意','腹侧注意','边缘','额顶','默认网络'];

function pickRegion(cx, cy, doClick) {
  const rect = canvas.getBoundingClientRect();
  _mouse.x =  ((cx - rect.left) / rect.width ) * 2 - 1;
  _mouse.y = -((cy - rect.top ) / rect.height) * 2 + 1;
  raycaster.setFromCamera(_mouse, camera);
  const hits = raycaster.intersectObjects(regionMeshes);
  if (!hits.length) return -1;
  const id = hits[0].object.userData.regionId;
  if (doClick) {
    const m = regionMeshes[id];
    m.userData.selected = !m.userData.selected;
    m.userData.selected ? selected.add(id) : selected.delete(id);
    document.getElementById('sel-count').textContent = selected.size;
  }
  return id;
}

canvas.addEventListener('click', e => {
  if (!orbit.dragging) pickRegion(e.clientX, e.clientY, true);
});

canvas.addEventListener('mousemove', e => {
  const id = pickRegion(e.clientX, e.clientY, false);
  if (id >= 0) {
    const m   = regionMeshes[id];
    const pct = (m.userData.activity * 100).toFixed(1);
    const net = NETWORK_NAMES[Math.min(Math.floor(id / (N_REGIONS / 7)), 6)];
    tooltip.style.display = 'block';
    tooltip.style.left    = `${e.offsetX + 14}px`;
    tooltip.style.top     = `${e.offsetY - 10}px`;
    tooltip.innerHTML =
      `<strong>区域 ${id + 1}</strong>&nbsp;&nbsp;${net}<br/>` +
      `活动: ${pct}%&nbsp;&nbsp;${id < 100 ? '左脑' : '右脑'}`;
    canvas.style.cursor = 'pointer';
  } else {
    tooltip.style.display = 'none';
    canvas.style.cursor   = orbit.dragging ? 'grabbing' : 'grab';
  }
});
canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

// ── Demo Mode (no backend) ────────────────────────────────────────────────────
const NET_FREQS  = [0.012, 0.020, 0.035, 0.028, 0.008, 0.025, 0.010];
const NET_PHASES = [0.0,   1.0,   2.1,   0.7,   3.2,   1.8,   0.4  ];
const NET_SIZE   = N_REGIONS / 7;

function demoUpdate() {
  demoTick += 0.05;
  const act = new Array(N_REGIONS);
  for (let i = 0; i < N_REGIONS; i++) {
    const n = Math.min(Math.floor(i / NET_SIZE), 6);
    const v = 0.36 + 0.22 * Math.sin(NET_FREQS[n] * demoTick + NET_PHASES[n] + i * 0.08)
                   + 0.04 * Math.sin(0.003 * demoTick + i * 0.25);
    act[i] = Math.max(0, Math.min(1, v));
  }
  updateActivity(act);
}

// ── Render Loop ───────────────────────────────────────────────────────────────
(function animate() {
  requestAnimationFrame(animate);
  if (!connected && frameSeq.length === 0) demoUpdate();
  applyOrbit();
  renderer.render(scene, camera);
})();

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {
  try { ws = new WebSocket(WS_URL); } catch { return setTimeout(connect, RECONNECT_MS); }

  ws.onopen = () => {
    connected = true;
    setStatus(true);
    ws.send(JSON.stringify({ type: "get_state" }));
  };

  ws.onmessage = ev => {
    try { handleMsg(JSON.parse(ev.data)); }
    catch (e) { console.warn("WS parse error:", e); }
  };

  ws.onerror = () => {};
  ws.onclose = () => { connected = false; setStatus(false); setTimeout(connect, RECONNECT_MS); };
}

function handleMsg(msg) {
  // Single-frame update (brain_state)
  if (msg.type === 'brain_state' && Array.isArray(msg.activity)) {
    if (frameSeq.length === 0) updateActivity(msg.activity);
    return;
  }
  // Multi-frame sequence (stimulation result or cache data)
  if ((msg.type === 'simulation_result' || msg.type === 'cache_loaded')
      && Array.isArray(msg.frames) && msg.frames.length > 0) {
    // Store per-modality frames when the server provides them
    if (Array.isArray(msg.frames_fmri) && msg.frames_fmri.length > 0) {
      framesFmri = msg.frames_fmri;
    } else {
      framesFmri = [];
    }
    if (Array.isArray(msg.frames_eeg) && msg.frames_eeg.length > 0) {
      framesEeg = msg.frames_eeg;
    } else {
      framesEeg = [];
    }
    // Show/configure modality toggle
    updateModalityToggle(msg.modalities || []);
    // Pick the active modality's frames (fall back to primary frames)
    const modFrames = _getModalityFrames();
    loadFrameSeq(modFrames.length > 0 ? modFrames : msg.frames, msg.path || null);
    return;
  }
  // EC inference result
  if (msg.type === 'ec_result') {
    handleECResult(msg);
    return;
  }
  // Server greeting
  if (msg.type === 'welcome') {
    document.getElementById('status-text').textContent =
      `已连接 (v${msg.version || '?'})`;
    return;
  }
  if (msg.type === 'error') {
    console.warn('Server error:', msg.message);
    const ecStatus = document.getElementById('ec-status');
    if (ecStatus && ecStatus.textContent === '推断中…') {
      ecStatus.textContent = `⚠ ${msg.message}`;
    }
  }
}

// ── Modality toggle (fMRI / EEG) ──────────────────────────────────────────────
function _getModalityFrames() {
  if (activeModality === 'eeg' && framesEeg.length > 0) return framesEeg;
  if (framesFmri.length > 0) return framesFmri;
  return framesEeg;
}

function updateModalityToggle(modalities) {
  const toggle   = document.getElementById('modality-toggle');
  const btnFmri  = document.getElementById('btn-mod-fmri');
  const btnEeg   = document.getElementById('btn-mod-eeg');
  const infoEl   = document.getElementById('modality-info');
  const hasFmri  = modalities.includes('fmri');
  const hasEeg   = modalities.includes('eeg');

  if (hasFmri || hasEeg) {
    toggle.style.display = '';
    btnFmri.disabled = !hasFmri;
    btnEeg.disabled  = !hasEeg;
    // Keep active modality if still available, else fall back to first available
    if (activeModality === 'eeg' && !hasEeg) activeModality = 'fmri';
    if (activeModality === 'fmri' && !hasFmri) activeModality = 'eeg';
    _applyModalityButtonStyles();
    const labels = [];
    if (hasFmri) labels.push(`fMRI ${framesFmri.length}帧`);
    if (hasEeg)  labels.push(`EEG ${framesEeg.length}帧`);
    infoEl.textContent = labels.join(' · ');
  } else {
    toggle.style.display = 'none';
  }
}

function _applyModalityButtonStyles() {
  const btnFmri = document.getElementById('btn-mod-fmri');
  const btnEeg  = document.getElementById('btn-mod-eeg');
  btnFmri.className = activeModality === 'fmri' ? 'btn-primary' : 'btn-secondary';
  btnEeg.className  = activeModality === 'eeg'  ? 'btn-primary' : 'btn-secondary';
}

function setStatus(ok) {
  document.getElementById('status-dot').className           = ok ? 'connected' : '';
  document.getElementById('status-text').textContent        = ok ? '已连接后端' : '演示模式（后端未连接）';
  document.getElementById('backend-status').textContent     = ok ? '已连接' : '未连接';
}

// ── Timeline ──────────────────────────────────────────────────────────────────
function loadFrameSeq(frames, label) {
  frameSeq  = frames;
  curFrame  = 0;
  slider.min   = 0;
  slider.max   = Math.max(0, frames.length - 1);
  slider.value = 0;
  updateFrameLabel();
  document.getElementById('frame-info').textContent =
    `${frames.length} 帧${label ? ' — ' + label.split(/[\\/]/).pop() : ''}`;
  if (frames[0]) updateActivity(frames[0].activity);
  playSeq();
}

function updateFrameLabel() {
  frameLabel.textContent = frameSeq.length > 0
    ? `${curFrame + 1} / ${frameSeq.length}` : '—';
}

function playSeq() {
  clearInterval(playTimer);
  isPlaying = true;
  btnPlay.textContent = '⏸';
  playTimer = setInterval(() => {
    if (!frameSeq.length) return;
    curFrame = (curFrame + 1) % frameSeq.length;
    slider.value = curFrame;
    updateFrameLabel();
    updateActivity(frameSeq[curFrame].activity);
  }, 100);   // 10 fps
}

function pauseSeq() {
  clearInterval(playTimer);
  isPlaying = false;
  btnPlay.textContent = '▶';
}

btnPlay.addEventListener('click', () => isPlaying ? pauseSeq() : playSeq());

slider.addEventListener('input', e => {
  curFrame = parseInt(e.target.value) || 0;
  updateFrameLabel();
  if (frameSeq[curFrame]) updateActivity(frameSeq[curFrame].activity);
});

// ── Button wiring ─────────────────────────────────────────────────────────────
document.getElementById('amplitude').addEventListener('input', e => {
  document.getElementById('amp-val').textContent = parseFloat(e.target.value).toFixed(2);
});
document.getElementById('frequency').addEventListener('input', e => {
  document.getElementById('freq-val').textContent = e.target.value;
});

document.getElementById('btn-stim').addEventListener('click', () => {
  const amplitude = parseFloat(document.getElementById('amplitude').value);
  const pattern   = document.getElementById('pattern').value;
  const frequency = parseFloat(document.getElementById('frequency').value);
  let targets     = [...selected];

  if (targets.length === 0) {
    const rnd = new Set();
    while (rnd.size < 5) rnd.add(Math.floor(Math.random() * N_REGIONS));
    targets = [...rnd];
  }

  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: "simulate",
      target_regions: targets,
      amplitude, pattern, frequency,
      duration: 60,
    }));
  } else {
    // Local demo stimulation: direct excitation + spatial spread
    const act = regionMeshes.map(m => m.userData.activity);
    targets.forEach(id => {
      if (id < N_REGIONS) act[id] = Math.min(1, act[id] + amplitude * 0.7);
    });
    for (const id of targets) {
      const p0 = BRAIN_POS[id];
      regionMeshes.forEach((m, j) => {
        if (j === id) return;
        const d = p0.distanceTo(BRAIN_POS[j]);
        if (d < 55) act[j] = Math.min(1, act[j] + amplitude * 0.28 * (1 - d / 55));
      });
    }
    updateActivity(act);
  }
});

document.getElementById('btn-reset').addEventListener('click', () => {
  selected.clear();
  regionMeshes.forEach(m => {
    m.userData.selected = false;
    m.scale.setScalar(0.65 + m.userData.activity * 0.60);
  });
  document.getElementById('sel-count').textContent = '0';
  pauseSeq();
  frameSeq  = [];
  framesFmri = [];
  framesEeg  = [];
  curFrame  = 0;
  slider.max = 0;
  slider.value = 0;
  updateFrameLabel();
  document.getElementById('modality-toggle').style.display = 'none';
  document.getElementById('frame-info').textContent = connected ? '实时数据' : '演示模式';
  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "get_state" }));
  }
});

document.getElementById('btn-mod-fmri').addEventListener('click', () => {
  if (framesFmri.length === 0) return;
  activeModality = 'fmri';
  _applyModalityButtonStyles();
  loadFrameSeq(framesFmri, null);
  document.getElementById('modality-info').textContent =
    `fMRI ${framesFmri.length}帧` + (framesEeg.length > 0 ? ` · EEG ${framesEeg.length}帧` : '');
});

document.getElementById('btn-mod-eeg').addEventListener('click', () => {
  if (framesEeg.length === 0) return;
  activeModality = 'eeg';
  _applyModalityButtonStyles();
  loadFrameSeq(framesEeg, null);
  document.getElementById('modality-info').textContent =
    (framesFmri.length > 0 ? `fMRI ${framesFmri.length}帧 · ` : '') + `EEG ${framesEeg.length}帧`;
});

document.getElementById('btn-load').addEventListener('click', () => {
  const path = document.getElementById('cache-path').value.trim() || null;
  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "load_cache", path }));
  } else {
    alert('请先连接后端服务器（运行 python start.py）');
  }
});

// ── Effective Connectivity (EC) Inference ─────────────────────────────────────
let ecLines      = [];          // Three.js Line objects
let ecTopSources = [];          // top-source region IDs
let ecTopTargets = [];          // top-target region IDs
let ecActivityDelta = null;     // (200,) predicted Δactivity from top source
const EC_LINE_COLOR   = new THREE.Color(0xffcc44);
const EC_SOURCE_COLOR = new THREE.Color(0xffffff);
const EC_TARGET_COLOR = new THREE.Color(0x44ffcc);

function clearECViz() {
  ecLines.forEach(l => scene.remove(l));
  ecLines = [];
  regionMeshes.forEach(m => {
    if (m.userData._ecHighlight) {
      m.material.emissive.setRGB(
        m.material.color.r * 0.22,
        m.material.color.g * 0.22,
        m.material.color.b * 0.22
      );
      m.userData._ecHighlight = false;
    }
  });
}

function drawECLines(topSources, topTargets, ecFlat, n) {
  clearECViz();
  if (!ecFlat) return;

  // For each top source: draw lines to its 5 strongest targets
  topSources.slice(0, 5).forEach(src => {
    // Find top-5 targets for this source (row src of EC matrix)
    const rowStart = src * n;
    const row = ecFlat.slice(rowStart, rowStart + n);
    const topT = Array.from({length: n}, (_, i) => i)
      .sort((a, b) => Math.abs(row[b]) - Math.abs(row[a]))
      .slice(0, 5);

    topT.forEach(dst => {
      if (dst === src) return;
      const weight = Math.abs(row[dst]);
      if (weight < 0.05) return;
      const p0 = BRAIN_POS[src], p1 = BRAIN_POS[dst];
      const pts = new Float32Array([p0.x, p0.y, p0.z, p1.x, p1.y, p1.z]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: EC_LINE_COLOR,
        transparent: true,
        opacity: Math.min(0.75, weight * 1.5),
      }));
      scene.add(line);
      ecLines.push(line);
    });

    // Highlight source sphere
    const srcMesh = regionMeshes[src];
    if (srcMesh) {
      srcMesh.material.emissive.copy(EC_SOURCE_COLOR).multiplyScalar(0.55);
      srcMesh.userData._ecHighlight = true;
    }
  });

  // Highlight top target spheres
  topTargets.slice(0, 5).forEach(dst => {
    const m = regionMeshes[dst];
    if (m) {
      m.material.emissive.copy(EC_TARGET_COLOR).multiplyScalar(0.45);
      m.userData._ecHighlight = true;
    }
  });
}

function handleECResult(msg) {
  const ecStatus = document.getElementById('ec-status');
  if (!msg.ec_flat || !Array.isArray(msg.ec_flat)) {
    ecStatus.textContent = '⚠ EC 数据格式错误';
    return;
  }
  ecTopSources    = msg.top_sources   || [];
  ecTopTargets    = msg.top_targets   || [];
  ecActivityDelta = msg.activity_delta|| null;

  const n = msg.n_regions || 200;
  drawECLines(ecTopSources, ecTopTargets, msg.ec_flat, n);

  // Show activity delta overlay
  if (ecActivityDelta) updateActivity(ecActivityDelta);

  // Enable "刺激推荐靶点" button
  document.getElementById('btn-ec-stim').disabled = false;
  document.getElementById('btn-hide-ec').style.display = '';

  ecStatus.textContent =
    // Region IDs are 0-indexed internally; display as 1-indexed for users
    `✓ ${msg.method} | 最强源: 区域 ${(ecTopSources[0]||0)+1} | ` +
    `连接线: ${ecLines.length}`;
}

document.getElementById('btn-infer-ec').addEventListener('click', () => {
  const method = document.getElementById('ec-method').value;
  document.getElementById('ec-status').textContent = '推断中…';
  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'infer_ec', method, n_lags: 5 }));
  } else {
    // Offline demo: generate synthetic EC client-side
    document.getElementById('ec-status').textContent = '（离线演示，后端未连接）';
    const n = N_REGIONS;
    // goldenAngle: angular increment from the golden ratio for Fibonacci sphere
    // (unused here since we use pre-computed BRAIN_POS positions directly)
    const sigma2 = 40.0 ** 2;
    const ecFlat = new Array(n * n).fill(0);
    for (let i = 0; i < n; i++) {
      const pi = BRAIN_POS[i];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const pj = BRAIN_POS[j];
        const d2 = (pi.x-pj.x)**2 + (pi.y-pj.y)**2 + (pi.z-pj.z)**2;
        ecFlat[i * n + j] = Math.exp(-d2 / (2 * sigma2));
      }
      // Homotopic connection
      const homo = i < 100 ? i + 100 : i - 100;
      ecFlat[i * n + homo] = 0.55;
    }
    const rowSums = Array.from({length:n}, (_, i) =>
      ecFlat.slice(i*n, i*n+n).reduce((a,b) => a+b, 0)
    );
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (rowSums[i] > 0) ecFlat[i*n+j] /= rowSums[i];

    const outScores = Array.from({length:n}, (_, i) =>
      ecFlat.slice(i*n, i*n+n).reduce((a,b) => a+b, 0)
    );
    const inScores = Array.from({length:n}, (_, j) =>
      Array.from({length:n}, (_, i) => ecFlat[i*n+j]).reduce((a,b)=>a+b,0)
    );
    const topSrc = [...Array(n).keys()].sort((a,b) => outScores[b]-outScores[a]).slice(0,10);
    const topTgt = [...Array(n).keys()].sort((a,b) => inScores[b]-inScores[a]).slice(0,10);
    const actDelta = Array.from({length:n}, (_, i) => ecFlat[topSrc[0]*n + i]);
    const maxD = Math.max(...actDelta);
    handleECResult({
      ec_flat: ecFlat, top_sources: topSrc, top_targets: topTgt,
      activity_delta: maxD > 0 ? actDelta.map(v => v/maxD) : actDelta,
      n_regions: n, method: 'demo (offline)'
    });
  }
});

document.getElementById('btn-ec-stim').addEventListener('click', () => {
  if (!ecTopSources.length) return;
  // Select the top-5 source regions and trigger stimulation
  ecTopSources.slice(0, 5).forEach(id => {
    selected.add(id);
    if (regionMeshes[id]) regionMeshes[id].userData.selected = true;
  });
  document.getElementById('sel-count').textContent = selected.size;
  document.getElementById('btn-stim').click();
});

document.getElementById('btn-hide-ec').addEventListener('click', () => {
  clearECViz();
  document.getElementById('btn-hide-ec').style.display = 'none';
  document.getElementById('ec-status').textContent = '';
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────
setStatus(false);
connect();
