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
// Metadata about the most recently loaded dataset (channel/region counts)
let loadedMeta = { nFmriRegions: 0, nEegChannels: 0 };
// Counterfactual (null-stimulation) frames from the last simulation response.
// Shown when the user clicks "○ 对照轨迹" in the timeline bar.
let counterFactualFrames = [];
let showingCounterFactual = false;
// Original loaded data frames (never overwritten by simulation results).
// Used as the source for stimulation initial_state so re-running a simulation
// always starts from the actual recorded data, not from a previous simulation.
let origFrames  = [];
let origCurFrame = 0;     // frame index the user was at in the original data
let viewingSimulation = false;  // true when frameSeq holds simulation output
// Backup of the last stimulation frames (used to restore after CF toggle).
let stimFrames  = [];

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

function updateActivity(arr, rawArr) {
  for (let i = 0; i < regionMeshes.length; i++) {
    const m  = regionMeshes[i];
    const v  = Math.max(0, Math.min(1, arr[i] ?? 0.2));
    m.userData.activity = v;
    m.userData.rawActivity = rawArr ? rawArr[i] : undefined;
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
    const rawLine = m.userData.rawActivity !== undefined
      ? `<br/><span style="color:#aaa;font-size:0.75em">原始值: ${m.userData.rawActivity.toFixed(4)}</span>`
      : '';
    tooltip.style.display = 'block';
    tooltip.style.left    = `${e.offsetX + 14}px`;
    tooltip.style.top     = `${e.offsetY - 10}px`;
    tooltip.innerHTML =
      `<strong>区域 ${id + 1}</strong>&nbsp;&nbsp;${net}<br/>` +
      `活动: ${pct}%&nbsp;&nbsp;${id < 100 ? '左脑' : '右脑'}` +
      rawLine;
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
    // Store actual channel / region counts so the UI can describe the data
    loadedMeta = {
      nFmriRegions: msg.n_fmri_regions || (framesFmri.length > 0 ? 200 : 0),
      nEegChannels: msg.n_eeg_channels || 0,
    };
    // Show/configure modality toggle
    updateModalityToggle(msg.modalities || []);
    // Pick the active modality's frames (fall back to primary frames)
    const modFrames = _getModalityFrames();
    const finalFrames = modFrames.length > 0 ? modFrames : msg.frames;
    // Determine label for modality badge
    let modalityLabel = msg.modality || null;
    if (!modalityLabel) {
      if (activeModality === 'fmri' && framesFmri.length > 0) modalityLabel = 'fMRI';
      else if (activeModality === 'eeg' && framesEeg.length > 0) modalityLabel = 'EEG';
      else if (framesFmri.length > 0) modalityLabel = 'fMRI';
      else if (framesEeg.length > 0)  modalityLabel = 'EEG';
    } else if (modalityLabel === 'simulation') {
      modalityLabel = '⚡ 仿真';
    }
    // Store counterfactual frames (null-stimulation baseline) for comparison
    if (msg.type === 'simulation_result' && Array.isArray(msg.counterfactual_frames)
        && msg.counterfactual_frames.length > 0) {
      counterFactualFrames = msg.counterfactual_frames;
      showingCounterFactual = false;
      const cfBtn = document.getElementById('btn-cf-toggle');
      if (cfBtn) { cfBtn.style.display = ''; cfBtn.textContent = '○ 对照轨迹'; }
    } else if (msg.type !== 'simulation_result') {
      // Clear CF button when loading new cache data (not simulation)
      counterFactualFrames = [];
      const cfBtn = document.getElementById('btn-cf-toggle');
      if (cfBtn) cfBtn.style.display = 'none';
    }
    const isStim = msg.type === 'simulation_result';
    if (isStim) {
      // Backup stimulation frames so the CF toggle can restore them later
      stimFrames = finalFrames.slice();
    }
    loadFrameSeq(finalFrames, msg.path || null, modalityLabel, isStim);
    return;
  }
  // EC inference result
  if (msg.type === 'ec_result') {
    handleECResult(msg);
    const vBtn = document.getElementById('btn-validate-ec');
    if (vBtn) vBtn.disabled = false;
    return;
  }
  // EC validation result
  if (msg.type === 'ec_validation_result') {
    handleECValidationResult(msg);
    return;
  }
  // Brain analysis result
  if (msg.type === 'brain_analysis_result') {
    handleBrainAnalysisResult(msg);
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
    const aStatus = document.getElementById('analysis-status');
    if (aStatus && aStatus.textContent.endsWith('…')) {
      aStatus.textContent = `⚠ ${msg.message}`;
    }
  }
}

// ── Modality toggle (fMRI / EEG) ──────────────────────────────────────────────
function _getModalityFrames() {
  if (activeModality === 'eeg' && framesEeg.length > 0) return framesEeg;
  if (framesFmri.length > 0) return framesFmri;
  return framesEeg.length > 0 ? framesEeg : [];
}

function _updateModalityInfo() {
  const labels = [];
  if (framesFmri.length > 0) {
    const n = loadedMeta.nFmriRegions > 0 ? loadedMeta.nFmriRegions : N_REGIONS;
    labels.push(`fMRI ${n}区×${framesFmri.length}帧`);
  }
  if (framesEeg.length > 0) {
    const n = loadedMeta.nEegChannels > 0
      ? `${loadedMeta.nEegChannels}通道→${N_REGIONS}`
      : `?通道→${N_REGIONS}`;
    labels.push(`EEG ${n}×${framesEeg.length}帧`);
  }
  document.getElementById('modality-info').textContent =
    labels.length > 0 ? labels.join(' · ') : '加载 .pt 文件后可切换';
}

function updateModalityToggle(modalities) {
  const btnFmri = document.getElementById('btn-mod-fmri');
  const btnEeg  = document.getElementById('btn-mod-eeg');
  const hasFmri = modalities.includes('fmri');
  const hasEeg  = modalities.includes('eeg');

  btnFmri.disabled = !hasFmri;
  btnEeg.disabled  = !hasEeg;

  // Keep active modality if still available, else fall back to first available
  if (activeModality === 'eeg'  && !hasEeg)  activeModality = 'fmri';
  if (activeModality === 'fmri' && !hasFmri) activeModality = 'eeg';
  _applyModalityButtonStyles();
  _updateModalityInfo();
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

// ── Modality badge (timeline bar) ─────────────────────────────────────────────
function setModalityBadge(label) {
  const badge = document.getElementById('modality-badge');
  if (label) {
    badge.textContent = label;
    badge.style.display = '';
  } else {
    badge.style.display = 'none';
  }
}

// ── Timeline ──────────────────────────────────────────────────────────────────
// isSimulation=true: frames come from a stimulation run (don't overwrite origFrames)
// isSimulation=false: frames come from cache load or modality switch (real data)
function loadFrameSeq(frames, label, modality, isSimulation = false) {
  viewingSimulation = isSimulation;
  // Keep origFrames pointing at the most recently loaded real data.
  // Simulation results must NOT overwrite this so that subsequent "施加刺激"
  // requests always use actual brain activity as the starting point, not the
  // output of a previous simulation.
  if (!isSimulation) {
    origFrames   = frames;
    origCurFrame = 0;
  }
  frameSeq  = frames;
  curFrame  = 0;
  slider.min   = 0;
  slider.max   = Math.max(0, frames.length - 1);
  slider.value = 0;
  updateFrameLabel();
  document.getElementById('frame-info').textContent =
    `${frames.length} 帧${label ? ' — ' + label.split(/[\\/]/).pop() : ''}`;
  // Show modality badge
  setModalityBadge(modality || null);
  if (frames[0]) updateActivity(frames[0].activity, frames[0].raw);
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
    // Track position in original data when NOT in simulation mode
    if (!viewingSimulation) origCurFrame = curFrame;
    slider.value = curFrame;
    updateFrameLabel();
    updateActivity(frameSeq[curFrame].activity, frameSeq[curFrame].raw);
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
  // Keep origCurFrame in sync while viewing original data
  if (!viewingSimulation) origCurFrame = curFrame;
  updateFrameLabel();
  if (frameSeq[curFrame]) updateActivity(frameSeq[curFrame].activity, frameSeq[curFrame].raw);
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
    // ALWAYS use original loaded data as the stimulation starting point.
    // If the user is viewing a previous simulation result (viewingSimulation=true),
    // frameSeq holds simulation frames.  Using origFrames[origCurFrame] ensures
    // the new simulation starts from REAL brain activity, not a prior simulation
    // output — which is what makes repeated simulations at different time points
    // produce genuinely different results.
    //
    // origCurFrame is updated whenever the slider moves or play advances in
    // non-simulation mode, so it reflects the user's last scrub position in
    // the original data.
    let initial_state;
    if (origFrames.length > 0) {
      // Snapshot the current position in original data before we trigger a new simulation
      if (!viewingSimulation) origCurFrame = curFrame;
      const srcIdx = Math.min(origCurFrame, origFrames.length - 1);
      initial_state = origFrames[srcIdx].activity;
    } else if (frameSeq.length > 0) {
      initial_state = frameSeq[curFrame].activity;
    } else {
      initial_state = regionMeshes.map(m => m.userData.activity);
    }
    ws.send(JSON.stringify({
      type: "simulate",
      target_regions: targets,
      amplitude, pattern, frequency,
      duration: 60,
      initial_state,
    }));
  } else {
    // Offline mode: generate a simple stimulation animation so the effect persists
    // and is not immediately overwritten by the demo render loop.
    const act0 = origFrames.length > 0
      ? origFrames[Math.min(origCurFrame, origFrames.length - 1)].activity.slice()
      : regionMeshes.map(m => m.userData.activity);

    // Gaussian spatial spread weights
    const sw = new Array(N_REGIONS).fill(0);
    targets.forEach(tid => {
      const p0 = BRAIN_POS[tid];
      for (let j = 0; j < N_REGIONS; j++) {
        const d2 = p0.distanceTo(BRAIN_POS[j]) ** 2;
        sw[j] += Math.exp(-d2 / 1800);
      }
    });
    const swMax = sw.reduce((a, b) => Math.max(a, b), 0);
    if (swMax > 0) sw.forEach((_, i) => sw[i] /= swMax);

    // Simple single-step WC stimulation (no connectivity) for offline preview
    const PRE = 5, DUR = 20, POST = 5;
    const localFrames = [];
    let curr = act0.slice();

    function stepLocal(state, stimIn) {
      return state.map((v, i) => {
        const exc = Math.tanh(v + stimIn[i]);
        return Math.max(0, Math.min(1, v * 0.85 + exc * 0.15));
      });
    }

    for (let k = 0; k < PRE; k++) {
      curr = stepLocal(curr, new Array(N_REGIONS).fill(0));
      localFrames.push({ activity: curr.slice() });
    }
    for (let k = 0; k < DUR; k++) {
      const progress = (k + 0.5) / Math.max(DUR, 1);
      const amp = amplitude * Math.sin(Math.PI * progress);
      curr = stepLocal(curr, sw.map(w => amp * w));
      localFrames.push({ activity: curr.slice() });
    }
    for (let k = 0; k < POST; k++) {
      curr = stepLocal(curr, new Array(N_REGIONS).fill(0));
      localFrames.push({ activity: curr.slice() });
    }
    stimFrames = localFrames;
    loadFrameSeq(localFrames, null, '⚡ 仿真 (离线)', true);
  }
});

document.getElementById('btn-reset').addEventListener('click', () => {
  selected.clear();
  regionMeshes.forEach(m => {
    m.userData.selected = false;
    m.userData.rawActivity = undefined;
    m.scale.setScalar(0.65 + m.userData.activity * 0.60);
  });
  document.getElementById('sel-count').textContent = '0';
  pauseSeq();
  frameSeq  = [];
  framesFmri = [];
  framesEeg  = [];
  origFrames = [];
  stimFrames = [];
  counterFactualFrames = [];
  showingCounterFactual = false;
  viewingSimulation = false;
  origCurFrame = 0;
  curFrame  = 0;
  slider.max = 0;
  slider.value = 0;
  updateFrameLabel();
  // Hide counterfactual toggle
  const cfBtn = document.getElementById('btn-cf-toggle');
  if (cfBtn) cfBtn.style.display = 'none';
  // Reset analysis and validation status
  document.getElementById('analysis-status').textContent = '';
  document.getElementById('ec-validation-status').textContent = '';
  // Reset modality buttons to disabled state (no data loaded)
  loadedMeta = { nFmriRegions: 0, nEegChannels: 0 };
  document.getElementById('btn-mod-fmri').disabled = true;
  document.getElementById('btn-mod-eeg').disabled  = true;
  document.getElementById('btn-mod-fmri').className = 'btn-secondary';
  document.getElementById('btn-mod-eeg').className  = 'btn-secondary';
  document.getElementById('modality-info').textContent = '加载 .pt 文件后可切换';
  document.getElementById('frame-info').textContent = connected ? '实时数据' : '演示模式';
  setModalityBadge(null);
  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "get_state" }));
  }
});

document.getElementById('btn-mod-fmri').addEventListener('click', () => {
  if (framesFmri.length === 0) return;
  activeModality = 'fmri';
  _applyModalityButtonStyles();
  loadFrameSeq(framesFmri, null, 'fMRI', false);
  _updateModalityInfo();
});

document.getElementById('btn-mod-eeg').addEventListener('click', () => {
  if (framesEeg.length === 0) return;
  activeModality = 'eeg';
  _applyModalityButtonStyles();
  loadFrameSeq(framesEeg, null, 'EEG', false);
  _updateModalityInfo();
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

  // Build status string — include surrogate reliability when available
  let statusStr =
    `✓ ${msg.method} | 最强源: 区域 ${(ecTopSources[0]||0)+1} | 连接线: ${ecLines.length}`;
  if (msg.fit_quality) {
    const fq = msg.fit_quality;
    const reliStr = fq.reliable
      ? `<span style="color:#44ff88">✓ 可靠</span>`
      : `<span style="color:#ffaa44">⚠ 过拟合 ${fq.overfit_ratio}×</span>`;
    statusStr += ` | ${reliStr}`;
  }
  ecStatus.innerHTML = statusStr;
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

// ── Counterfactual toggle ──────────────────────────────────────────────────────
document.getElementById('btn-cf-toggle').addEventListener('click', () => {
  if (!counterFactualFrames.length) return;
  showingCounterFactual = !showingCounterFactual;
  const cfBtn = document.getElementById('btn-cf-toggle');
  if (showingCounterFactual) {
    // Switch to counterfactual (null-stimulation baseline)
    cfBtn.textContent = '⚡ 刺激轨迹';
    cfBtn.style.background = 'rgba(126,184,255,0.18)';
    cfBtn.style.color = '#7eb8ff';
    cfBtn.style.borderColor = 'rgba(126,184,255,0.4)';
    loadFrameSeq(counterFactualFrames, null, '○ 对照', true);
  } else {
    // Switch back to stimulated trajectory using the backed-up stimFrames
    cfBtn.textContent = '○ 对照轨迹';
    cfBtn.style.background = 'rgba(255,160,44,0.18)';
    cfBtn.style.color = '#ffcc66';
    cfBtn.style.borderColor = 'rgba(255,160,44,0.4)';
    loadFrameSeq(stimFrames.length > 0 ? stimFrames : [], null, '⚡ 仿真', true);
  }
});

// ── EC Validation ─────────────────────────────────────────────────────────────
document.getElementById('btn-validate-ec').addEventListener('click', () => {
  document.getElementById('ec-validation-status').textContent = '验证中…';
  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'validate_ec' }));
  } else {
    document.getElementById('ec-validation-status').textContent =
      '（需要后端连接才能运行 EC 验证）';
  }
});

function handleECValidationResult(msg) {
  const el = document.getElementById('ec-validation-status');
  if (!msg.success || !msg.results) {
    el.textContent = `⚠ 验证失败: ${msg.message || '未知错误'}`;
    return;
  }
  const r = msg.results;
  const parts = [];

  if (r.half_split) {
    const hs = r.half_split;
    if (hs.error) {
      // Show the actual error so the user knows what went wrong
      parts.push(`半分可靠性: ⚠ 计算失败 — ${hs.error.substring(0, 80)}`);
    } else {
      const rVal = hs.half_split_r !== undefined ? hs.half_split_r.toFixed(3) : '?';
      parts.push(`半分可靠性 r=${rVal} — ${hs.interpretation || ''}`);
    }
  }
  if (r.distance) {
    const d = r.distance;
    if (d.error) {
      parts.push(`EC vs 距离: ⚠ ${d.error.substring(0, 60)}`);
    } else {
      const rVal = d.ec_vs_distance_r !== undefined ? d.ec_vs_distance_r.toFixed(3) : '?';
      parts.push(`EC vs 距离 r=${rVal} (p${d.p_approx || ''}) — ${d.interpretation || ''}`);
    }
  }
  if (r.fc_vs_ec) {
    const fe = r.fc_vs_ec;
    if (fe.error) {
      parts.push(`EC vs FC: ⚠ ${fe.error.substring(0, 60)}`);
    } else {
      const rVal = fe.ec_fc_pearson_r !== undefined ? fe.ec_fc_pearson_r.toFixed(3) : '?';
      parts.push(`EC vs FC r=${rVal} — ${fe.interpretation || ''}`);
    }
  }
  el.innerHTML = parts.map(p => `• ${p}`).join('<br/>');
}

// ── Brain State Analysis ───────────────────────────────────────────────────────
document.getElementById('btn-analyze').addEventListener('click', () => {
  const method = document.getElementById('analysis-method').value;
  document.getElementById('analysis-status').textContent = '分析中…';
  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    // Collect optional window parameters (empty = auto-split at midpoint)
    const parseWin = id => { const v = parseInt(document.getElementById(id).value); return isNaN(v) ? undefined : v; };
    const w1s = parseWin('w1s'), w1e = parseWin('w1e');
    const w2s = parseWin('w2s'), w2e = parseWin('w2e');
    const payload = { type: 'analyze_brain', method };
    if (w1s !== undefined) payload.window1_start = w1s;
    if (w1e !== undefined) payload.window1_end   = w1e;
    if (w2s !== undefined) payload.window2_start = w2s;
    if (w2e !== undefined) payload.window2_end   = w2e;
    ws.send(JSON.stringify(payload));
  } else {
    document.getElementById('analysis-status').textContent =
      '（需要后端连接才能运行大脑分析）';
  }
});

// Show window controls only for deviation method
document.getElementById('analysis-method').addEventListener('change', e => {
  const win = document.getElementById('analysis-windows');
  if (win) win.style.display = e.target.value === 'deviation' ? '' : 'none';
});
// Initialise visibility
(function () {
  const win = document.getElementById('analysis-windows');
  if (win) win.style.display = '';
})();

function handleBrainAnalysisResult(msg) {
  const el = document.getElementById('analysis-status');
  if (!msg.success) {
    el.textContent = `⚠ ${msg.message || '分析失败'}`;
    return;
  }
  const s = msg.summary || {};
  if (msg.method === 'deviation') {
    const totalFrames = s.total_frames ? ` (共${s.total_frames}帧)` : '';
    el.innerHTML =
      `偏差分析完成${totalFrames}<br/>` +
      `均值z: <strong>${s.mean_z_score}</strong>  最大z: <strong>${s.max_z_score}</strong><br/>` +
      `异常区域(>2σ): <strong>${s.n_outliers_2std}</strong>  (>3σ): <strong>${s.n_outliers_3std ?? '?'}</strong><br/>` +
      `窗口1: ${s.window1} → 窗口2: ${s.window2}<br/>` +
      `<span style="color:#aaa">${s.interpretation || ''}</span>`;
  } else if (msg.method === 'graph_metrics') {
    el.innerHTML =
      `图论分析完成<br/>` +
      `全脑效率: <strong>${s.global_efficiency}</strong>  ` +
      `密度: <strong>${s.density}</strong><br/>` +
      `<span style="color:#aaa">${s.interpretation || ''}</span>`;
  } else {
    el.textContent = `分析完成 (${msg.method})`;
  }
  // Overlay the analysis result on the 3D brain
  if (Array.isArray(msg.activity)) updateActivity(msg.activity);
  // Highlight regions of interest
  if (Array.isArray(msg.regions_of_interest)) {
    msg.regions_of_interest.slice(0, 10).forEach(id => {
      if (regionMeshes[id]) {
        regionMeshes[id].material.emissive.setRGB(0.3, 0.15, 0.0);
        regionMeshes[id].userData._ecHighlight = true;
      }
    });
  }
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
setStatus(false);
connect();
