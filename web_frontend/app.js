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
let frameSeq   = [];      // [{activity:[…200 floats…]}, …]
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
    loadFrameSeq(msg.frames, msg.path || null);
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
  }
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
  curFrame  = 0;
  slider.max = 0;
  slider.value = 0;
  updateFrameLabel();
  document.getElementById('frame-info').textContent = connected ? '实时数据' : '演示模式';
  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "get_state" }));
  }
});

document.getElementById('btn-load').addEventListener('click', () => {
  const path = document.getElementById('cache-path').value.trim() || null;
  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "load_cache", path }));
  } else {
    alert('请先连接后端服务器（运行 python start.py）');
  }
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────
setStatus(false);
connect();
