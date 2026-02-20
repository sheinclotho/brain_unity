/**
 * TwinBrain Web Frontend
 * 简单、交互式的大脑活动可视化
 */

// ── 配置 ────────────────────────────────────────────────────────────────────
const WS_URL      = "ws://127.0.0.1:8765";
const N_REGIONS   = 200;          // 模拟脑区数量
const RADIUS      = 260;          // 画布半径（脑轮廓）
const REGION_R    = 12;           // 每个脑区圆的半径
const RECONNECT_MS = 3000;        // 断线重连间隔

// ── 状态 ────────────────────────────────────────────────────────────────────
const canvas  = document.getElementById("brain");
const ctx     = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

let regions    = [];              // {x, y, id, activity, selected}
let selected   = new Set();
let ws         = null;
let animFrame  = null;
let connected  = false;
let demoTick   = 0;               // 演示模式帧计数器

// ── 颜色映射（蓝→青→绿→黄→红） ────────────────────────────────────────────
function activityColor(v) {
  // v in [0, 1]
  const stops = [
    [0.00, [0,  30, 200]],
    [0.25, [0, 120, 255]],
    [0.50, [0, 220, 180]],
    [0.75, [255, 220, 0]],
    [1.00, [255,  40,  0]],
  ];
  for (let i = 1; i < stops.length; i++) {
    const [t0, c0] = stops[i - 1];
    const [t1, c1] = stops[i];
    if (v <= t1) {
      const f = (v - t0) / (t1 - t0);
      const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
      const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
      const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(255,40,0)";
}

// ── 布局：将脑区排列成椭圆形 ────────────────────────────────────────────────
function initRegions() {
  const cx = canvas.width  / 2;
  const cy = canvas.height / 2;
  const rx = 220, ry = 195;  // 椭圆半轴

  for (let i = 0; i < N_REGIONS; i++) {
    const angle = (i / N_REGIONS) * 2 * Math.PI;
    // 加入少量随机偏移，看起来更自然
    const jitter = () => (Math.random() - 0.5) * 30;
    regions.push({
      id:       i,
      x:        cx + rx * Math.cos(angle) + jitter(),
      y:        cy + ry * Math.sin(angle) + jitter(),
      activity: Math.random() * 0.3,   // 初始低活动
      selected: false,
      label:    `脑区 ${i + 1}`,
    });
  }
}

// ── 绘制 ────────────────────────────────────────────────────────────────────
function draw() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // 背景脑轮廓（简化椭圆）
  const cx = w / 2, cy = h / 2;
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, 235, 210, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(100,140,255,0.12)";
  ctx.lineWidth   = 2;
  ctx.stroke();
  // 内侧装饰圆
  ctx.beginPath();
  ctx.ellipse(cx, cy, 175, 155, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(100,140,255,0.06)";
  ctx.stroke();
  ctx.restore();

  // 连线（选中脑区之间）
  if (selected.size >= 2) {
    const sel = [...selected].map(id => regions[id]);
    ctx.save();
    ctx.globalAlpha = 0.25;
    ctx.strokeStyle = "#7eb8ff";
    ctx.lineWidth   = 1;
    for (let i = 0; i < sel.length; i++) {
      for (let j = i + 1; j < sel.length; j++) {
        ctx.beginPath();
        ctx.moveTo(sel[i].x, sel[i].y);
        ctx.lineTo(sel[j].x, sel[j].y);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  // 脑区圆圈
  for (const r of regions) {
    const color = activityColor(r.activity);
    const radius = r.selected ? REGION_R + 3 : REGION_R;

    ctx.beginPath();
    ctx.arc(r.x, r.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.fill();

    if (r.selected) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth   = 2;
      ctx.globalAlpha = 1;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  animFrame = requestAnimationFrame(draw);
}

// ── 演示动画（后端未连接时） ─────────────────────────────────────────────────
function demoUpdate() {
  demoTick++;
  for (const r of regions) {
    // 各自独立的低频振荡
    r.activity = 0.15 + 0.12 * Math.sin(demoTick * 0.03 + r.id * 0.4)
               + 0.05 * Math.random();
    r.activity = Math.max(0, Math.min(1, r.activity));
  }
}

// ── WebSocket ────────────────────────────────────────────────────────────────
function connect() {
  try {
    ws = new WebSocket(WS_URL);
  } catch {
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    connected = true;
    setStatus(true);
    ws.send(JSON.stringify({ type: "get_brain_state" }));
  };

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.brain_state && msg.brain_state.activity) {
        const act = msg.brain_state.activity;
        for (let i = 0; i < Math.min(act.length, regions.length); i++) {
          regions[i].activity = Math.max(0, Math.min(1, act[i]));
        }
      } else if (msg.stimulation_result && msg.stimulation_result.activity) {
        const act = msg.stimulation_result.activity;
        for (let i = 0; i < Math.min(act.length, regions.length); i++) {
          regions[i].activity = Math.max(0, Math.min(1, act[i]));
        }
      }
    } catch (e) {
      console.warn("Failed to parse WebSocket message:", ev.data, e);
    }
  };

  ws.onerror = () => {};
  ws.onclose = () => {
    connected = false;
    setStatus(false);
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  setTimeout(connect, RECONNECT_MS);
}

function setStatus(ok) {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  const bk   = document.getElementById("backend-status");
  dot.className  = ok ? "connected" : "";
  text.textContent = ok ? "已连接后端" : "演示模式（后端未连接）";
  if (bk) bk.textContent = ok ? "已连接" : "未连接";
}

// ── 事件：鼠标点击脑区 ───────────────────────────────────────────────────────
canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  for (const r of regions) {
    const dx = mx - r.x, dy = my - r.y;
    if (dx * dx + dy * dy <= (REGION_R + 4) ** 2) {
      r.selected = !r.selected;
      if (r.selected) selected.add(r.id);
      else             selected.delete(r.id);
      document.getElementById("sel-count").textContent = selected.size;
      return;
    }
  }
});

// ── 事件：鼠标悬停提示 ───────────────────────────────────────────────────────
canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  let hit = null;
  for (const r of regions) {
    const dx = mx - r.x, dy = my - r.y;
    if (dx * dx + dy * dy <= (REGION_R + 6) ** 2) { hit = r; break; }
  }

  if (hit) {
    tooltip.style.display = "block";
    tooltip.style.left    = `${e.offsetX + 14}px`;
    tooltip.style.top     = `${e.offsetY - 10}px`;
    tooltip.innerHTML     = `<strong>${hit.label}</strong><br/>活动: ${(hit.activity * 100).toFixed(1)}%`;
    canvas.style.cursor   = "pointer";
  } else {
    tooltip.style.display = "none";
    canvas.style.cursor   = "crosshair";
  }
});

canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });

// ── 按钮：施加刺激 ───────────────────────────────────────────────────────────
document.getElementById("btn-stim").addEventListener("click", () => {
  const amplitude = parseFloat(document.getElementById("amplitude").value);
  const pattern   = document.getElementById("pattern").value;
  const targets   = [...selected];

  if (targets.length === 0) {
    // 没有选中区域时，随机选几个做演示（使用 Set 避免重复）
    const randomSet = new Set();
    while (randomSet.size < 5) {
      randomSet.add(Math.floor(Math.random() * N_REGIONS));
    }
    randomSet.forEach(idx => targets.push(idx));
  }

  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: "simulate_stimulation",
      target_regions: targets,
      amplitude,
      pattern,
      frequency: 10.0,
      duration:  50,
    }));
  } else {
    // 演示模式：本地模拟刺激效果
    for (const id of targets) {
      if (id < regions.length) {
        regions[id].activity = Math.min(1, regions[id].activity + amplitude * 0.6);
      }
    }
    // 临近扩散
    for (const id of targets) {
      const r0 = regions[id];
      for (const r of regions) {
        const d = Math.hypot(r.x - r0.x, r.y - r0.y);
        if (d < 60 && d > 0) {
          r.activity = Math.min(1, r.activity + amplitude * 0.25 * (1 - d / 60));
        }
      }
    }
  }
});

// ── 按钮：重置 ───────────────────────────────────────────────────────────────
document.getElementById("btn-reset").addEventListener("click", () => {
  for (const r of regions) {
    r.activity  = Math.random() * 0.2;
    r.selected  = false;
  }
  selected.clear();
  document.getElementById("sel-count").textContent = "0";

  if (connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "get_brain_state" }));
  }
});

// ── 滑块实时显示 ─────────────────────────────────────────────────────────────
document.getElementById("amplitude").addEventListener("input", (e) => {
  document.getElementById("amp-val").textContent = parseFloat(e.target.value).toFixed(2);
});

// ── 主循环 ───────────────────────────────────────────────────────────────────
function tick() {
  if (!connected) demoUpdate();
  setTimeout(tick, 80);   // ~12 fps 数据更新（渲染由 rAF 驱动）
}

// ── 初始化 ───────────────────────────────────────────────────────────────────
initRegions();
connect();
setStatus(false);
draw();
tick();
