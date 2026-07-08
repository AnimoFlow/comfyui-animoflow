// AnimoFlow draw pad — canvas widgets for AnimoFlow_DrawTrajectory /
// AnimoFlow_DrawWaypoints. Interaction model ported from the webui's
// trajectory pad (animoflow-webui/app/app.js): 8 m × 8 m top-down square,
// origin at center, up-the-page = +Z (the models' forward). Freehand
// stroke → RDP fit → draggable control points → Catmull-Rom smoothing;
// waypoints are numbered pins (click to add, drag to move, double-click
// to delete). Pin TIMES live on a strip under the pad: spread evenly
// until you drag a marker; "Even times" resets. Labels are in seconds,
// read live from the node's duration widget.
//
// The drawn geometry serializes into the node's hidden `points_json`
// STRING widget, so it travels with the workflow JSON and the prompt —
// the Python node does the canonicalization (animoflow_stages.curves).
import { app } from "../../scripts/app.js";

const WORLD_HALF = 4.0;          // ± 4 m each side → 8 m square
const PAD_PX = 512;              // internal canvas resolution
const GRID_M = 1.0;
const MAX_POINTS = 256;
const MIN_STEP_M = 0.05;         // freehand decimation (5 cm)
const FIT_EPS_M = 0.12;          // RDP tolerance — mouse wobble ≈ 0.08 m
const HIT_PX = 14;               // control-point / pin grab radius (canvas px)

const AMBER = "#ec9b18";
const GRID = "rgba(255,255,255,0.07)";
const AXIS = "rgba(255,255,255,0.18)";
const START = "#3fbf62";
const END = "#e0564a";

function toWorld(px, py) {
  const u = px / PAD_PX, v = py / PAD_PX;
  return [-WORLD_HALF + u * 2 * WORLD_HALF, WORLD_HALF - v * 2 * WORLD_HALF];
}
function toCanvas(x, z) {
  const u = (x + WORLD_HALF) / (2 * WORLD_HALF);
  const v = (WORLD_HALF - z) / (2 * WORLD_HALF);
  return [u * PAD_PX, v * PAD_PX];
}

function rdpSimplify(points, eps) {
  if (points.length < 3) return points.slice();
  const keep = new Array(points.length).fill(false);
  keep[0] = keep[points.length - 1] = true;
  const stack = [[0, points.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const [ax, az] = points[a], [bx, bz] = points[b];
    const dx = bx - ax, dz = bz - az;
    const segLen2 = dx * dx + dz * dz;
    let maxD = -1, maxI = -1;
    for (let i = a + 1; i < b; i++) {
      const [px, pz] = points[i];
      let d;
      if (segLen2 < 1e-12) d = Math.hypot(px - ax, pz - az);
      else {
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / segLen2));
        d = Math.hypot(px - (ax + t * dx), pz - (az + t * dz));
      }
      if (d > maxD) { maxD = d; maxI = i; }
    }
    if (maxD > eps) { keep[maxI] = true; stack.push([a, maxI], [maxI, b]); }
  }
  return points.filter((_, i) => keep[i]);
}

function catmullRomSample(ctrl, perSeg) {
  if (ctrl.length < 3) return ctrl.slice();
  const pts = [ctrl[0], ...ctrl, ctrl[ctrl.length - 1]];
  const out = [];
  for (let i = 1; i < pts.length - 2; i++) {
    const p0 = pts[i - 1], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2];
    for (let j = 0; j < perSeg; j++) {
      const t = j / perSeg, t2 = t * t, t3 = t2 * t;
      out.push([
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
               (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
               (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
               (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
               (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
      ]);
    }
  }
  out.push([ctrl[ctrl.length - 1][0], ctrl[ctrl.length - 1][1]]);
  return out;
}

function drawBase(ctx) {
  ctx.clearRect(0, 0, PAD_PX, PAD_PX);
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fillRect(0, 0, PAD_PX, PAD_PX);
  const cells = (2 * WORLD_HALF) / GRID_M;
  const step = PAD_PX / cells;
  ctx.lineWidth = 1;
  for (let i = 0; i <= cells; i++) {
    const p = i * step;
    ctx.strokeStyle = i === cells / 2 ? AXIS : GRID;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, PAD_PX); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(PAD_PX, p); ctx.stroke();
  }
  // forward arrow (+Z, up the page) at origin
  ctx.strokeStyle = AXIS; ctx.fillStyle = AXIS;
  const [ox, oy] = toCanvas(0, 0);
  ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy - 26); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(ox, oy - 32); ctx.lineTo(ox - 4, oy - 24); ctx.lineTo(ox + 4, oy - 24);
  ctx.closePath(); ctx.fill();
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.fillText("1 m grid · up = forward (+Z)", 8, PAD_PX - 8);
}

function makePad(node, mode) {
  // ---- state -------------------------------------------------------------
  const widget = node.widgets.find((w) => w.name === "points_json");
  let ctrl = [];        // trajectory: [[x,z],...]  waypoints: [{x,z,f},...]
  let stroke = [];
  let dragIdx = null;
  let strokeActive = false;
  let lastPt = null;
  let customTimes = false;   // waypoints: true once the user drags a time marker
  let timeDragIdx = null;    // waypoints: marker being dragged on the strip

  try {
    const parsed = JSON.parse(widget.value || "[]");
    if (Array.isArray(parsed)) ctrl = parsed;
  } catch (e) { /* start empty */ }
  if (mode === "waypoints" && ctrl.length > 1) {
    // Loaded workflow with hand-set times? Keep them.
    customTimes = ctrl.some((p, i) =>
      Math.abs((p.f ?? 0) - i / (ctrl.length - 1)) > 1e-3);
  }

  function nodeDuration() {
    const w = node.widgets.find((x) => x.name === "duration");
    const v = w ? parseFloat(w.value) : NaN;
    return Number.isFinite(v) && v > 0 ? v : 4.0;
  }

  // hide the raw JSON widget (the pad owns it)
  widget.hidden = true;
  widget.computeSize = () => [0, -4];

  // ---- DOM ----------------------------------------------------------------
  const wrap = document.createElement("div");
  wrap.style.cssText =
    "display:flex;flex-direction:column;gap:4px;width:100%;height:100%;";
  const canvas = document.createElement("canvas");
  canvas.width = PAD_PX; canvas.height = PAD_PX;
  canvas.style.cssText =
    "width:100%;aspect-ratio:1/1;border:1px solid rgba(255,255,255,0.15);" +
    "border-radius:6px;cursor:crosshair;touch-action:none;";
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:6px;align-items:center;";
  const clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear";
  const BTN_CSS =
    "font-size:11px;padding:2px 10px;border-radius:4px;border:1px solid " +
    "rgba(255,255,255,0.2);background:transparent;color:inherit;cursor:pointer;";
  clearBtn.style.cssText = BTN_CSS;
  const meta = document.createElement("span");
  meta.style.cssText = "font-size:11px;opacity:0.6;";
  bar.appendChild(clearBtn);

  // Time strip (waypoints only): one draggable marker per pin, 0 → duration.
  const TL_H = 44, TL_PAD = 16;
  let timeline = null, tctx = null, evenBtn = null;
  if (mode === "waypoints") {
    timeline = document.createElement("canvas");
    timeline.width = PAD_PX; timeline.height = TL_H;
    timeline.style.cssText =
      "width:100%;border:1px solid rgba(255,255,255,0.15);" +
      "border-radius:6px;cursor:ew-resize;touch-action:none;";
    tctx = timeline.getContext("2d");
    evenBtn = document.createElement("button");
    evenBtn.textContent = "Even times";
    evenBtn.style.cssText = BTN_CSS;
    bar.appendChild(evenBtn);
  }
  bar.appendChild(meta);
  wrap.appendChild(canvas);
  if (timeline) wrap.appendChild(timeline);
  wrap.appendChild(bar);

  const ctx = canvas.getContext("2d");

  // ---- serialization -------------------------------------------------------
  function commit() {
    if (mode === "trajectory") {
      const perSeg = ctrl.length > 1
        ? Math.max(2, Math.min(10, Math.floor((MAX_POINTS - 1) / (ctrl.length - 1))))
        : 2;
      const sampled = ctrl.length >= 3 ? catmullRomSample(ctrl, perSeg) : ctrl.slice();
      widget.value = JSON.stringify(sampled.map(([x, z]) =>
        [Math.round(x * 1000) / 1000, Math.round(z * 1000) / 1000]));
      let len = 0;
      for (let i = 1; i < sampled.length; i++)
        len += Math.hypot(sampled[i][0] - sampled[i-1][0], sampled[i][1] - sampled[i-1][1]);
      meta.textContent = ctrl.length
        ? `${ctrl.length} control point${ctrl.length === 1 ? "" : "s"} · ${len.toFixed(2)} m`
        : "draw a path — start point = where the walk begins";
    } else {
      const n = ctrl.length;
      if (!customTimes) {
        ctrl.forEach((p, i) => { p.f = n > 1 ? i / (n - 1) : 0; });
      }
      widget.value = JSON.stringify(ctrl.map((p) => ({
        x: Math.round(p.x * 1000) / 1000, z: Math.round(p.z * 1000) / 1000,
        f: Math.round((p.f ?? 0) * 1000) / 1000 })));
      meta.textContent = n
        ? `${n} pin${n === 1 ? "" : "s"} · ${customTimes ? "custom times" : "times spread evenly"}`
        : "click to place pins — pin 1 = start";
    }
    node.graph?.setDirtyCanvas(true, true);
  }

  // ---- painting -------------------------------------------------------------
  function repaint() {
    drawBase(ctx);
    if (mode === "trajectory") {
      const line = strokeActive ? stroke
        : (ctrl.length >= 3 ? catmullRomSample(ctrl, 8) : ctrl);
      if (line.length > 1) {
        ctx.strokeStyle = AMBER; ctx.lineWidth = 3; ctx.lineJoin = "round";
        ctx.beginPath();
        line.forEach(([x, z], i) => {
          const [cx, cy] = toCanvas(x, z);
          i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy);
        });
        ctx.stroke();
      }
      const dots = strokeActive ? [] : ctrl;
      dots.forEach(([x, z], i) => {
        const [cx, cy] = toCanvas(x, z);
        ctx.fillStyle = i === 0 ? START : (i === dots.length - 1 ? END : AMBER);
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
      });
    } else {
      if (ctrl.length > 1) {
        ctx.strokeStyle = "rgba(236,155,24,0.45)";
        ctx.setLineDash([6, 5]); ctx.lineWidth = 2;
        ctx.beginPath();
        ctrl.forEach((p, i) => {
          const [cx, cy] = toCanvas(p.x, p.z);
          i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy);
        });
        ctx.stroke(); ctx.setLineDash([]);
      }
      ctrl.forEach((p, i) => {
        const [cx, cy] = toCanvas(p.x, p.z);
        ctx.fillStyle = i === 0 ? START : (i === ctrl.length - 1 ? END : AMBER);
        ctx.beginPath(); ctx.arc(cx, cy, 9, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#111";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), cx, cy);
      });
      ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
    }
    if (timeline) repaintTimeline();
  }

  // ---- time strip (waypoints) -----------------------------------------------
  const timeToX = (f) => TL_PAD + f * (PAD_PX - 2 * TL_PAD);
  const xToTime = (x) => Math.min(1, Math.max(0, (x - TL_PAD) / (PAD_PX - 2 * TL_PAD)));
  const MIN_GAP = 0.02;

  function repaintTimeline() {
    const dur = nodeDuration();
    tctx.clearRect(0, 0, PAD_PX, TL_H);
    tctx.fillStyle = "rgba(0,0,0,0.35)";
    tctx.fillRect(0, 0, PAD_PX, TL_H);
    const midY = 18;
    tctx.strokeStyle = AXIS; tctx.lineWidth = 2;
    tctx.beginPath(); tctx.moveTo(TL_PAD, midY); tctx.lineTo(PAD_PX - TL_PAD, midY); tctx.stroke();
    tctx.font = "10px sans-serif";
    tctx.fillStyle = "rgba(255,255,255,0.45)";
    tctx.textAlign = "left";  tctx.fillText("0s", 3, midY + 4);
    tctx.textAlign = "right"; tctx.fillText(`${dur}s`, PAD_PX - 3, midY + 4);
    ctrl.forEach((p, i) => {
      const x = timeToX(p.f ?? 0);
      tctx.fillStyle = i === 0 ? START : (i === ctrl.length - 1 ? END : AMBER);
      tctx.beginPath(); tctx.arc(x, midY, 8, 0, Math.PI * 2); tctx.fill();
      tctx.fillStyle = "#111";
      tctx.font = "bold 9px sans-serif";
      tctx.textAlign = "center"; tctx.textBaseline = "middle";
      tctx.fillText(String(i + 1), x, midY);
      tctx.fillStyle = "rgba(255,255,255,0.6)";
      tctx.font = "9px sans-serif"; tctx.textBaseline = "alphabetic";
      tctx.fillText(`${((p.f ?? 0) * dur).toFixed(1)}`, x, TL_H - 5);
    });
    tctx.textAlign = "start";
  }

  function tlPx(e) {
    const r = timeline.getBoundingClientRect();
    return (e.clientX - r.left) * (PAD_PX / r.width);
  }

  if (timeline) {
    timeline.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      const x = tlPx(e);
      let best = null, bestD = 12;
      ctrl.forEach((p, i) => {
        const d = Math.abs(timeToX(p.f ?? 0) - x);
        if (d < bestD) { bestD = d; best = i; }
      });
      timeDragIdx = best;
      if (timeDragIdx !== null) timeline.setPointerCapture?.(e.pointerId);
    });
    timeline.addEventListener("pointermove", (e) => {
      if (timeDragIdx === null) return;
      e.stopPropagation();
      const i = timeDragIdx;
      const lo = i > 0 ? (ctrl[i - 1].f ?? 0) + MIN_GAP : 0;
      const hi = i < ctrl.length - 1 ? (ctrl[i + 1].f ?? 1) - MIN_GAP : 1;
      ctrl[i].f = Math.min(hi, Math.max(lo, xToTime(tlPx(e))));
      customTimes = true;
      commit(); repaintTimeline();
    });
    timeline.addEventListener("pointerup", (e) => {
      e.stopPropagation();
      if (timeDragIdx !== null) { timeDragIdx = null; commit(); repaint(); }
    });
    evenBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      customTimes = false;
      commit(); repaint();
    });
  }

  // ---- pointer handling ------------------------------------------------------
  function evPx(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (PAD_PX / r.width),
            (e.clientY - r.top) * (PAD_PX / r.height)];
  }
  function hitIndex(px, py) {
    for (let i = ctrl.length - 1; i >= 0; i--) {
      const p = mode === "trajectory" ? ctrl[i] : [ctrl[i].x, ctrl[i].z];
      const [cx, cy] = toCanvas(p[0], p[1]);
      if (Math.hypot(px - cx, py - cy) <= HIT_PX) return i;
    }
    return null;
  }

  canvas.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    const [px, py] = evPx(e);
    dragIdx = hitIndex(px, py);
    if (dragIdx !== null) { canvas.setPointerCapture?.(e.pointerId); return; }
    const [wx, wz] = toWorld(px, py);
    if (mode === "trajectory") {
      strokeActive = true;
      lastPt = [wx, wz];
      stroke = [[wx, wz]];
      ctrl = [];
    } else {
      if (ctrl.length >= 16) return;
      let f = 0;
      if (customTimes && ctrl.length) {
        // Keep the user's hand-set times; slot the new pin after the last.
        let last = ctrl[ctrl.length - 1].f ?? 0;
        if (last > 0.9) {  // no room — compress existing times toward 0
          ctrl.forEach((p) => { p.f = (p.f ?? 0) * 0.9; });
          last = ctrl[ctrl.length - 1].f;
        }
        f = last + (1 - last) / 2;
      }
      ctrl.push({ x: wx, z: wz, f });
      commit();
    }
    repaint();
    canvas.setPointerCapture?.(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (dragIdx === null && !strokeActive) return;
    e.stopPropagation();
    const [px, py] = evPx(e);
    const [wx, wz] = toWorld(px, py);
    if (dragIdx !== null) {
      if (mode === "trajectory") ctrl[dragIdx] = [wx, wz];
      else { ctrl[dragIdx].x = wx; ctrl[dragIdx].z = wz; }
      commit(); repaint(); return;
    }
    if (!lastPt || Math.hypot(wx - lastPt[0], wz - lastPt[1]) >= MIN_STEP_M) {
      lastPt = [wx, wz];
      stroke.push([wx, wz]);
      repaint();
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    e.stopPropagation();
    if (dragIdx !== null) { dragIdx = null; commit(); repaint(); return; }
    if (strokeActive && stroke.length >= 2) {
      ctrl = rdpSimplify(stroke, FIT_EPS_M);
    }
    stroke = []; strokeActive = false; lastPt = null;
    commit(); repaint();
  });

  canvas.addEventListener("dblclick", (e) => {
    if (mode !== "waypoints") return;
    e.stopPropagation();
    const [px, py] = evPx(e);
    const i = hitIndex(px, py);
    if (i !== null) { ctrl.splice(i, 1); commit(); repaint(); }
  });

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    ctrl = []; stroke = [];
    commit(); repaint();
  });

  const extraH = timeline ? 96 : 40;  // strip + bar vs bar only
  // ComfyUI ≥1.45 sizes DOM widgets from getMinHeight/getMaxHeight and IGNORES
  // a widget.computeSize override. The pad canvas is `width:100%;
  // aspect-ratio:1/1`, so its pixel height tracks the node WIDTH — the widget
  // box must track the same width or the box and canvas disagree and the node
  // visibly jumps/shrinks on interaction. Derive the box height from the node's
  // current width (square canvas + the strip/bar below it), and pin it with
  // getMaxHeight so a stray vertical drag can't desync them.
  const boxHeight = () => Math.max(240, (node.size?.[0] ?? 340) - 20) + extraH;
  node.addDOMWidget("animoflow_pad", "AF_PAD", wrap, {
    serialize: false,
    getMinHeight: boxHeight,
    getMaxHeight: boxHeight,
  });

  // Keep the strip's second labels live when the duration widget changes.
  const durWidget = node.widgets.find((w) => w.name === "duration");
  if (timeline && durWidget) {
    const prevCb = durWidget.callback;
    durWidget.callback = function (...args) {
      const r = prevCb?.apply(this, args);
      repaintTimeline();
      return r;
    };
  }

  commit(); repaint();
  // Pre-fit the node to its computed size so the workflow's saved (oversized)
  // height doesn't leave slack that ComfyUI clamps away on the first click —
  // that clamp is exactly the "shrinks when you interact" glitch.
  node.size[0] = Math.max(node.size[0], 340);
  const fit = node.computeSize();
  node.setSize([Math.max(340, fit[0]), fit[1]]);
}

app.registerExtension({
  name: "animoflow.drawpad",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const mode = nodeData.name === "AnimoFlow_DrawTrajectory" ? "trajectory"
               : nodeData.name === "AnimoFlow_DrawWaypoints" ? "waypoints" : null;
    if (!mode) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      makePad(this, mode);
    };
  },
});
