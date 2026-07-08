// AnimoFlow timeline widget — canvas segment editor for
// AnimoFlow_PriorMDMTimeline. An NLE-style timeline for prompt segments,
// shown instead of exposing a raw `segments_json` textarea.
//
// Interaction grammar (matches the webui):
//   - Click the "+" tile          → append a new segment.
//   - Click a segment body         → edit its prompt inline (textarea overlay).
//   - Drag a segment's right edge  → resize its duration.
//   - Drag a segment body sideways → reorder past a neighbour's midpoint.
//   - Click the "×" chip / right-click / Delete → remove the segment.
//
// The node stores {prompt, num_frames}; the editor thinks in
// {prompt, duration_seconds}. priorMDM is 20 fps, so the two convert at the
// serialization boundary (num_frames = round(duration * 20)). The editor
// serializes into the node's hidden `segments_json` STRING widget, so the
// segments travel with the workflow JSON exactly like the raw textarea did.
import { app } from "../../scripts/app.js";

const PRIORMDM_FPS = 20;          // priorMDM native fps (HumanML3D)

// ── Tunables (mirrored from the webui timeline) ─────────────────────────────
const MIN_SEG_SECONDS     = 0.5;
const MAX_SEG_SECONDS     = 8.0;
const DEFAULT_SEG_MIN_SEC = 3.0;
const DEFAULT_SEG_MAX_SEC = 5.0;
const MIN_TOTAL_SECONDS   = 8.0;
const MAX_TOTAL_SECONDS   = 20.0;
const DRAG_THRESHOLD_PX   = 4;

const EXAMPLE_PROMPTS = [
  "a person walks forward",
  "the person sits down while crossing their legs",
  "the person makes a long leap forward",
  "the person runs in a circle",
  "the person waves with their right hand",
  "the person dances energetically",
  "the person picks something up off the floor",
  "the person turns and runs",
];

function _pickPromptDifferentFrom(prevPrompt) {
  const choices = EXAMPLE_PROMPTS.filter((p) => p !== prevPrompt);
  const pool = choices.length ? choices : EXAMPLE_PROMPTS;
  return pool[Math.floor(Math.random() * pool.length)];
}
function _randomSegmentDuration() {
  const span = DEFAULT_SEG_MAX_SEC - DEFAULT_SEG_MIN_SEC;
  return Math.round((DEFAULT_SEG_MIN_SEC + Math.random() * span) * 2) / 2;
}

const PLUS_TILE_PX   = 56;
const RESIZE_HIT_PX  = 10;
const TICK_PAD_BOTTOM = 26;
const SEG_RADIUS     = 8;
const TEXT_PAD_X     = 12;
const TEXT_TOP_Y     = 10;
const CANVAS_CSS_H   = 110;       // canvas height (px), matches the webui

const DEL_BTN_SIZE = 18;
const DEL_BTN_PAD  = 4;

// Editorial palette — 7 bright pastels tuned to sit beside the warm-amber
// brand accent (#ffcf7b). Black text reads on all of them. Cycled per new
// segment; index stored on the segment so reorders preserve identity.
const SEG_COLORS = [
  "#a8c8e8", "#d0b8e8", "#a8e0c0", "#f5c878",
  "#f0a8a8", "#a8d8da", "#e8d088",
];

// Dark palette — ComfyUI's canvas is dark, so the track/plus-tile/ticks are
// dark-mode; only the segment fills stay bright pastel (black text reads on
// them, and they match the webui). Affordances drawn ON a segment (hover/
// select borders, resize grip, delete chip, inline caret) stay dark since
// their backdrop is the light pastel; ticks/stats sit on the dark track so
// they're light.
const COLORS = {
  surface:        "#2a2a2a",
  border:         "rgba(255,255,255,0.12)",
  segText:        "#0f0f0f",
  segTextMuted:   "rgba(15,15,15,0.55)",
  hoverBorder:    "rgba(15,15,15,0.35)",
  selectBorder:   "rgba(15,15,15,0.8)",
  resizeHint:     "rgba(15,15,15,0.6)",
  tick:           "rgba(255,255,255,0.35)",
  tickStrong:     "rgba(255,255,255,0.65)",
  plusBg:         "#3a3a3a",
  plusBorder:     "rgba(255,255,255,0.15)",
  plusBgHover:    "#474747",
  plusBorderHover:"rgba(255,255,255,0.35)",
  plusFg:         "#e8e8e8",
};

let _segIdCounter = 0;
const _newSegId = () => ++_segIdCounter;

// Inject the widget stylesheet once. Ported from the webui .tl-* rules; the
// inline-edit caret/text is BLACK (the segments are bright pastels — the
// webui's own white caret is stale relative to that palette).
function ensureStyle() {
  if (document.getElementById("animoflow-timeline-style")) return;
  const s = document.createElement("style");
  s.id = "animoflow-timeline-style";
  s.textContent = `
  .aftl-root { position: relative; width: 100%; padding: 4px 2px; box-sizing: border-box; }
  .aftl-canvas { display: block; width: 100%; user-select: none; -webkit-user-select: none; }
  .aftl-stats { margin-top: 6px; display: flex; gap: 8px; align-items: center;
    font-size: 12px; color: #b8b8b8; flex-wrap: wrap;
    font-family: Inter, -apple-system, system-ui, sans-serif; }
  .aftl-stats-segments, .aftl-stats-total { font-variant-numeric: tabular-nums; }
  .aftl-stats-spacer { flex: 1; }
  .aftl-stats-hint { color: #8a8a8a; transition: color 0.2s; }
  .aftl-stats-hint strong { color: #cfcfcf; font-weight: 600; }
  .aftl-stats-hint--warn { color: #d98a2b; font-weight: 500; }
  .aftl-stats-dot { color: #8a8a8a; }
  .aftl-inline-edit { position: absolute; z-index: 5; background: transparent;
    color: #0f0f0f; caret-color: rgba(15,15,15,0.95);
    font: 500 13px Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    padding: 4px; margin: 0; border: none; outline: none; resize: none;
    overflow: hidden; line-height: 1.3; }
  .aftl-inline-edit::selection { background: rgba(15,15,15,0.18); }
  .aftl-inline-edit[hidden] { display: none; }
  `;
  document.head.appendChild(s);
}

class TimelineEditor extends EventTarget {
  constructor(rootEl, opts = {}) {
    super();
    this.root = rootEl;
    this.opts = { minSegments: 2, maxTotal: MAX_TOTAL_SECONDS, initial: [], ...opts };
    this._segments = this.opts.initial.map((s, i) => ({
      id: _newSegId(),
      prompt: s.prompt,
      duration: Math.max(MIN_SEG_SECONDS, Math.min(MAX_SEG_SECONDS,
        s.duration ?? _randomSegmentDuration())),
      colorIdx: i % SEG_COLORS.length,
    }));
    this._nextColorIdx = this._segments.length % SEG_COLORS.length;

    this.root.classList.add("aftl-root");
    this.root.innerHTML = `
      <canvas class="aftl-canvas"></canvas>
      <textarea class="aftl-inline-edit" hidden rows="1" spellcheck="true"
                maxlength="500" aria-label="Edit segment prompt"></textarea>
      <div class="aftl-stats">
        <span class="aftl-stats-segments"></span>
        <span class="aftl-stats-dot">·</span>
        <span class="aftl-stats-total"></span>
        <span class="aftl-stats-spacer"></span>
        <span class="aftl-stats-hint">Click <strong>+</strong> to add · click to edit · drag right edge <strong>‖</strong> to resize · <strong>×</strong> to delete</span>
      </div>`;
    this.canvas        = this.root.querySelector(".aftl-canvas");
    this.inlineEdit    = this.root.querySelector(".aftl-inline-edit");
    this.statsSegments = this.root.querySelector(".aftl-stats-segments");
    this.statsTotal    = this.root.querySelector(".aftl-stats-total");
    this.statsHint     = this.root.querySelector(".aftl-stats-hint");
    this._defaultHintHTML = this.statsHint.innerHTML;
    this._capNoteTimer = null;

    this._hoverId = null; this._hoverIsResize = false; this._hoverIsPlus = false;
    this._hoverIsDelete = false; this._selectedId = null;
    this._dragKind = null; this._dragSegId = null; this._dragStartX = 0;
    this._dragStartDuration = 0; this._editingId = null; this._totalDuration = 0;

    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp   = this._onPointerUp.bind(this);
    this._onDblClick    = this._onDblClick.bind(this);
    this._onContextMenu = this._onContextMenu.bind(this);
    this._onKeyDown     = this._onKeyDown.bind(this);
    this._onEditKey = (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this._commitEdit(); }
      else if (e.key === "Escape")          { e.preventDefault(); this._cancelEdit(); }
    };
    this._onEditBlur = () => { if (this._editingId !== null) this._commitEdit(); };
    this._onDocumentPointerDown = (e) => {
      if (this.root.contains(e.target)) return;
      if (this._selectedId !== null || this._hoverId !== null) {
        this._selectedId = null; this._hoverId = null;
        this._hoverIsResize = false; this._hoverIsDelete = false;
        this._render();
      }
    };

    this.canvas.addEventListener("pointerdown", this._onPointerDown);
    window.addEventListener("pointermove",      this._onPointerMove);
    window.addEventListener("pointerup",        this._onPointerUp);
    this.canvas.addEventListener("dblclick",    this._onDblClick);
    this.canvas.addEventListener("contextmenu", this._onContextMenu);
    window.addEventListener("keydown",          this._onKeyDown, true);
    document.addEventListener("pointerdown",    this._onDocumentPointerDown);
    this.inlineEdit.addEventListener("keydown", this._onEditKey);
    this.inlineEdit.addEventListener("blur",    this._onEditBlur);

    // Re-fit the canvas whenever the node (and thus our root) is resized.
    this._ro = new ResizeObserver(() => { this._resizeCanvas(); this._render(); });
    this._ro.observe(this.root);

    this._resizeCanvas();
    this._render();
  }

  // ── Public API ───────────────────────────────────────────────────────────
  getSegments() { return this._segments.map(({ prompt, duration }) => ({ prompt, duration })); }
  destroy() {
    this._ro?.disconnect();
    this.canvas.removeEventListener("pointerdown", this._onPointerDown);
    window.removeEventListener("pointermove",      this._onPointerMove);
    window.removeEventListener("pointerup",        this._onPointerUp);
    this.canvas.removeEventListener("dblclick",    this._onDblClick);
    this.canvas.removeEventListener("contextmenu", this._onContextMenu);
    window.removeEventListener("keydown",          this._onKeyDown, true);
    document.removeEventListener("pointerdown",    this._onDocumentPointerDown);
  }

  // ── Layout math ──────────────────────────────────────────────────────────
  _totalSec() { return this._segments.reduce((a, s) => a + s.duration, 0); }

  _layoutSegments() {
    const W = this.canvas.width / (window.devicePixelRatio || 1);
    const usableW = W - PLUS_TILE_PX - 8;
    const totalSec = Math.max(MIN_TOTAL_SECONDS, this._totalSec());
    this._totalDuration = this._totalSec();
    const pxPerSec = usableW / totalSec;
    let x = 0;
    const out = [];
    for (const seg of this._segments) {
      const w = Math.max(8, seg.duration * pxPerSec);
      out.push({ seg, x, w });
      x += w;
    }
    return { tiles: out, plusX: x + 4, plusW: PLUS_TILE_PX, pxPerSec, totalSec };
  }

  _deleteBtnRect(tile) {
    if (this._segments.length <= this.opts.minSegments) return null;
    if (tile.w < DEL_BTN_SIZE + DEL_BTN_PAD * 2 + 8) return null;
    return {
      cx: tile.x + tile.w - DEL_BTN_PAD - DEL_BTN_SIZE / 2,
      cy: 6 + DEL_BTN_PAD + DEL_BTN_SIZE / 2,
      r: DEL_BTN_SIZE / 2,
    };
  }

  _hitTest(px, py) {
    const { tiles, plusX, plusW } = this._layoutSegments();
    const trackBottom = (this.canvas.height / (window.devicePixelRatio || 1)) - TICK_PAD_BOTTOM;
    if (py < 0 || py > trackBottom) return { kind: null };
    if (px >= plusX && px <= plusX + plusW) return { kind: "plus" };
    for (const t of tiles) {
      if (px >= t.x && px <= t.x + t.w) {
        const armed = (this._hoverId === t.seg.id || this._selectedId === t.seg.id);
        if (armed && this._editingId !== t.seg.id) {
          const r = this._deleteBtnRect(t);
          if (r) {
            const dx = px - r.cx, dy = py - r.cy;
            if (dx * dx + dy * dy <= r.r * r.r) return { kind: "delete", seg: t.seg, tile: t };
          }
        }
        return { kind: "segment", seg: t.seg, isResize: px >= t.x + t.w - RESIZE_HIT_PX, tile: t };
      }
    }
    return { kind: null };
  }

  // ── Rendering ────────────────────────────────────────────────────────────
  _resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(200, this.root.clientWidth - 4);
    this.canvas.style.width  = cssW + "px";
    this.canvas.style.height = CANVAS_CSS_H + "px";
    this.canvas.width  = Math.floor(cssW * dpr);
    this.canvas.height = Math.floor(CANVAS_CSS_H * dpr);
    this.canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this._editingId !== null) this._positionInlineEdit();
  }

  _render() {
    const ctx = this.canvas.getContext("2d");
    const W = this.canvas.width  / (window.devicePixelRatio || 1);
    const H = this.canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, W, H);

    const trackH = H - TICK_PAD_BOTTOM;
    this._fillRoundRect(ctx, 0, 0, W, trackH, SEG_RADIUS, COLORS.surface);
    ctx.lineWidth = 1;
    this._strokeRoundRect(ctx, 0.5, 0.5, W - 1, trackH - 1, SEG_RADIUS, COLORS.border);

    const { tiles, plusX, plusW, totalSec } = this._layoutSegments();

    for (const t of tiles) {
      const isHover = this._hoverId === t.seg.id;
      const isSelected = this._selectedId === t.seg.id;
      const isEditing = this._editingId === t.seg.id;
      const isDragging = this._dragSegId === t.seg.id;
      const x = t.x + 2, w = Math.max(2, t.w - 2), y = 6, h = trackH - 12;

      ctx.save();
      ctx.globalAlpha = isEditing ? 0.7 : 1.0;
      this._fillRoundRect(ctx, x, y, w, h, 6, SEG_COLORS[t.seg.colorIdx % SEG_COLORS.length]);
      ctx.restore();

      if (isSelected || isHover || isDragging || isEditing) {
        const strong = isSelected || isDragging || isEditing;
        ctx.lineWidth = strong ? 2 : 1.5;
        this._strokeRoundRect(ctx, x + 0.5, y + 0.5, w - 1, h - 1, 6,
          strong ? COLORS.selectBorder : COLORS.hoverBorder);
      }

      if ((isHover || isSelected) && !isEditing && w >= 24) {
        const inZone = isHover && this._hoverIsResize;
        ctx.fillStyle = inZone ? COLORS.selectBorder : COLORS.resizeHint;
        const gripH = Math.min(22, h - 18), gripY = y + (h - gripH) / 2;
        ctx.fillRect(x + w - 8, gripY, 1.6, gripH);
        ctx.fillRect(x + w - 5, gripY, 1.6, gripH);
      }

      if ((isHover || isSelected || isDragging) && !isEditing) {
        const r = this._deleteBtnRect(t);
        if (r) {
          const isOver = this._hoverIsDelete && this._hoverId === t.seg.id;
          ctx.save();
          ctx.beginPath(); ctx.arc(r.cx, r.cy, r.r, 0, Math.PI * 2);
          ctx.fillStyle = isOver ? "#0f0f0f" : "rgba(15,15,15,0.55)"; ctx.fill();
          ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.6; ctx.lineCap = "round";
          const arm = r.r - 5;
          ctx.beginPath();
          ctx.moveTo(r.cx - arm, r.cy - arm); ctx.lineTo(r.cx + arm, r.cy + arm);
          ctx.moveTo(r.cx + arm, r.cy - arm); ctx.lineTo(r.cx - arm, r.cy + arm);
          ctx.stroke(); ctx.restore();
        }
      }

      if (!isEditing) {
        const xChipReserved = ((isHover || isSelected || isDragging) && this._deleteBtnRect(t))
          ? (DEL_BTN_SIZE + DEL_BTN_PAD + 4) : 0;
        ctx.fillStyle = COLORS.segText;
        ctx.font = "500 13px Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif";
        ctx.textBaseline = "top";
        const promptText = this._truncate(ctx, t.seg.prompt || "(empty prompt)",
          w - TEXT_PAD_X * 2 - xChipReserved);
        ctx.fillText(promptText, x + TEXT_PAD_X, y + TEXT_TOP_Y);

        ctx.fillStyle = COLORS.segTextMuted;
        ctx.font = "400 11px Inter, -apple-system, system-ui, sans-serif";
        const durLabel = this._formatDuration(t.seg.duration);
        if (ctx.measureText(durLabel).width < w - TEXT_PAD_X * 2)
          ctx.fillText(durLabel, x + TEXT_PAD_X, y + h - 18);
      }
    }

    // "+" tile
    const plusY = 6, plusH = trackH - 12, plusHover = this._hoverIsPlus === true;
    this._fillRoundRect(ctx, plusX, plusY, plusW, plusH, 6,
      plusHover ? COLORS.plusBgHover : COLORS.plusBg);
    ctx.strokeStyle = plusHover ? COLORS.plusBorderHover : COLORS.plusBorder;
    ctx.lineWidth = 1;
    this._strokeRoundRect(ctx, plusX + 0.5, plusY + 0.5, plusW - 1, plusH - 1, 6, ctx.strokeStyle);
    ctx.fillStyle = COLORS.plusFg;
    ctx.font = "500 22px Inter, -apple-system, system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("+", plusX + plusW / 2, plusY + plusH / 2 + 1);
    ctx.textAlign = "start"; ctx.textBaseline = "top";

    // Tick marks
    const tickY = trackH + 6;
    const stepSec = totalSec <= 12 ? 1 : (totalSec <= 30 ? 5 : 10);
    ctx.font = "400 11px Inter, system-ui, sans-serif"; ctx.textBaseline = "top";
    for (let s = 0; s <= totalSec + 0.001; s += stepSec) {
      const px = (s / totalSec) * (W - PLUS_TILE_PX - 8);
      ctx.fillStyle = (s === 0 || s === Math.round(totalSec)) ? COLORS.tickStrong : COLORS.tick;
      ctx.fillRect(px, trackH + 1, 1, 4);
      ctx.fillText(this._formatTickLabel(s), px + 4, tickY);
    }
    ctx.fillStyle = COLORS.tickStrong;
    ctx.font = "500 11px Inter, system-ui, sans-serif"; ctx.textAlign = "end";
    ctx.fillText(`total ${this._formatDuration(this._totalDuration)}`, W - 4, tickY);
    ctx.textAlign = "start";

    this.statsSegments.textContent = `${this._segments.length} segment${this._segments.length === 1 ? "" : "s"}`;
    this.statsTotal.textContent = `total ${this._formatDuration(this._totalDuration)} · ${Math.round(this._totalDuration * PRIORMDM_FPS)} frames`;
  }

  _fillRoundRect(ctx, x, y, w, h, r, fill) {
    ctx.fillStyle = fill; ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);         ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath(); ctx.fill();
  }
  _strokeRoundRect(ctx, x, y, w, h, r, stroke) {
    ctx.strokeStyle = stroke; ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);         ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath(); ctx.stroke();
  }
  _truncate(ctx, text, maxW) {
    if (ctx.measureText(text).width <= maxW) return text;
    let lo = 0, hi = text.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ctx.measureText(text.slice(0, mid) + "…").width <= maxW) lo = mid + 1; else hi = mid;
    }
    return text.slice(0, Math.max(0, lo - 1)) + "…";
  }
  _formatDuration(sec) {
    if (sec < 1)  return `${(sec * 1000).toFixed(0)} ms`;
    if (sec < 10) return `${sec.toFixed(1)} s`;
    return `${Math.round(sec)} s`;
  }
  _formatTickLabel(sec) {
    if (sec === 0) return "0";
    if (sec < 1)   return `${(sec * 1000).toFixed(0)}ms`;
    return `${Math.round(sec)}s`;
  }

  // ── Pointer interaction ──────────────────────────────────────────────────
  _canvasPoint(e) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  _onPointerMove(e) {
    if (this._dragKind) { this._continueDrag(e); return; }
    if (e.target !== this.canvas) {
      if (this._hoverId !== null || this._hoverIsPlus || this._hoverIsDelete) {
        this._hoverId = null; this._hoverIsResize = false;
        this._hoverIsPlus = false; this._hoverIsDelete = false;
        this.canvas.style.cursor = "default"; this._render();
      }
      return;
    }
    const { x, y } = this._canvasPoint(e);
    const hit = this._hitTest(x, y);
    let cursor = "default";
    let hId = null, hResize = false, hPlus = false, hDelete = false;
    if (hit.kind === "segment") {
      hId = hit.seg.id; hResize = hit.isResize;
      cursor = hit.isResize ? "col-resize" : "grab";
    } else if (hit.kind === "delete") { hId = hit.seg.id; hDelete = true; cursor = "pointer"; }
    else if (hit.kind === "plus")     { hPlus = true; cursor = "pointer"; }
    if (hId !== this._hoverId || hResize !== this._hoverIsResize
        || hPlus !== this._hoverIsPlus || hDelete !== this._hoverIsDelete) {
      this._hoverId = hId; this._hoverIsResize = hResize;
      this._hoverIsPlus = hPlus; this._hoverIsDelete = hDelete;
      this.canvas.style.cursor = cursor; this._render();
    }
  }

  _onPointerDown(e) {
    if (e.button === 2) return;
    e.stopPropagation();   // don't let litegraph drag the node out from under us
    const { x, y } = this._canvasPoint(e);
    const hit = this._hitTest(x, y);
    if (hit.kind === "plus")   { this._appendSegment(); return; }
    if (hit.kind === "delete") {
      this._deleteSegment(hit.seg.id);
      this._hoverIsDelete = false; this.canvas.style.cursor = "default"; return;
    }
    if (hit.kind !== "segment") { this._selectedId = null; this._render(); return; }
    this._selectedId = hit.seg.id; this._dragSegId = hit.seg.id; this._dragStartX = x;
    if (hit.isResize) {
      this._dragKind = "resize"; this._dragStartDuration = hit.seg.duration;
      this.canvas.style.cursor = "col-resize";
    } else {
      this._dragKind = "pending"; this.canvas.style.cursor = "grabbing";
    }
    try { this.canvas.setPointerCapture?.(e.pointerId); } catch {}
    this._render();
  }

  _continueDrag(e) {
    const { x } = this._canvasPoint(e);
    const dx = x - this._dragStartX;
    const { pxPerSec } = this._layoutSegments();
    const dSec = dx / pxPerSec;
    if (this._dragKind === "pending") {
      if (Math.abs(dx) < DRAG_THRESHOLD_PX) return;
      this._dragKind = "reorder";
    }
    if (this._dragKind === "resize") {
      const seg = this._segments.find((s) => s.id === this._dragSegId);
      if (!seg) return;
      let next = this._dragStartDuration + dSec;
      next = Math.max(MIN_SEG_SECONDS, Math.min(MAX_SEG_SECONDS, next));
      next = Math.min(next, this.opts.maxTotal - (this._totalSec() - seg.duration));
      seg.duration = next; this._render(); return;
    }
    if (this._dragKind === "reorder") {
      const idx = this._segments.findIndex((s) => s.id === this._dragSegId);
      if (idx < 0) return;
      const seg = this._segments[idx];
      if (dSec > 0 && idx + 1 < this._segments.length) {
        const next = this._segments[idx + 1];
        if (dSec > next.duration / 2) {
          this._segments.splice(idx, 1); this._segments.splice(idx + 1, 0, seg);
          this._dragStartX += next.duration * pxPerSec; this._emitChange();
        }
      } else if (dSec < 0 && idx > 0) {
        const prev = this._segments[idx - 1];
        if (-dSec > prev.duration / 2) {
          this._segments.splice(idx, 1); this._segments.splice(idx - 1, 0, seg);
          this._dragStartX -= prev.duration * pxPerSec; this._emitChange();
        }
      }
      this._render();
    }
  }

  _onPointerUp(e) {
    if (!this._dragKind) return;
    const wasPending = this._dragKind === "pending";
    const draggedSegId = this._dragSegId;
    if (this._dragKind === "resize") this._emitChange();
    this._dragKind = null; this._dragSegId = null;
    this.canvas.style.cursor = "default";
    try { this.canvas.releasePointerCapture?.(e.pointerId); } catch {}
    this._render();
    if (wasPending && draggedSegId != null) this._beginEdit(draggedSegId);
  }

  _onDblClick(e) {
    const { x, y } = this._canvasPoint(e);
    const hit = this._hitTest(x, y);
    if (hit.kind !== "segment") return;
    e.preventDefault(); e.stopPropagation();
    this._beginEdit(hit.seg.id);
  }

  _onContextMenu(e) {
    const { x, y } = this._canvasPoint(e);
    const hit = this._hitTest(x, y);
    if (hit.kind !== "segment") return;
    e.preventDefault(); e.stopPropagation();
    this._deleteSegment(hit.seg.id);
  }

  _onKeyDown(e) {
    if (this._editingId !== null) return;
    if (this._selectedId === null) return;
    const tag = ((e.target && e.target.tagName) || "").toUpperCase();
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target?.isContentEditable) return;
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault(); e.stopPropagation(); this._deleteSegment(this._selectedId);
    } else if (e.key === "Enter") {
      e.preventDefault(); e.stopPropagation(); this._beginEdit(this._selectedId);
    } else if (e.key === "Escape") {
      this._selectedId = null; this._render();
    }
  }

  // ── Mutations ────────────────────────────────────────────────────────────
  _showCapNote(msg) {
    this.statsHint.classList.add("aftl-stats-hint--warn");
    this.statsHint.textContent = msg;
    if (this._capNoteTimer) clearTimeout(this._capNoteTimer);
    this._capNoteTimer = setTimeout(() => {
      this.statsHint.classList.remove("aftl-stats-hint--warn");
      this.statsHint.innerHTML = this._defaultHintHTML;
      this._capNoteTimer = null;
    }, 3500);
  }

  _appendSegment() {
    const dur = _randomSegmentDuration();
    if (this._totalSec() + dur > this.opts.maxTotal) {
      this._showCapNote(`Timeline is capped at ${this.opts.maxTotal.toFixed(0)} s — shorten a segment to make room.`);
      return;
    }
    const prevPrompt = this._segments.length ? this._segments[this._segments.length - 1].prompt : null;
    const seg = {
      id: _newSegId(), prompt: _pickPromptDifferentFrom(prevPrompt),
      duration: dur, colorIdx: this._nextColorIdx,
    };
    this._nextColorIdx = (this._nextColorIdx + 1) % SEG_COLORS.length;
    this._segments.push(seg); this._selectedId = seg.id;
    this._emitChange(); this._render();
  }

  _deleteSegment(id) {
    if (this._segments.length <= this.opts.minSegments) return;
    this._segments = this._segments.filter((s) => s.id !== id);
    if (this._selectedId === id) this._selectedId = null;
    if (this._editingId === id) this._cancelEdit();
    this._emitChange(); this._render();
  }

  // ── Inline edit ──────────────────────────────────────────────────────────
  _beginEdit(id) {
    const seg = this._segments.find((s) => s.id === id);
    if (!seg) return;
    this._editingId = id;
    this.inlineEdit.value = seg.prompt;
    this.inlineEdit.hidden = false;
    this._positionInlineEdit();
    this.inlineEdit.focus();
    const len = this.inlineEdit.value.length;
    try { this.inlineEdit.setSelectionRange(len, len); } catch {}
    this._render();
  }

  _positionInlineEdit() {
    if (this._editingId === null) return;
    const { tiles } = this._layoutSegments();
    const tile = tiles.find((t) => t.seg.id === this._editingId);
    if (!tile) return;
    const trackH = (this.canvas.height / (window.devicePixelRatio || 1)) - TICK_PAD_BOTTOM;
    const segX = tile.x + 2, segW = Math.max(2, tile.w - 2), segY = 6, segH = trackH - 12;
    // The canvas sits below the .aftl-root top padding; offset the overlay to
    // match. clientTop of the canvas within root:
    const top = this.canvas.offsetTop;
    const left = this.canvas.offsetLeft;
    this.inlineEdit.style.left   = (left + segX + TEXT_PAD_X - 4) + "px";
    this.inlineEdit.style.top    = (top + segY + TEXT_TOP_Y - 4) + "px";
    this.inlineEdit.style.width  = Math.max(60, segW - TEXT_PAD_X * 2 + 8) + "px";
    this.inlineEdit.style.height = (segH - TEXT_TOP_Y - 8) + "px";
  }

  _commitEdit() {
    if (this._editingId === null) return;
    const seg = this._segments.find((s) => s.id === this._editingId);
    const next = this.inlineEdit.value.trim();
    if (seg && next.length > 0 && next !== seg.prompt) { seg.prompt = next; this._emitChange(); }
    this._cancelEdit();
  }

  _cancelEdit() {
    this._editingId = null; this.inlineEdit.hidden = true; this.inlineEdit.blur();
    this._render();
  }

  _emitChange() {
    this.dispatchEvent(new CustomEvent("change", { detail: { segments: this.getSegments() } }));
  }
}

// ── ComfyUI wiring ──────────────────────────────────────────────────────────
// segments_json in the node is [{prompt, num_frames}]; the editor works in
// seconds. Convert at the boundary. num_frames = round(duration * 20).
function segmentsFromJson(raw) {
  try {
    const arr = JSON.parse(raw || "[]");
    if (!Array.isArray(arr)) return null;
    const out = arr
      .filter((s) => s && typeof s.prompt === "string" && s.num_frames != null)
      .map((s) => ({ prompt: s.prompt, duration: Math.max(MIN_SEG_SECONDS, (+s.num_frames) / PRIORMDM_FPS) }));
    return out.length ? out : null;
  } catch { return null; }
}
function segmentsToJson(segs) {
  return JSON.stringify(segs.map((s) => ({
    prompt: s.prompt,
    num_frames: Math.max(1, Math.round(s.duration * PRIORMDM_FPS)),
  })));
}

app.registerExtension({
  name: "animoflow.timeline",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AnimoFlow_PriorMDMTimeline") return;
    ensureStyle();
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const jsonWidget = this.widgets?.find((w) => w.name === "segments_json");
      if (!jsonWidget) return;

      // Hide the raw JSON textarea — the timeline owns it now.
      jsonWidget.hidden = true;
      jsonWidget.computeSize = () => [0, -4];
      if (jsonWidget.inputEl) jsonWidget.inputEl.style.display = "none";

      const initial = segmentsFromJson(jsonWidget.value) || [
        { prompt: "a person walks forward", duration: 4.0 },
        { prompt: "the person turns and runs", duration: 4.0 },
      ];

      const wrap = document.createElement("div");
      wrap.style.cssText = "width:100%;height:100%;";
      const editor = new TimelineEditor(wrap, { initial });

      // Seed the widget so a queue without any interaction still round-trips.
      jsonWidget.value = segmentsToJson(editor.getSegments());
      editor.addEventListener("change", () => {
        jsonWidget.value = segmentsToJson(editor.getSegments());
        this.graph?.setDirtyCanvas(true, true);
      });

      // ComfyUI ≥1.45 sizes DOM widgets from getMinHeight/getMaxHeight (a
      // widget.computeSize override is ignored). The canvas is a fixed 110 px
      // tall + the stats row, so pin the box height — no width coupling here.
      const TL_WIDGET_H = 150;
      this.addDOMWidget("animoflow_timeline", "AF_TIMELINE", wrap, {
        serialize: false,
        getMinHeight: () => TL_WIDGET_H,
        getMaxHeight: () => TL_WIDGET_H,
      });
      const onRemoved = this.onRemoved;
      this.onRemoved = function () { editor.destroy(); onRemoved?.apply(this, arguments); };

      // Pre-fit the node to its computed size so the workflow's saved height
      // doesn't leave slack that ComfyUI clamps away on the first interaction
      // (the "shrinks when you click +" glitch).
      this.size[0] = Math.max(this.size[0], 420);
      const fit = this.computeSize();
      this.setSize([Math.max(420, fit[0]), fit[1]]);
    };
  },
});
