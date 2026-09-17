// Lightweight vanilla canvas chart helpers (no dependencies).

const ChartTheme = {
  accent: "#2dd4bf",
  accent2: "#38bdf8",
  violet: "#a78bfa",
  amber: "#fbbf24",
  rose: "#fb7185",
  grid: "rgba(255,255,255,0.06)",
  text: "#8b97b3",
  axis: "#5b6883",
};

function setupCanvas(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const w = rect.width || 600;
  const h = cssHeight || parseInt(canvas.getAttribute("height") || "260", 10);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function roundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawLineChart(canvas, data, { yLabel = "", color = ChartTheme.accent, fill = true } = {}) {
  const { ctx, w, h } = setupCanvas(canvas, canvas.getAttribute("height") || 260);
  if (!data || data.length < 2) return;
  const pad = { l: 46, r: 16, t: 14, b: 30 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const min = Math.min(...data), max = Math.max(...data);
  const lo = min - (max - min) * 0.05, hi = max + (max - min) * 0.05;

  // grid + axis
  ctx.strokeStyle = ChartTheme.grid; ctx.lineWidth = 1; ctx.fillStyle = ChartTheme.text;
  ctx.font = "11px Inter, sans-serif";
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const y = pad.t + ih - (ih * i) / ticks;
    const val = lo + ((hi - lo) * i) / ticks;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(val.toFixed(1), 4, y + 4);
  }
  ctx.fillStyle = ChartTheme.text;
  ctx.fillText(yLabel, pad.l, h - 8);

  // line
  const step = iw / (data.length - 1);
  const pts = data.map((v, i) => [pad.l + i * step, pad.t + ih - ((v - lo) / (hi - lo)) * ih]);
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.stroke();
  if (fill) {
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);
    grad.addColorStop(0, hexA(color, 0.25));
    grad.addColorStop(1, hexA(color, 0));
    ctx.lineTo(pts[pts.length - 1][0], pad.t + ih);
    ctx.lineTo(pts[0][0], pad.t + ih);
    ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
  }
  // glow dots
  ctx.fillStyle = color;
  for (let i = 0; i < pts.length; i += Math.max(1, Math.floor(pts.length / 24))) {
    ctx.beginPath(); ctx.arc(pts[i][0], pts[i][1], 2, 0, Math.PI * 2); ctx.fill();
  }
}

function drawDonut(canvas, items, { colors = [] } = {}) {
  const { ctx, w, h } = setupCanvas(canvas, canvas.getAttribute("height") || 240);
  const cx = w / 2, cy = h / 2 - 6;
  const R = Math.min(w, h) / 2 - 34, rIn = R - 28;
  const total = items.reduce((s, x) => s + x.value, 0) || 1;
  let a0 = -Math.PI / 2;
  items.forEach((it, i) => {
    const a1 = a0 + (it.value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, R, a0, a1);
    ctx.closePath();
    ctx.fillStyle = colors[i % colors.length] || ChartTheme.accent;
    ctx.fill();
    a0 = a1;
  });
  ctx.beginPath(); ctx.arc(cx, cy, rIn, 0, Math.PI * 2); ctx.fillStyle = "#0d1424"; ctx.fill();
  ctx.fillStyle = "#fff"; ctx.font = "800 20px Inter, sans-serif"; ctx.textAlign = "center";
  ctx.fillText(total.toLocaleString(), cx, cy - 2);
  ctx.fillStyle = ChartTheme.text; ctx.font = "11px Inter, sans-serif";
  ctx.fillText("total", cx, cy + 16);
  ctx.textAlign = "left";

  // legend
  let ly = 6;
  items.forEach((it, i) => {
    ctx.fillStyle = colors[i % colors.length] || ChartTheme.accent;
    ctx.fillRect(6, ly, 9, 9);
    ctx.fillStyle = ChartTheme.text; ctx.font = "12px Inter, sans-serif";
    ctx.fillText(`${it.label}: ${it.value.toLocaleString()}`, 20, ly + 9);
    ly += 18;
  });
}

function drawBars(canvas, data, { horizontal = false, color = ChartTheme.accent } = {}) {
  const { ctx, w, h } = setupCanvas(canvas, canvas.getAttribute("height") || 220);
  if (!data.length) return;
  const valid = data.filter((d) => Number.isFinite(d.value));
  if (!valid.length) {
    ctx.fillStyle = ChartTheme.text; ctx.font = "12px Inter, sans-serif";
    ctx.fillText("No comparable metric", 16, 24);
    return;
  }
  const pad = { l: 60, r: 16, t: 34, b: 30 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const max = Math.max(1e-9, ...valid.map((d) => d.value));
  ctx.font = "11px Inter, sans-serif"; ctx.fillStyle = ChartTheme.text; ctx.textAlign = "right";

  const n = data.length;
  const slot = iw / n;
  const bw = Math.min(46, slot * 0.62);
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);
  grad.addColorStop(0, hexA(color, 0.9)); grad.addColorStop(1, hexA(color, 0.25));

  data.forEach((d, i) => {
    const value = Number.isFinite(d.value) ? Math.max(0, d.value) : 0;
    const bh = (value / max) * ih;
    const x = pad.l + i * slot + (slot - bw) / 2;
    const y = pad.t + ih - bh;
    ctx.fillStyle = grad;
    roundRectPath(ctx, x, y, bw, bh, 6); ctx.fill();
    const label = (d.text != null ? String(d.text) : shortNum(d.value));
    if (value > 0) {
      const labelY = Math.max(12, y - 8);
      ctx.textAlign = "center";
      ctx.font = "800 11px Inter, sans-serif";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(7,12,24,0.9)";
      ctx.strokeText(label, x + bw / 2, labelY);
      ctx.fillStyle = "#e6fdff";
      ctx.fillText(label, x + bw / 2, labelY);
      ctx.font = "11px Inter, sans-serif";
    }
    ctx.fillStyle = ChartTheme.text;
    ctx.fillText(shortLabel(d.label), x + bw / 2, h - 8);
    ctx.textAlign = "right";
  });
  ctx.fillStyle = ChartTheme.text; ctx.textAlign = "left";
}

function drawHeatmap(canvas, matrix, labels, { cellSize = 48 } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const n = matrix.length;
  if (!n) return;
  const pad = { l: 150, r: 24, t: 30, b: 190 };
  const w = n * cellSize + pad.l + pad.r;
  const h = Math.max(430, n * cellSize + pad.t + pad.b);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  if (canvas.parentElement) canvas.parentElement.style.width = w + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const ox = pad.l, oy = pad.t;
  const size = n * cellSize;

  // tiles
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) { continue; }
      ctx.fillStyle = corrColor(matrix[i][j]);
      ctx.fillRect(ox + j * cellSize, oy + i * cellSize, cellSize - 1, cellSize - 1);
    }
  }
  // numeric text (only when cells are big enough)
  if (cellSize >= 42) {
    ctx.fillStyle = "#0a0e18";
    ctx.font = "600 11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        ctx.fillText(matrix[i][j].toFixed(2), ox + j * cellSize + cellSize / 2, oy + i * cellSize + cellSize / 2);
      }
    }
  }

  // y-axis labels (horizontal, short)
  ctx.font = "11px Inter, sans-serif";
  ctx.fillStyle = ChartTheme.text;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  labels.forEach((lbl, i) => {
    ctx.fillText((lbl || "").length > 18 ? lbl.slice(0, 17) + "…" : lbl, ox - 10, oy + i * cellSize + cellSize / 2);
  });

  // x-axis labels (rotated -45°)
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  labels.forEach((lbl, i) => {
    const cx = ox + i * cellSize + cellSize / 2;
    const cy = oy + size + 12;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText((lbl || "").length > 18 ? lbl.slice(0, 17) + "…" : lbl, 0, 0);
    ctx.restore();
  });

  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

function drawHistogram(canvas, { labels = [], values = [], color = ChartTheme.accent2, colorStops = null } = {}) {
  const { ctx, w, h } = setupCanvas(canvas, canvas.getAttribute("height") || 260);
  if (!values.length) return;
  const pad = { l: 52, r: 14, t: 18, b: 46 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const max = Math.max(...values);
  const n = values.length;
  const slot = iw / n;
  const bw = Math.max(6, Math.min(34, slot * 0.7));
  const grad = colorStops || ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);

  values.forEach((v, i) => {
    const bh = (v / max) * ih;
    const x = pad.l + i * slot + (slot - bw) / 2;
    const y = pad.t + ih - bh;
    ctx.fillStyle = grad;
    roundRectPath(ctx, x, y, bw, bh, Math.min(4, bw / 2)); ctx.fill();
    if (slot > 22 && i % Math.max(1, Math.round(n / 22)) === 0) {
      ctx.fillStyle = ChartTheme.text; ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "center";
      ctx.fillText(String(labels[i]), x + bw / 2, h - 8);
    }
  });
  ctx.fillStyle = ChartTheme.text; ctx.font = "11px Inter, sans-serif"; ctx.textAlign = "left";
}

function corrColor(v) {
  const a = Math.min(1, Math.abs(v) * 1.6);
  if (v >= 0) return `rgba(45,212,191,${(0.12 + 0.75 * a).toFixed(3)})`;
  return `rgba(251,113,133,${(0.12 + 0.75 * a).toFixed(3)})`;
}

function hexA(hex, a) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

function shortNum(v) {
  if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return String(Math.round(v * 10) / 10);
}
function shortLabel(s) {
  s = String(s); return s.length > 14 ? s.slice(0, 13) + "…" : s;
}

window.Charts = { drawLineChart, drawDonut, drawBars, drawHeatmap, drawHistogram, ChartTheme };