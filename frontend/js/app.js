(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const API = window.API;
  const Charts = window.Charts;

  // ---- shared client-side rules (mirror backend validation.py) -------------
  const BOUNDS = {
    consumption_kwh: [0, 40],
    hour: [0, 23],
    day_of_week: [0, 6],
    temp_c: [-20, 60],
    humidity_pct: [0, 100],
    grid_price_usd_per_kwh: [0, 10],
    next_hour_consumption_kwh: [0, 40],
    outage_risk_score: [0, 1],
  };
  const INTEGER_FIELDS = new Set(["hour", "day_of_week"]);
  const CATEGORICAL = {
    region: ["MW", "NE", "SE", "SW", "W"],
    building_type: ["residential", "commercial", "industrial"],
    tariff_tier: ["mid_peak", "off_peak", "on_peak"],
  };
  const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

  const NUMERIC_FIELDS = [
    { key: "temp_c", label: "Temp (°C)", step: 0.1 },
    { key: "humidity_pct", label: "Humidity (%)", step: 0.1 },
    { key: "grid_price_usd_per_kwh", label: "Grid price ($/kWh)", step: 0.01 },
    { key: "consumption_kwh", label: "Consumption (kWh)", step: 0.1 },
    { key: "next_hour_consumption_kwh", label: "Next-hour usage (kWh)", step: 0.1 },
    { key: "outage_risk_score", label: "Outage risk", step: 0.01 },
  ];

  const state = {
    models: [],
    selectedModel: null,
    predictSeq: 0,
    predictAbort: null,
  };

  const DEFAULT_FEATURES = {
    consumption_kwh: 1.5,
    hour: 12,
    day_of_week: 0,
    temp_c: 10,
    humidity_pct: 55,
    grid_price_usd_per_kwh: 0.18,
    next_hour_consumption_kwh: 1.5,
    outage_risk_score: 0.11,
    region: "MW",
    building_type: "residential",
    tariff_tier: "mid_peak",
  };

  // ---- small helpers --------------------------------------------------------
  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }
  function fmt(v) {
    if (typeof v !== "number" || !isFinite(v)) return String(v ?? "");
    return v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  function pctStr(v) {
    return (isFinite(v) ? String(Math.round(v * 100) / 100) : "0") + "%";
  }
  function setStatus(ok, text) {
    const chip = $("#apiStatus");
    if (!chip) return;
    chip.classList.toggle("ok", ok);
    chip.classList.toggle("err", !ok);
    chip.innerHTML = `<span class="dot"></span>${esc(text)}`;
  }
  function modelName(key) {
    const m = state.models.find((x) => x.key === key);
    return m ? m.name : key;
  }
  function modelBadge(kind) {
    return kind === "classification" ? "Classification" : "Regression";
  }
  const TARGET_TITLES = {
    anomaly_flag: "Anomaly flag",
    high_usage_flag: "High usage flag",
    next_hour_consumption_kwh: "Next hour consumption",
  };
  function targetTitle(key) {
    return TARGET_TITLES[key] || String(key || "").replace(/_/g, " ");
  }

  // ---- nav -------------------------------------------------------------------
  function bindNav() {
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.addEventListener("click", () => switchView(item.dataset.view));
    });
    $("#refreshBtn").addEventListener("click", () => {
      loadHealth();
      const r = VIEWS[state.activeView] && VIEWS[state.activeView].render;
      if (r) r();
    });
  }

  const VIEWS = {
    dashboard: { title: "Dashboard", render: renderDashboard },
    eda: { title: "EDA", render: renderEDA },
    processing: { title: "Processing", render: renderProcessing },
    models: { title: "Models", render: renderModels },
    predict: { title: "Predict", render: renderPredict },
    data: { title: "Data", render: renderData },
  };

  function switchView(name) {
    state.activeView = name;
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.view === name));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
    const meta = VIEWS[name];
    if (meta) $("#pageTitle").textContent = meta.title;
    if (meta) meta.render();
  }

  // ---- health / models -------------------------------------------------------
  async function loadHealth() {
    try {
      const h = await API.endpoints.health();
      setStatus(true, `API online · ${h.shape[0].toLocaleString()} rows`);
    } catch (e) {
      setStatus(false, "API offline");
    }
  }

  async function loadModels() {
    const res = await API.endpoints.modelsList();
    state.models = res.models;
    return state.models;
  }

  // ============================================================ DASHBOARD
  async function renderDashboard() {
    try {
      const [ov, models] = await Promise.all([API.endpoints.overview(), state.models.length ? state.models : loadModels()]);
      state.models = models;

      const kpis = [
        { label: "Rows", value: fmt(ov.shape[0]), line: "meter readings" },
        { label: "Avg consumption", value: fmt(ov.avg_consumption) + " kWh", line: "per reading" },
        { label: "Avg next hour", value: fmt(ov.avg_next_hour) + " kWh", line: "per reading" },
        { label: "Anomalies", value: fmt(ov.anomaly.count), line: pctStr(ov.anomaly.percent) + " of readings" },
        { label: "High usage", value: fmt(ov.high_usage.count), line: pctStr(ov.high_usage.percent) + " of readings" },
        { label: "Period", value: fmt(ov.period.days) + " days", line: fmt(ov.period.hours) + " hours sampled" },
      ];
      $("#kpiGrid").innerHTML = kpis
        .map((k) => `<div class="kpi glass"><div class="k-label">${esc(k.label)}</div><div class="k-value">${esc(k.value)}</div><div class="k-line">${esc(k.line)}</div></div>`)
        .join("");

      Charts.drawLineChart($("#consumptionChart"), ov.hourly_profile.values, {
        yLabel: "Avg kWh by hour",
        color: Charts.ChartTheme.accent,
      });

      Charts.drawDonut(
        $("#mixChart"),
        ov.mix.building_type.map((b) => ({ label: b.label, value: b.value })),
        { colors: [Charts.ChartTheme.accent, Charts.ChartTheme.violet, Charts.ChartTheme.amber] }
      );

      Charts.drawBars(
        $("#leaderboardChart"),
        models.map((m) => ({ label: m.name, value: m.score, text: m.score.toFixed(4) })),
        { color: Charts.ChartTheme.accent }
      );
    } catch (e) {
      $("#kpiGrid").innerHTML = `<div class="placeholder">Failed to load dashboard: ${esc(e.message)}</div>`;
    }
  }

  // ============================================================ EDA
  async function renderEDA() {
    try {
      const [basic, missing, dup, uniq, susp, out, corr, targets] = await Promise.all([
        API.endpoints.edaBasic(),
        API.endpoints.edaMissing(),
        API.endpoints.edaDuplicates(),
        API.endpoints.edaUnique(),
        API.endpoints.edaSuspicious(),
        API.endpoints.edaOutliers(),
        API.endpoints.edaCorrelation(),
        API.endpoints.targets(),
      ]);

      const stats = [
        { b: fmt(basic.shape[0]), t: "rows" },
        { b: fmt(basic.shape[1]), t: "columns" },
        { b: fmt(basic.numeric_cols.length), t: "numeric" },
        { b: fmt(basic.categorical_cols.length), t: "categorical" },
        { b: fmt(missing.total), t: "missing values" },
        { b: fmt(dup.duplicate_rows), t: "duplicate rows" },
      ];
      $("#edaStatRow").innerHTML = stats
        .map((s) => `<div class="stat-chip"><b>${esc(s.b)}</b> ${esc(s.t)}</div>`)
        .join("");

      $("#edaDtypesTable").innerHTML =
        `<thead><tr><th>Column</th><th>Dtype</th></tr></thead><tbody>` +
        Object.entries(basic.dtypes)
          .map(([c, d]) => `<tr><td>${esc(c)}</td><td>${esc(d)}</td></tr>`)
          .join("") +
        `</tbody>`;

      $("#missingTable").innerHTML =
        `<thead><tr><th>Column</th><th>Missing</th><th>%</th></tr></thead><tbody>` +
        (missing.rows.length
          ? missing.rows.map((r) => `<tr><td>${esc(r.column)}</td><td>${fmt(r.missing)}</td><td>${pctStr(r.percent)}</td></tr>`).join("")
          : `<tr><td colspan="3">No missing values</td></tr>`) +
        `</tbody>`;

      $("#outlierTable").innerHTML =
        `<thead><tr><th>Column</th><th>Q1</th><th>Q3</th><th>IQR</th><th>Lower</th><th>Upper</th><th>Outliers</th><th>%</th></tr></thead><tbody>` +
        out.rows
          .slice(0, 12)
          .map(
            (r) =>
              `<tr><td>${esc(r.column)}</td><td>${fmt(r.q1)}</td><td>${fmt(r.q3)}</td><td>${fmt(r.iqr)}</td><td>${fmt(r.lower)}</td><td>${fmt(r.upper)}</td><td>${fmt(r.outliers)}</td><td>${pctStr(r.outlier_pct)}</td></tr>`
          )
          .join("") +
        `</tbody>`;

      renderTargetCards(targets);

      const holder = $("#corrHeatmap");
      holder.innerHTML = "";
      if (window.Plotly) {
        Plotly.newPlot(
          holder,
          [
            {
              type: "heatmap",
              z: corr.matrix,
              x: corr.columns,
              y: corr.columns,
              colorscale: [
                [0, "#0b1a2e"],
                [0.5, "#0d1424"],
                [1, "#2dd4bf"],
              ],
              zmin: -1,
              zmid: 0,
              zmax: 1,
              colorbar: { title: "corr", outlinewidth: 0 },
            },
          ],
          {
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "#0d1424",
            font: { color: "#8b97b3", family: "Inter, sans-serif", size: 11 },
            margin: { l: 120, r: 24, t: 24, b: 120, pad: 0 },
            autosize: true,
            xaxis: { tickangle: -45, tickfont: { size: 10 } },
            yaxis: { tickfont: { size: 10 } },
          },
          { displayModeBar: false, responsive: true }
        );
      } else {
        const canvas = document.createElement("canvas");
        holder.appendChild(canvas);
        Charts.drawHeatmap(canvas, corr.matrix, corr.columns, { cellSize: 42 });
      }

      $("#corrPairsTable").innerHTML =
        `<thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr></thead><tbody>` +
        (corr.high_pairs.length
          ? corr.high_pairs.map((p) => `<tr><td>${esc(p.feature_1)}</td><td>${esc(p.feature_2)}</td><td>${fmt(p.correlation)}</td></tr>`).join("")
          : `<tr><td colspan="3">No pairs ≥ 0.80</td></tr>`) +
        `</tbody>`;
    } catch (e) {
      $("#edaStatRow").innerHTML = `<div class="placeholder">Failed to load EDA: ${esc(e.message)}</div>`;
    }
  }

  function renderTargetCards(targets) {
    const box = $("#targetCards");
    box.innerHTML = "";
    const colors = [Charts.ChartTheme.accent, Charts.ChartTheme.rose, Charts.ChartTheme.amber, Charts.ChartTheme.violet];
    Object.entries(targets).forEach(([name, t], idx) => {
      const id = "tgt-" + idx + "-" + Math.random().toString(36).slice(2, 7);
      const card = document.createElement("div");
      card.className = "target-card";
      card.innerHTML = `<div class="target-title"><span>${esc(name)}</span> <span class="hl">${esc(t.kind)}</span></div><div class="target-chart" id="${id}"></div>`;
      box.appendChild(card);

      const el = document.getElementById(id);
      const color = colors[idx % colors.length];
      let data, layout;
      if (t.kind === "discrete") {
        data = [{
          type: "bar",
          x: t.items.map((it) => it.label === "0" ? "0" : it.label === "1" ? "1" : it.label),
          y: t.items.map((it) => it.value),
          marker: { color },
        }];
        layout = {
          paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#0d1424",
          font: { color: "#8b97b3", family: "Inter, sans-serif" },
          margin: { t: 8, r: 8, b: 26, l: 38 },
          xaxis: { title: "class" }, yaxis: { title: "count" },
        };
      } else {
        data = [{
          type: "histogram",
          x: (() => {
            const h = t.histogram;
            const out = [];
            for (let i = 0; i < h.counts.length; i++) {
              for (let k = 0; k < h.counts[i]; k++) out.push((h.bin_edges[i] + h.bin_edges[i + 1]) / 2);
            }
            return out;
          })(),
          nbinsx: Math.min(40, t.histogram.counts.length),
          marker: { color },
        }];
        layout = {
          paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#0d1424",
          font: { color: "#8b97b3", family: "Inter, sans-serif" },
          margin: { t: 8, r: 8, b: 26, l: 44 },
          xaxis: { title: name }, yaxis: { title: "count" },
        };
      }
      if (window.Plotly) Plotly.newPlot(el, data, layout, { displayModeBar: false });
    });
  }

  // ============================================================ PROCESSING
  async function renderProcessing() {
    const cols = ["consumption_kwh", "hour", "day_of_week", "temp_c", "humidity_pct", "grid_price_usd_per_kwh"];
    try {
      const [info, missing, dup] = await Promise.all([API.endpoints.dataInfo(), API.endpoints.edaMissing(), API.endpoints.edaDuplicates()]);
      const p = [
        { t: "Raw shape", v: `${fmt(info.shape[0])} × ${fmt(info.shape[1])}`, d: "rows × columns" },
        { t: "Missing values", v: fmt(missing.total), d: "imputed median / mode" },
        { t: "Duplicate rows", v: fmt(dup.duplicate_rows), d: "dropped in cleaning" },
        { t: "Numeric columns", v: fmt(cols.length), d: "after selection" },
      ];
      $("#processCards").innerHTML = p
        .map((c) => `<div class="card glass"><h3>${esc(c.t)}</h3><div class="placeholder" style="font-size:22px;font-weight:700;color:var(--text)">${esc(c.v)}</div><div style="color:var(--text-dim);font-size:12.5px">${esc(c.d)}</div></div>`)
        .join("");

      $("#encodedTable").innerHTML =
        `<thead><tr>` + info.columns.slice(0, 10).map((c) => `<th>${esc(c)}</th>`).join("") + `</tr></thead><tbody>` +
        info.head.slice(0, 6).map((r) => `<tr>` + info.columns.slice(0, 10).map((c) => `<td>${esc(r[c])}</td>`).join("") + `</tr>`).join("") +
        `</tbody>`;
    } catch (e) {
      $("#processCards").innerHTML = `<div class="placeholder">Failed to load processing view: ${esc(e.message)}</div>`;
    }
  }

  // ============================================================ MODELS
  async function renderModels() {
    try {
      const models = state.models.length ? state.models : await loadModels();
      state.models = models;
      const grid = $("#modelGrid");
      grid.innerHTML = models
        .map(
          (m) => `<div class="model-card glass ${state.selectedModel === m.key ? "active" : ""}" data-key="${esc(m.key)}">
            <div class="mc-badge" style="color:${m.kind === "classification" ? Charts.ChartTheme.accent : Charts.ChartTheme.amber}">${esc(modelBadge(m.kind))}</div>
            <h3>${esc(m.name)}</h3>
            <div class="mc-target">target: <b>${esc(targetTitle(m.target))}</b></div>
            <div>Score (<b>${esc(m.score_label)}</b>): <b style="color:var(--accent)">${fmt(m.score)}</b></div>
          </div>`
        )
        .join("");
      grid.querySelectorAll(".model-card").forEach((card) => {
        card.addEventListener("click", () => selectModel(card.dataset.key, card));
      });
      if (state.selectedModel) {
        await renderModelDetail(state.selectedModel);
      } else if (models.length) {
        await selectModel(models[0].key, grid.querySelector(`[data-key="${models[0].key}"]`));
      }
    } catch (e) {
      $("#modelGrid").innerHTML = `<div class="placeholder">Failed to load models: ${esc(e.message)}</div>`;
    }
  }

  async function selectModel(key, card) {
    state.selectedModel = key;
    document.querySelectorAll("#modelGrid .model-card").forEach((c) => c.classList.remove("active"));
    if (card) card.classList.add("active");
    await renderModelDetail(key);
  }

  async function renderModelDetail(key) {
    const box = $("#modelDetail");
    if (!box) return;
    box.innerHTML = `<div class="placeholder">Loading ${esc(key)}…</div>`;
    let m;
    try {
      m = await API.endpoints.model(key);
    } catch (e) {
      box.innerHTML = `<div class="placeholder">Failed to load model: ${esc(e.message)}</div>`;
      return;
    }

    const metricChips = Object.entries(m.metrics || {})
      .map(([k, v]) => `<div class="stat-chip"><b>${esc(k.toUpperCase())}</b> ${fmt(v)}</div>`)
      .join("");

    let cmHtml = "";
    if (m.cm_flat) {
      const c = m.cm_flat;
      cmHtml = `<h3 style="margin-top:18px">Confusion matrix (test)</h3>
        <table class="glass-table"><thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead><tbody>
        <tr><td><b>Actual 0</b></td><td>${fmt(c.TN)}</td><td>${fmt(c.FP)}</td></tr>
        <tr><td><b>Actual 1</b></td><td>${fmt(c.FN)}</td><td>${fmt(c.TP)}</td></tr></tbody></table>
        <div class="stat-row" style="margin-top:10px">
          <div class="stat-chip"><b>Precision</b> ${fmt(c.TP / (c.TP + c.FP || 1))}</div>
          <div class="stat-chip"><b>Recall</b> ${fmt(c.TP / (c.TP + c.FN || 1))}</div>
          <div class="stat-chip"><b>F1</b> ${fmt((2 * c.TP) / (2 * c.TP + c.FP + c.FN || 1))}</div>
        </div>`;
    }

    let regHtml = "";
    if (m.train_r2 != null) {
      regHtml = `<h3 style="margin-top:18px">Regression fit</h3>
        <div class="stat-row">
          <div class="stat-chip"><b>Train R²</b> ${fmt(m.train_r2)}</div>
          <div class="stat-chip"><b>Test R²</b> ${fmt(m.test_r2)}</div>
          <div class="stat-chip"><b>Mean error</b> ${fmt(m.mean_error)}</div>
          <div class="stat-chip"><b>Median error</b> ${fmt(m.median_error)}</div>
        </div>`;
    }

    let impHtml = "";
    if (m.top10 && m.top10.length) {
      impHtml = `<h3 style="margin-top:18px">Feature importance (top ${m.top10.length})</h3>` +
        `<div style="max-width:560px"><canvas id="importanceChart" height="260"></canvas></div>`;
    }

    box.innerHTML = `
      <h3>${esc(m.name)} <span style="color:var(--text-dim);font-weight:400;font-size:12.5px">(${esc(m.kind)})</span></h3>
      <div class="mc-target">target: <b>${esc(targetTitle(m.target))}</b></div>
      <div class="stat-row">${metricChips || `<div class="stat-chip">No metrics</div>`}</div>
      ${cmHtml}
      ${regHtml}
      ${impHtml}
      ${m.verdict ? `<h3 style="margin-top:18px">Verdict</h3><div class="placeholder">${esc(m.verdict)}</div>` : ""}
      `;
    if (m.top10 && m.top10.length) {
      Charts.drawBars(
        $("#importanceChart"),
        m.top10.map((r) => ({ label: r.Feature, value: r.Importance, text: r.Importance.toFixed(3) })),
        { color: Charts.ChartTheme.violet }
      );
    }
  }

  // ============================================================ PREDICT
  async function renderPredict() {
    const modelKeys = state.models.length ? state.models : await loadModels();
    state.models = modelKeys;

    const form = $("#predictForm");
    const prev = state.selectedModel;
    form.innerHTML = "";

    const modelSel = makeSelect("predictModel", null, modelKeys.map((m) => ({ value: m.key, label: `${m.name}  ·  ${targetTitle(m.target)}` })));
    wrapField(form, "Model", modelSel);

    NUMERIC_FIELDS.forEach((f) => {
      const [lo, hi] = BOUNDS[f.key];
      wrapField(form, f.label, makeNumber(f.key, DEFAULT_FEATURES[f.key], lo, hi, f.step));
    });

    wrapField(form, "Hour (0-23)", makeSelect("hour", "12", Array.from({ length: 24 }, (_, i) => ({ value: String(i), label: String(i) }))));
    wrapField(form, "Day of week", makeSelect("day_of_week", "0", WEEKDAYS.map((w, i) => ({ value: String(i), label: w }))));
    wrapField(form, "Region", makeSelect("region", "MW", CATEGORICAL.region.map((v) => ({ value: v, label: v }))));
    wrapField(form, "Building type", makeSelect("building_type", "residential", CATEGORICAL.building_type.map((v) => ({ value: v, label: v }))));
    wrapField(form, "Tariff tier", makeSelect("tariff_tier", "mid_peak", CATEGORICAL.tariff_tier.map((v) => ({ value: v, label: v }))));

    if (prev && modelKeys.some((m) => m.key === prev)) modelSel.value = prev;
    state.selectedModel = modelSel.value;

    modelSel.addEventListener("change", () => {
      state.selectedModel = modelSel.value;
      clearPredictResult();
    });

    const btn = $("#predictBtn");
    btn.onclick = null;
    btn.addEventListener("click", runPredict);
  }

  function makeSelect(id, value, options) {
    const s = document.createElement("select");
    s.id = id;
    options.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o.value;
      opt.textContent = o.label;
      s.appendChild(opt);
    });
    s.value = value;
    return s;
  }

  function makeNumber(key, value, lo, hi, step) {
    const input = document.createElement("input");
    input.type = "number";
    input.id = key;
    input.value = value;
    input.step = step;
    input.min = lo;
    input.max = hi;
    return input;
  }

  function wrapField(form, label, input) {
    const f = document.createElement("div");
    f.className = "form-field";
    const l = document.createElement("label");
    l.textContent = label;
    f.appendChild(l);
    f.appendChild(input);
    form.appendChild(f);
  }

  function validateLocal(features) {
    const errors = [];
    for (const [key, value] of Object.entries(features)) {
      if (value == null || value === "") continue;
      if (key in BOUNDS) {
        const num = Number(value);
        if (!isFinite(num)) { errors.push(`'${key}' must be a number.`); continue; }
        if (INTEGER_FIELDS.has(key) && num !== Math.trunc(num)) { errors.push(`'${key}' must be a whole number.`); continue; }
        const [lo, hi] = BOUNDS[key];
        if (num < lo || num > hi) errors.push(`'${key}' must be between ${lo} and ${hi}, got ${num}.`);
      }
      if (key in CATEGORICAL && !CATEGORICAL[key].includes(value)) errors.push(`'${key}' must be one of: ${CATEGORICAL[key].join(", ")}.`);
    }
    return errors;
  }

  function collectFeatures() {
    const features = { ...DEFAULT_FEATURES };
    document.querySelectorAll("#predictForm [id]").forEach((el) => {
      if (el.id === "predictModel") return;
      if (el.id in BOUNDS) features[el.id] = el.value === "" ? null : Number(el.value);
      else features[el.id] = el.value;
    });
    return features;
  }

  function clearPredictResult() {
    state.predictSeq++;
    if (state.predictAbort) state.predictAbort.abort();
    const body = $("#predictResultBody");
    if (body) body.innerHTML = `<div class="placeholder">Pick a model, adjust features, and run.</div>`;
  }

  function setResultLoading(modelKey) {
    const body = $("#predictResultBody");
    if (body) body.innerHTML = `<div class="placeholder">Running ${esc(modelName(modelKey))}…</div>`;
  }

  function setResultError(msg) {
    const body = $("#predictResultBody");
    if (body) body.innerHTML = `<div class="placeholder" style="color:var(--rose,#fb7185)">${esc(msg)}</div>`;
  }

  async function runPredict() {
    const model = state.selectedModel;
    if (!model) return;
    const features = collectFeatures();

    const errors = validateLocal(features);
    if (errors.length) {
      setResultError("Invalid input: " + errors.join(" "));
      return;
    }

    const seq = ++state.predictSeq;
    if (state.predictAbort) state.predictAbort.abort();
    const ctrl = new AbortController();
    state.predictAbort = ctrl;
    setResultLoading(model);

    try {
      const res = await API.endpoints.predict(model, features, ctrl.signal);
      if (seq !== state.predictSeq) return;
      if (res.model_key !== state.selectedModel) return;
      renderPredictResult(res);
    } catch (e) {
      if (e && e.name === "AbortError") return;
      if (seq !== state.predictSeq) return;
      setResultError(e && e.message ? e.message : "Prediction failed.");
    }
  }

  function renderPredictResult(res) {
    const body = $("#predictResultBody");
    if (!body) return;
    const m = state.models.find((x) => x.key === res.model_key);
    const chip = m && m.score != null ? `<div class="stat-chip" style="margin-top:10px"><b>${esc(m.score_label)}</b> ${fmt(m.score)} · test-time evaluation</div>` : "";

    let main;
    if (res.kind === "classification") {
      const isPos = res.prediction === 1;
      const labelColor = isPos ? "#fb7185" : "#2dd4bf";
      const labelText = isPos ? (res.label || "Positive") : "Normal";
      const pPos = Math.max(0, Math.min(1, res.probability || 0));
      const pNorm = 1 - pPos;
      const shownP = isPos ? pPos : pNorm;
      const otherP = isPos ? pNorm : pPos;
      const otherText = isPos ? "Normal" : (res.label || "Positive");
      main = `<div style="display:flex;align-items:baseline;gap:10px;margin-top:6px">
        <span style="font-size:40px;font-weight:800;color:${labelColor}">${esc(labelText)}</span>
        <span style="color:var(--text-dim);font-size:13px">class ${fmt(res.prediction)}</span>
      </div>
      <div style="margin-top:14px">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-dim);margin-bottom:4px">
          <span>P(${esc(labelText)})</span><span>${pctStr(shownP * 100)}</span>
        </div>
        <div style="width:100%;background:rgba(255,255,255,0.08);border-radius:8px;overflow:hidden">
          <div style="width:${(shownP * 100).toFixed(1)}%;height:8px;background:${labelColor}"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-dim);margin-top:6px">
          <span>P(${esc(otherText)})</span><span>${pctStr(otherP * 100)}</span>
        </div>
      </div>`;
    } else {
      main = `<div style="display:flex;align-items:baseline;gap:8px;margin-top:6px">
        <span style="font-size:40px;font-weight:800;color:var(--accent)">${fmt(res.prediction)}</span>
        <span style="color:var(--text-dim);font-size:14px">kWh</span>
      </div>`;
    }

    body.innerHTML = `
      <div style="color:var(--text-dim);font-size:13px">${esc(modelName(res.model_key))} · ${esc(targetTitle(res.task))} · ${esc(modelBadge(res.kind))}</div>
      ${main}
      ${chip}
      <div class="stat-row" style="margin-top:12px">
        <div class="stat-chip"><b>Model key</b> ${esc(res.model_key)}</div>
        <div class="stat-chip"><b>Target</b> ${esc(targetTitle(res.task))}</div>
      </div>`;
  }

  // ============================================================ DATA
  async function renderData() {
    try {
      const info = await API.endpoints.dataInfo();
      $("#dataStatRow").innerHTML = [
        { b: fmt(info.shape[0]), t: "rows" },
        { b: fmt(info.shape[1]), t: "columns" },
      ]
        .map((s) => `<div class="stat-chip"><b>${esc(s.b)}</b> ${esc(s.t)}</div>`)
        .join("");

      $("#dataTable").innerHTML =
        `<thead><tr>` + info.columns.map((c) => `<th>${esc(c)}</th>`).join("") + `</tr></thead><tbody>` +
        info.head
          .map(
            (r) =>
              `<tr>` + info.columns.map((c) => `<td>${esc(r[c])}</td>`).join("") + `</tr>`
          )
          .join("") +
        `</tbody>`;
    } catch (e) {
      $("#dataStatRow").innerHTML = `<div class="placeholder">Failed to load data: ${esc(e.message)}</div>`;
    }
  }

  // ---- boot -------------------------------------------------------------------
  async function init() {
    bindNav();
    setStatus(false, "Connecting…");
    try {
      await Promise.all([loadHealth(), loadModels()]);
      renderDashboard();
    } catch (e) {
      setStatus(false, "API offline");
    }
  }

  init();
})();