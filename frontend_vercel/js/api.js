// Minimal fetch wrapper for the Smart Grid API.
window.API = (() => {
  const BASE = "https://smartgrid.fastapicloud.dev";

  async function errorMessage(method, path, res) {
    let detail = "";
    try {
      const data = await res.json();
      const d = data && data.detail;
      if (typeof d === "string") detail = d;
      else if (d && Array.isArray(d.errors) && d.errors.length) detail = d.errors.join(" ");
      else if (d && d.message) detail = d.message;
      else if (d) detail = JSON.stringify(d);
      else detail = JSON.stringify(data);
    } catch {
      /* non-JSON error body */
    }
    return `${method} ${path} -> ${res.status}${detail ? `: ${detail}` : ""}`;
  }

  async function get(path, signal) {
    const res = await fetch(BASE + path, { signal });
    if (!res.ok) throw new Error(await errorMessage("GET", path, res));
    return res.json();
  }

  async function post(path, body, signal) {
    const res = await fetch(BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) throw new Error(await errorMessage("POST", path, res));
    return res.json();
  }

  const endpoints = {
    health: () => get("/api/health"),
    dataInfo: () => get("/api/data/info"),
    overview: () => get("/api/data/overview"),
    targets: () => get("/api/data/targets"),
    edaBasic: () => get("/api/eda/basic"),
    edaMissing: () => get("/api/eda/missing"),
    edaDuplicates: () => get("/api/eda/duplicates"),
    edaUnique: () => get("/api/eda/unique"),
    edaSuspicious: () => get("/api/eda/suspicious"),
    edaOutliers: () => get("/api/eda/outliers"),
    edaCorrelation: () => get("/api/eda/correlation"),
    modelsList: () => get("/api/models"),
    model: (key) => get(`/api/models/${key}`),
    // `signal` lets the caller abort a stale request when the model changes.
    predict: (model, features, signal) => post("/api/predict", { model, features }, signal),
  };

  return { get, post, endpoints };
})();
