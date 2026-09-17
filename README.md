# Smart Grid ML Platform

A full-stack rebuild of the Smart Grid Machine Learning project:

- **Backend** — FastAPI serving the ML models and data processing pipeline as REST APIs.
- **Frontend** — Pure HTML / CSS / Vanilla JS dashboard (high-tech dark theme, glassmorphism).
- **No framework, no build step** — open-source-ready, easy to extend.

## Live URLs 🌐

| What | URL |
|------|-----|
| **Dashboard (frontend)** | https://frontendvercel-iota.vercel.app |
| **REST API (FastAPI Cloud)** | https://smartgrid.fastapicloud.dev |
| API interactive docs (`/docs`) | https://smartgrid.fastapicloud.dev/docs |

## 📓 The analysis notebook

`SmartGrid.ipynb` is the original data-science notebook behind this project —
68 cells covering the full pipeline: EDA, missing values, duplicates, unique
values, suspicious entries, IQR outlier analysis, correlation heatmaps,
time-based feature engineering, missing-value handling, feature selection and
preparation. GitHub renders `.ipynb` files natively in the browser, so you can
read the whole analysis without installing anything.

## Folder structure

```
Project ML/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, CORS, static frontend
│   │   ├── config.py          # paths + constants
│   │   ├── data_loader.py     # CSV loader + synthetic generator
│   │   ├── preprocessing.py   # feature engineering, missing values, one-hot, scaling
│   │   ├── models.py          # Logistic, Ridge, Random Forest, XGBoost training
│   │   ├── service.py         # cached pipeline + prediction service
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── serialize.py       # numpy/pandas -> JSON
│   │   └── routers/
│   │       ├── data.py        # /api/data/*        (info, targets, download)
│   │       ├── eda.py         # /api/eda/*         (missing, dupes, outliers, corr)
│   │       ├── models.py      # /api/models/*      (4 trained models)
│   │       └── predict.py     # /api/predict       (live predictions)
│   ├── data/                  # capstone_smartgrid_20000.csv
│   ├── requirements.txt
│   └── run.py                 # uvicorn dev server
├── frontend/
│   ├── index.html             # SPA shell (sidebar + views)
│   ├── css/style.css          # dark glassmorphism theme
│   └── js/
│       ├── api.js             # fetch wrapper for the REST API
│       ├── charts.js          # vanilla canvas charts (line, donut, bars, heatmap)
│       └── app.js             # view controller (dashboard, EDA, models, predict)
└── README.md
```

## Run it

```bash
# 1. backend
pip install -r backend/requirements.txt
cd backend
python run.py                 # -> http://127.0.0.1:8000

# 2. frontend (served automatically by FastAPI at /)
#    open http://127.0.0.1:8000
```

## REST API

| Method | Endpoint                 | Description                          |
|--------|--------------------------|--------------------------------------|
| GET    | `/api/health`            | Status + model list                  |
| GET    | `/api/data/info`         | Shape, dtypes, head sample           |
| GET    | `/api/data/targets`      | Target label distributions           |
| GET    | `/api/data/download`     | Download the original CSV            |
| GET    | `/api/eda/basic`         | Basic inspection                     |
| GET    | `/api/eda/missing`       | Missing values                       |
| GET    | `/api/eda/duplicates`    | Duplicate rows                       |
| GET    | `/api/eda/unique`        | Unique value counts                  |
| GET    | `/api/eda/suspicious`    | Negative / zero values               |
| GET    | `/api/eda/outliers`      | IQR outlier analysis                 |
| GET    | `/api/eda/correlation`   | Correlation matrix + high pairs      |
| GET    | `/api/models`            | Leaderboard of all 4 models          |
| GET    | `/api/models/{key}`      | Full detail for one model            |
| POST   | `/api/predict`           | Predict on raw CSV-style features    |

### Example prediction

```json
POST /api/predict
{
  "model": "logistic",
  "features": {
    "region": "SW",
    "building_type": "residential",
    "tariff_tier": "mid_peak",
    "consumption_kwh": 1.24,
    "hour": 11,
    "day_of_week": 3,
    "temp_c": 5.49
  }
}
```

Interactive API docs are available at `/docs`.

---

## The four ML models

The platform trains **four models** on the 20,000-row smart grid dataset, each
answering a different question the grid operator cares about.

| Model | Key | Task | Predicts | Business value |
|-------|-----|------|----------|----------------|
| **Logistic Regression** | `logistic` | Classification | `anomaly_flag` — is this reading anomalous? | Baseline, fast, and explainable. Flags suspicious meters/readings for review. |
| **Ridge Regression** | `ridge` | Regression | `next_hour_consumption_kwh` — kWh next hour | Load forecasting → capacity planning and demand-side management. |
| **Random Forest** | `random_forest` | Classification | `anomaly_flag` | More accurate anomalies with feature-importance insights (which drivers caused the flag). |
| **XGBoost** | `xgboost` | Classification | `high_usage_flag` — is consumption about to spike? | Early warning for peak-demand events → pricing and load-shedding decisions. |

### What each model actually does for the grid

- **Logistic Regression** is the *baseline classifier*: it learns a linear
  decision boundary between normal and anomalous readings. Because the dataset
  is imbalanced, the reported `probability` is the calibrated probability of
  the *positive (anomaly / high-usage)* class — the UI shows the complement as
  `P(Normal) = 1 − probability`.

- **Ridge Regression** is the *forecaster*: it maps weather + tariff + usage
  features to the **next hour's consumption**. It uses `L2` regularization, so
  it stays robust even when input features are correlated. The prediction is a
  continuous kWh value.

- **Random Forest** is the *robust anomaly detector*: an ensemble of decision
  trees that captures non-linear patterns Linear Regression can miss, and it
  is less sensitive to outliers/feature scaling. Good default for anomaly
  detection when you need more signal than the linear baseline.

- **XGBoost** is the *peak-spike warning*: gradient-boosted trees optimize for
  a stronger `high_usage_flag` (consumption much higher than expected). In the
  leaderboard (`/api/models`) you can compare it against the others by
  accuracy / precision / recall / F1.

### Feature boundary rules

Before any model sees your input, the API enforces the same bounds the models
were trained on (`backend/app/validation.py` is the single source of truth,
mirrored by the frontend form):

- Numeric ranges mirror the training data (`temp_c`, `humidity_pct`,
  `consumption_kwh`, `grid_price_usd_per_kwh`, `hour`, `day_of_week` …).
- `hour` and `day_of_week` must be whole numbers.
- `region` ∈ {`MW`, `NE`, `SE`, `SW`, `W`}, `building_type` ∈
  {`residential`, `commercial`, `industrial`}, `tariff_tier` ∈
  {`mid_peak`, `off_peak`, `on_peak`}.

Why enforce them? **Linear models extrapolate wildly outside their training
range** — e.g. Logistic Regression can report ~100% anomaly probability for an
impossible input. Restricting to dataset ranges keeps every prediction
physically meaningful.

### Example: model comparison (leaderboard)

`GET /api/models` returns the metrics for all four models so you can compare
them side by side:

```json
{
  "logistic": {
    "task": "anomaly_flag",
    "kind": "classification",
    "accuracy": 0.93,
    ...
  },
  "ridge": {
    "task": "next_hour_consumption_kwh",
    "kind": "regression",
    "r2": 0.87,
    ...
  }
}
```