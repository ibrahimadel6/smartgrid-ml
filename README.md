# Smart Grid ML Platform

A full-stack rebuild of the Smart Grid Machine Learning project:

- **Backend** — FastAPI serving the ML models and data processing pipeline as REST APIs.
- **Frontend** — Pure HTML / CSS / Vanilla JS dashboard (high-tech dark theme, glassmorphism).
- **No framework, no build step** — open-source-ready, easy to extend.

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