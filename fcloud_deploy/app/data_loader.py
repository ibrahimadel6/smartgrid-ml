from functools import lru_cache

import numpy as np
import pandas as pd

from .config import CSV_PATH


def generate_synthetic(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic generator with the exact project schema."""
    rng = np.random.default_rng(seed)

    meter_ids = [f"M{i:03d}" for i in range(1, 41)]
    regions = ["NW", "NE", "SW", "SE", "C"]
    building_types = ["residential", "commercial", "industrial"]
    tariff_tiers = ["off_peak", "mid_peak", "peak"]

    start = pd.Timestamp("2025-01-01 00:00:00")
    timestamps = [start + pd.Timedelta(hours=i) for i in range(n_rows)]

    meter_id = rng.choice(meter_ids, n_rows)
    region = rng.choice(regions, n_rows)
    building_type = rng.choice(building_types, n_rows, p=[0.575, 0.28, 0.145])

    floor_area = rng.integers(72, 310, n_rows).astype(np.float64)
    insulation = rng.uniform(0.39, 0.89, n_rows)
    hvac_age = rng.uniform(0.0, 20.0, n_rows)
    solar_kw = rng.uniform(0.0, 6.5, n_rows)
    ev_charger = rng.integers(0, 2, n_rows)

    hour = rng.integers(0, 24, n_rows)
    day_of_week = rng.integers(0, 7, n_rows)
    is_weekend = rng.choice([0, 1], n_rows, p=[0.712, 0.288])
    is_holiday = rng.choice([0, 1], n_rows, p=[0.952, 0.048])
    tariff = rng.choice(tariff_tiers, n_rows, p=[0.32, 0.378, 0.302])

    temp = rng.normal(5.5, 4.6, n_rows)
    humidity = rng.uniform(15.0, 95.0, n_rows)
    wind = np.clip(rng.gamma(2.0, 1.5, n_rows), 0.01, 18.0)
    precip = rng.choice([0.0, 0.5, 1.0, 2.0, 4.0, 6.0], n_rows, p=[0.88, 0.04, 0.03, 0.02, 0.02, 0.01])
    grid_price = np.clip(rng.normal(0.168, 0.04, n_rows), 0.098, 0.268)
    occupancy = np.clip(rng.normal(0.52, 0.21, n_rows), 0.05, 1.0)
    solar_gen = np.where(hour >= 7, rng.gamma(2.0, 0.8, n_rows), 0.0)
    solar_gen = np.where(hour <= 18, solar_gen, 0.0).clip(0.0, 7.9)

    consumption = 0.35 + 0.015 * floor_area / 100 + 0.9 * rng.gamma(1.6, 1.0, n_rows)
    consumption = np.where(hour >= 9, consumption * 1.3, consumption)
    consumption = np.where((hour >= 17) & (hour <= 21), consumption * 1.45, consumption)
    consumption = np.where(day_of_week >= 5, consumption * 0.75, consumption)
    consumption = consumption * (1.0 - 0.35 * (solar_kw / 6.5))
    consumption = np.clip(consumption, 0.0, 30.7)

    high_usage = (consumption >= np.quantile(consumption, 0.90)).astype(int)

    anomaly_prob = np.clip(0.004 + 0.12 * (np.abs(consumption - np.median(consumption)) > 6).astype(float), 0.0, 0.5)
    anomaly = rng.random(n_rows) < anomaly_prob
    anomaly = anomaly.astype(int)

    sensor_health = np.clip(rng.normal(0.971, 0.016, n_rows), 0.85, 1.0)
    outage_risk = np.clip(0.03 + 0.5 * (1.0 - sensor_health) + 0.02 * rng.gamma(1.0, 1.0, n_rows), 0.0, 0.39)

    df = pd.DataFrame(
        {
            "meter_id": meter_id,
            "timestamp": timestamps,
            "region": region,
            "building_type": building_type,
            "floor_area_m2": floor_area,
            "insulation_rating": insulation.round(4),
            "hvac_age_years": hvac_age.round(4),
            "solar_kw_installed": solar_kw.round(4),
            "ev_charger": ev_charger,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "tariff_tier": tariff,
            "temp_c": temp.round(2),
            "humidity_pct": humidity.round(1),
            "wind_ms": wind.round(2),
            "precip_mm": precip,
            "grid_price_usd_per_kwh": grid_price.round(4),
            "occupancy_index": occupancy.round(3),
            "solar_generation_kwh": solar_gen.round(4),
            "consumption_kwh": consumption.round(4),
        }
    )

    lag1 = consumption.copy()
    lag1[1:] = consumption[:-1]
    df["lag1_kwh"] = np.where(np.arange(n_rows) == 0, np.nan, lag1).round(4)

    window = pd.Series(consumption).rolling(24, min_periods=1)
    df["roll24_mean_kwh"] = window.mean().to_numpy().round(4)
    lag24 = consumption.copy()
    lag24[:24] = np.nan
    df["lag24_kwh"] = np.where(np.arange(n_rows) < 24, np.nan, lag24).round(4)

    next_hour = consumption.copy()
    next_hour[:-1] = consumption[1:]
    next_hour[-1] = np.nan
    df["next_hour_consumption_kwh"] = np.where(np.arange(n_rows) == n_rows - 1, np.nan, next_hour).round(4)

    df["high_usage_flag"] = high_usage
    df["anomaly_flag"] = anomaly
    df["sensor_health"] = sensor_health.round(3)
    df["outage_risk_score"] = outage_risk.round(4)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


@lru_cache(maxsize=4)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_dataset(source: str = "csv", n_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    if source == "synthetic":
        return generate_synthetic(n_rows or 1000, seed)
    path = str(CSV_PATH)
    return load_csv(path)