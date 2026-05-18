"""
Climate Data Fetcher — Open-Meteo (Free, No Auth Required)

Fetches real climate data for Kenya counties given lat/lon and date range.
Returns aggregated monthly or annual climate statistics.
Open-Meteo provides free historical and weather data without API keys.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import json
from pathlib import Path
import streamlit as st

# ── County Centroids ─────────────────────────────────────────────────
def load_county_coords() -> Dict[str, Tuple[float, float]]:
    """Load Kenya county centroids from JSON file."""
    coords_file = Path(__file__).resolve().parent.parent / "data" / "county_centroids.json"
    with open(coords_file, "r") as f:
        coords_dict = json.load(f)
    # Convert lists to tuples (lat, lon)
    return {county: tuple(coords) for county, coords in coords_dict.items()}

COUNTY_COORDS = load_county_coords()


def get_county_coords(county: str) -> Optional[Tuple[float, float]]:
    """Get (latitude, longitude) for a county. Returns (lat, lon) tuple or None."""
    return COUNTY_COORDS.get(county.title())


# ── Open-Meteo API ──────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_open_meteo(
    county: str,
    year: int,
    month: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch monthly climate data from Open-Meteo Archive for a county.
    
    Args:
        county: Kenyan county name
        year: Year (1940-present available)
        month: Optional; if provided, returns only that month
    
    Returns:
        DataFrame with columns: month, temperature_c, min_temp_c, max_temp_c, rainfall_mm
        or None if fetch fails
    """
    coords = get_county_coords(county)
    if not coords:
        return None
    
    lat, lon = coords
    
    # Open-Meteo Archive API (free, no auth required)
    # https://open-meteo.com/en/docs/historical-weather-api
    start_date = f"{year:04d}-01-01"
    
    # For current year, don't request future dates (API will reject)
    from datetime import datetime
    current_year = datetime.now().year
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    if year == current_year:
        end_date = current_date  # Up to today
    else:
        end_date = f"{year:04d}-12-31"  # Full year for past years
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
        "timezone": "Africa/Nairobi",
    }
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if "daily" not in data:
            return None
        
        daily = data["daily"]
        
        # Convert daily data to monthly aggregates
        dates = pd.to_datetime(daily["time"])
        df = pd.DataFrame({
            "date": dates,
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "temp_mean": daily["temperature_2m_mean"],
            "rainfall": daily["precipitation_sum"],
        })
        
        # Aggregate to monthly
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        
        monthly_rows = []
        for (y, m), group in df.groupby(["year", "month"]):
            if y != year:
                continue
            
            monthly_rows.append({
                "month": int(m),
                "year": int(y),
                "temperature_c": round(group["temp_mean"].mean(), 2),
                "min_temp_c": round(group["temp_min"].min(), 2),
                "max_temp_c": round(group["temp_max"].max(), 2),
                "rainfall_mm": round(group["rainfall"].sum(), 2),
            })
        
        if not monthly_rows:
            return None
        
        monthly_df = pd.DataFrame(monthly_rows).sort_values("month").reset_index(drop=True)
        
        # Filter by month if requested
        if month is not None:
            monthly_df = monthly_df[monthly_df["month"] == month]
        
        return monthly_df if len(monthly_df) > 0 else None
    
    except Exception as e:
        print(f"Open-Meteo fetch failed for {county} {year}: {e}")
        return None


# ── Feature Computation from Raw Climate Data ────────────────────────
def compute_features_option_a(monthly_data: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Compute annual climate features for Option A (annual aggregation).
    
    Requires: monthly_data with columns [month, year, temperature_c, min_temp_c, max_temp_c, rainfall_mm]
    
    Returns dict with keys matching FEATURES_A:
      - total_rainfall_mm
      - mean_rainfall_mm
      - max_rainfall_mm
      - rainfall_variability (std dev)
      - mean_temperature_c
      - max_temperature_c
      - min_temperature_c
      - temp_range_c
      - peak_rainfall_month
    """
    if monthly_data is None or len(monthly_data) == 0:
        return None
    
    try:
        total_rainfall = monthly_data["rainfall_mm"].sum()
        mean_rainfall = monthly_data["rainfall_mm"].mean()
        max_rainfall = monthly_data["rainfall_mm"].max()
        rainfall_variability = monthly_data["rainfall_mm"].std()
        
        mean_temperature = monthly_data["temperature_c"].mean()
        max_temperature = monthly_data["max_temp_c"].max()
        min_temperature = monthly_data["min_temp_c"].min()
        temp_range = max_temperature - min_temperature
        
        # Peak rainfall month (which month had most rainfall)
        peak_month = int(monthly_data.loc[monthly_data["rainfall_mm"].idxmax(), "month"])
        
        return {
            "total_rainfall_mm": round(total_rainfall, 2),
            "mean_rainfall_mm": round(mean_rainfall, 2),
            "max_rainfall_mm": round(max_rainfall, 2),
            "rainfall_variability": round(rainfall_variability, 2),
            "mean_temperature_c": round(mean_temperature, 2),
            "max_temperature_c": round(max_temperature, 2),
            "min_temperature_c": round(min_temperature, 2),
            "temp_range_c": round(temp_range, 2),
            "peak_rainfall_month": peak_month,
        }
    except Exception as e:
        print(f"Error computing Option A features: {e}")
        return None


def compute_features_option_b(
    monthly_data: pd.DataFrame,
    month: int,
) -> Optional[Dict[str, float]]:
    """
    Compute monthly climate features for Option B (with lagged aggregates).
    
    For a given month within the year, compute:
      - avg_rainfall_mm (current month)
      - avg_rainfall_mm_lag1, lag2 (previous months)
      - avg_rainfall_mm_roll3 (3-month rolling average)
      - Same for temperature
      - month_sin, month_cos (seasonality encoding)
    """
    import math
    
    if monthly_data is None or len(monthly_data) == 0:
        return None
    
    try:
        # Ensure data is sorted by month
        df = monthly_data.sort_values("month").reset_index(drop=True)
        
        # Find current month row
        current_row_idx = None
        for idx, row in df.iterrows():
            if row["month"] == month:
                current_row_idx = idx
                break
        
        if current_row_idx is None:
            return None
        
        current_row = df.iloc[current_row_idx]
        
        # Current values
        avg_rainfall = current_row["rainfall_mm"]
        mean_temp = current_row["temperature_c"]
        
        # Lag-1 (previous month)
        rainfall_lag1 = df.iloc[current_row_idx - 1]["rainfall_mm"] if current_row_idx >= 1 else avg_rainfall
        temp_lag1 = df.iloc[current_row_idx - 1]["temperature_c"] if current_row_idx >= 1 else mean_temp
        
        # Lag-2 (2 months ago)
        rainfall_lag2 = df.iloc[current_row_idx - 2]["rainfall_mm"] if current_row_idx >= 2 else rainfall_lag1
        temp_lag2 = df.iloc[current_row_idx - 2]["temperature_c"] if current_row_idx >= 2 else temp_lag1
        
        # 3-month rolling average
        start_idx = max(0, current_row_idx - 2)
        rainfall_roll3 = df.iloc[start_idx:current_row_idx + 1]["rainfall_mm"].mean()
        temp_roll3 = df.iloc[start_idx:current_row_idx + 1]["temperature_c"].mean()
        
        # Seasonality encoding
        month_sin = round(math.sin(2 * math.pi * month / 12), 6)
        month_cos = round(math.cos(2 * math.pi * month / 12), 6)
        
        return {
            "avg_rainfall_mm": round(avg_rainfall, 2),
            "avg_rainfall_mm_lag1": round(rainfall_lag1, 2),
            "avg_rainfall_mm_lag2": round(rainfall_lag2, 2),
            "avg_rainfall_mm_roll3": round(rainfall_roll3, 2),
            "mean_temperature_celcius": round(mean_temp, 2),
            "mean_temperature_celcius_lag1": round(temp_lag1, 2),
            "mean_temperature_celcius_lag2": round(temp_lag2, 2),
            "mean_temperature_celcius_roll3": round(temp_roll3, 2),
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
    except Exception as e:
        print(f"Error computing Option B features: {e}")
        return None


# ── Main wrapper function ────────────────────────────────────────────
def fetch_and_compute_features(
    county: str,
    year: int,
    option: str,
    month: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """
    Fetch climate data for a county/year and compute model features.
    
    Args:
        county: Kenyan county name
        year: Year (1940-present available from Open-Meteo)
        option: "a" (annual) or "b" (monthly with lags)
        month: Required if option == "b"; optional for "a"
    
    Returns:
        Dict of feature values ready for model prediction, or None if fetch/compute fails.
    """
    monthly_data = fetch_open_meteo(county, year)
    
    if monthly_data is None:
        return None
    
    if option == "a":
        return compute_features_option_a(monthly_data)
    elif option == "b":
        if month is None:
            month = 6  # Default to June if not specified
        return compute_features_option_b(monthly_data, month)
    
    return None
