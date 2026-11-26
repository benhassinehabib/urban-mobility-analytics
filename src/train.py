"""
Reproducible training pipeline for Urban Mobility Analytics.

Runs the full workflow from raw CSV files to saved models and reports:

    1. Load the raw Washington DC and Seoul datasets
    2. Clean and rename columns
    3. Engineer time / calendar features
    4. Split each dataset chronologically (first 80% train, last 20% test)
    5. Train a Linear Regression baseline and a Random Forest per city
    6. Evaluate with R², MAE and RMSE
    7. Save models (models/) and results (reports/)

Usage (from the project root):

    python src/train.py

The preprocessing mirrors notebooks/01_data_understanding.ipynb so the
results match the notebook exactly.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "bike_rentals"
TEST_SHARE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def get_season(month: int) -> str:
    """Return the meteorological season for a month number (1-12)."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekday, weekend flag, month, year and season derived from `date`."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.weekday          # 0 = Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month_from_date"] = df["date"].dt.month
    df["year_from_date"] = df["date"].dt.year
    df["season_from_date"] = df["month_from_date"].apply(get_season)
    return df


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_washington(path: Path) -> pd.DataFrame:
    """Load and prepare the Capital Bikeshare (Washington DC) hourly dataset.

    `casual` and `registered` are dropped because they add up to the target
    and would leak the answer into the model.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={
        "dteday": "date",
        "yr": "year",                 # 0 = 2011, 1 = 2012
        "mnth": "month",
        "hr": "hour",
        "weathersit": "weather",
        "temp": "temperature",        # normalised: (t + 8) / 47
        "atemp": "feels_like_temperature",  # normalised: (t + 16) / 66
        "hum": "humidity",            # normalised: % / 100
        "windspeed": "wind_speed",    # normalised: km/h / 67
        "cnt": TARGET,
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["instant", "casual", "registered"])
    df = add_date_features(df)
    df = pd.get_dummies(df, columns=["season_from_date"], drop_first=True, dtype=int)
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def prepare_seoul(path: Path) -> pd.DataFrame:
    """Load and prepare the Seoul Bike Sharing hourly dataset.

    `Functioning Day` says whether the bike system was operating. When it is
    "No", rentals are always zero, so it is kept as a feature
    (`functioning_day`). It is not the same as a working day.
    """
    df = pd.read_csv(path, encoding="unicode_escape")
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["Weekday"] = df["Date"].dt.dayofweek               # 0 = Monday
    df = df.rename(columns={
        "Date": "date",
        "Rented Bike Count": TARGET,
        "Hour": "hour",
        "Temperature(C)": "temperature",
        "Humidity(percentage)": "humidity",
        "Wind speed (m/s)": "wind_speed",
        "Visibility (10m)": "visibility",
        "Dew point temperature(C)": "dew_point",
        "Solar Radiation (MJ/m2)": "solar_radiation",
        "Rainfall(mm)": "rainfall",
        "Snowfall (cm)": "snowfall",
        "Seasons": "season",
        "Holiday": "holiday",
        "Functioning Day": "functioning_day",
    })
    df = add_date_features(df)
    # The original `Seasons` column is identical to `season_from_date`.
    df = df.drop(columns=["season"])
    df = pd.get_dummies(
        df,
        columns=["holiday", "functioning_day", "season_from_date"],
        drop_first=True,
        dtype=int,
    )
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    """Split a time-ordered dataframe: first 80% train, last 20% test."""
    X = df.drop(columns=[TARGET, "date"])
    y = df[TARGET]
    cut = int(len(df) * (1 - TEST_SHARE))
    test_start = df["date"].iloc[cut].date()
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:], test_start


def evaluate(y_true, y_pred) -> dict:
    """Return MAE, RMSE and R² rounded for reporting."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    cities = {
        "Washington DC": ("washington", prepare_washington(RAW_DIR / "hour.csv")),
        "Seoul": ("seoul", prepare_seoul(RAW_DIR / "SeoulBikeData.csv")),
    }

    rows = []
    for city, (slug, df) in cities.items():
        X_train, X_test, y_train, y_test, test_start = chronological_split(df)
        print(f"\n{city}: {len(X_train)} train rows, {len(X_test)} test rows "
              f"(test period starts {test_start})")

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            scores = evaluate(y_test, model.predict(X_test))
            rows.append({"City": city, "Model": name, **scores})
            print(f"  {name:<18} R² {scores['R2']:.4f}  "
                  f"MAE {scores['MAE']:.2f}  RMSE {scores['RMSE']:.2f}")

        rf = models["Random Forest"]
        joblib.dump(rf, MODELS_DIR / f"{slug}_rf_model.pkl", compress=3)

        importance = (
            pd.DataFrame({"Feature": X_train.columns,
                          "Importance": rf.feature_importances_})
            .sort_values("Importance", ascending=False)
        )
        importance.to_csv(REPORTS_DIR / f"{slug}_feature_importance.csv", index=False)

    pd.DataFrame(rows).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\nSaved models to models/ and results to reports/")


if __name__ == "__main__":
    main()