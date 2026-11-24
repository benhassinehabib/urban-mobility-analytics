import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Urban Mobility Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)
# =========================================================
# CUSTOM DASHBOARD STYLING
# =========================================================
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F7FA;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main title */
h1 {
    color: #16324F;
    font-weight: 800;
}

/* Section headings */
h2, h3 {
    color: #1F4E79;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E3E8EF;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #EAF0F6;
}

/* Buttons */
.stButton > button {
    background-color: #1F6E8C;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #164E63;
    color: white;
    border: none;
}

/* Tabs Navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: #E8EEF5;
    padding: 8px;
    border-radius: 12px;
}

/* Individual tabs */
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    color: #456179;
    transition: all 0.2s ease;
}

/* Hover effect */
.stTabs [data-baseweb="tab"]:hover {
    background-color: #D6E3EE;
    color: #16324F;
}

/* Selected tab */
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF;
    color: #1F6E8C !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
}

/* Dividers */
hr {
    border-color: #D8E1EA;
}
/* Dashboard header */
.dashboard-header {
    background: linear-gradient(135deg, #16324F, #1F6E8C);
    padding: 30px 35px;
    border-radius: 18px;
    margin-bottom: 15px;
}

.dashboard-header h1 {
    color: #FFFFFF !important;
    font-size: 2.5rem;
    margin-bottom: 10px;
}

.dashboard-header p {
    color: #DCEAF2 !important;
    font-size: 1.1rem;
    margin-bottom: 0;
}

.header-label {
    color: #A8DADC;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PROJECT PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    dc_model = joblib.load(MODELS_DIR / "washington_rf_model.pkl")
    seoul_model = joblib.load(MODELS_DIR / "seoul_rf_model.pkl")
    return dc_model, seoul_model

dc_model, seoul_model = load_models()

# =========================================================
# MODEL PERFORMANCE
# =========================================================
MODEL_RESULTS = {
    "Washington DC": {
        "R²": 0.8993,
        "MAE": 45.47,
        "RMSE": 69.95
    },
    "Seoul": {
        "R²": 0.7383,
        "MAE": 216.91,
        "RMSE": 313.77
    }
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"


def show_performance(city):
    results = MODEL_RESULTS[city]

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", f"{results['R²']:.4f}")
    col2.metric("MAE", f"{results['MAE']:,.2f}")
    col3.metric("RMSE", f"{results['RMSE']:,.2f}")


def show_feature_importance(model, city):
    st.subheader("Feature Importance")
    st.caption(
        "Shows which features contributed most to the Random Forest model."
    )

    importance_df = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df.sort_values("Importance", ascending=False)
        .head(10)
        .set_index("Feature")
    )

    st.bar_chart(
        importance_df,
        x_label="Feature",
        y_label="Importance",
        horizontal=True
    )


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.markdown("""
<div class="dashboard-header">
<div class="header-label">SMART CITY • MOBILITY INTELLIGENCE</div>
<h1>Urban Mobility Analytics</h1>
<p>Bike-sharing demand prediction using machine learning,
weather conditions, and temporal data.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Dashboard Settings")

city = st.sidebar.selectbox(
    "Select a city",
    ["Washington DC", "Seoul"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📅 Time Conditions")

year = st.sidebar.selectbox(
    "Year",
    list(range(2011, 2036)),
    index=15  # 2026
)

month = st.sidebar.selectbox(
    "Month",
    list(range(1, 13)),
    index=10,
    format_func=lambda x: pd.Timestamp(2026, x, 1).strftime("%B")
)

hour = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=12
)

season = get_season(month)

# =========================================================
# DASHBOARD TABS
# =========================================================
tab_overview, tab_prediction, tab_comparison = st.tabs([
    "📊 Overview",
    "🔮 Demand Prediction",
    "🏙️ City Comparison"
])

# =========================================================
# OVERVIEW TAB
# =========================================================
with tab_overview:

    st.header(f"📊 {city} Dashboard")

    st.markdown(
        "Overview of the machine learning model and its performance."
    )

    # Model overview
    col1, col2, col3 = st.columns(3)

    col1.metric("Selected City", city)
    col2.metric("Machine Learning Model", "Random Forest 🌲")
    col3.metric("Model Status", "🟢 Operational")

    st.divider()

    # Model performance
    st.subheader("📈 Model Performance")

    metrics = MODEL_RESULTS[city]

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", f"{metrics['R²']:.4f}")
    col2.metric("MAE", f"{metrics['MAE']:.2f}")
    col3.metric("RMSE", f"{metrics['RMSE']:.2f}")

    st.divider()

    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================
    st.subheader("🌲 What Influences Bike Demand?")

    if city == "Washington DC":
        model = dc_model
    else:
        model = seoul_model

    show_feature_importance(model, city)

    st.caption(
        "Feature importance shows how strongly each variable contributes "
        "to the Random Forest model's predictions. Higher importance "
        "indicates greater influence on the model."
    )

    st.divider()

    st.subheader("ℹ️ About the Project")

    st.markdown("""
    This project explores how **weather conditions and temporal factors**
    influence bike-sharing demand in two major cities: **Washington DC and Seoul**.

    The trained **Random Forest regression models** use environmental and
    calendar-related variables to predict bike rental demand.

    #### Data Factors
    - 🌡️ Temperature and humidity
    - 🌬️ Wind speed and weather conditions
    - 🕒 Hour of the day
    - 📅 Weekday, month, and season
    - 🏢 Working days and holidays
    """)

    st.info(
        "The goal is to understand urban mobility patterns and demonstrate "
        "how machine learning can support data-driven transportation planning."
    )

# =========================================================
# PREDICTION TAB
# =========================================================
with tab_prediction:

    st.header(f"🔮 Bike Demand Prediction — {city}")

    st.write(
        "Adjust the input conditions below and click the prediction button."
    )

    # =====================================================
    # WASHINGTON DC
    # =====================================================
    if city == "Washington DC":

        st.subheader("🌤️ Weather Conditions")

        col1, col2, col3 = st.columns(3)

        with col1:
            temperature = st.slider(
                "Temperature (normalized)",
                0.0, 1.0, 0.5
            )

        with col2:
            feels_like_temperature = st.slider(
                "Feels Like Temperature (normalized)",
                0.0, 1.0, 0.5
            )

        with col3:
            humidity = st.slider(
                "Humidity (normalized)",
                0.0, 1.0, 0.5
            )

        col4, col5 = st.columns(2)

        with col4:
            wind_speed = st.slider(
                "Wind Speed (normalized)",
                0.0, 1.0, 0.3
            )

        with col5:
            weather = st.selectbox(
                "Weather Situation",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Clear / Few Clouds",
                    2: "Mist / Cloudy",
                    3: "Light Rain / Snow",
                    4: "Heavy Rain / Snow"
                }[x]
            )

        st.subheader("📅 Calendar Conditions")

        col1, col2, col3 = st.columns(3)

        with col1:
            workingday = st.selectbox(
                "Working Day",
                ["Yes", "No"]
            )

        with col2:
            weekday = st.selectbox(
                "Day of Week",
                list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ][x]
            )

        with col3:
            holiday = st.selectbox(
                "Holiday",
                ["No", "Yes"]
            )

        season_code = {
            "Spring": 1,
            "Summer": 2,
            "Fall": 3,
            "Winter": 4
        }[season]

        dc_input = pd.DataFrame([{
            "season": season_code,
            "year": year,
            "month": month,
            "hour": hour,
            "holiday": 1 if holiday == "Yes" else 0,
            "weekday": weekday,
            "workingday": 1 if workingday == "Yes" else 0,
            "weather": weather,
            "temperature": temperature,
            "feels_like_temperature": feels_like_temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "day_of_week": weekday,
            "is_weekend": 1 if weekday >= 5 else 0,
            "month_from_date": month,
            "year_from_date": year,
            "season_from_date_Spring": 1 if season == "Spring" else 0,
            "season_from_date_Summer": 1 if season == "Summer" else 0,
            "season_from_date_Winter": 1 if season == "Winter" else 0
        }])

        dc_input = dc_input.reindex(
            columns=dc_model.feature_names_in_,
            fill_value=0
        )

        if st.button("🔮 Predict Bike Demand", use_container_width=True):

            prediction = dc_model.predict(dc_input)[0]
            prediction = max(0, prediction)

            st.success("Prediction completed successfully!")

            st.markdown("### 🚲 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated Bike Rentals",
                    f"{prediction:,.0f}"
                )

            with col2:
                st.metric(
                    "Selected Hour",
                    f"{hour:02d}:00"
                )

            st.info(
                f"Estimated bike-sharing demand in Washington DC "
                f"for {month}/{year} at {hour:02d}:00."
            )

    # =====================================================
    # SEOUL
    # =====================================================
    else:

        st.subheader("🌤️ Weather Conditions")

        col1, col2, col3 = st.columns(3)

        with col1:
            temperature = st.slider(
                "Temperature (°C)",
                -20.0, 40.0, 20.0
            )

        with col2:
            humidity = st.slider(
                "Humidity (%)",
                0, 100, 50
            )

        with col3:
            wind_speed = st.slider(
                "Wind Speed (m/s)",
                0.0, 50.0, 2.0
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            visibility = st.slider(
                "Visibility",
                0, 2000, 1000
            )

        with col5:
            dew_point = st.slider(
                "Dew Point (°C)",
                -30.0, 40.0, 10.0
            )

        with col6:
            solar_radiation = st.slider(
                "Solar Radiation",
                0.0, 5.0, 1.0
            )

        col7, col8 = st.columns(2)

        with col7:
            rainfall = st.slider(
                "Rainfall (mm)",
                0.0, 100.0, 0.0
            )

        with col8:
            snowfall = st.slider(
                "Snowfall (cm)",
                0.0, 100.0, 0.0
            )

        st.subheader("📅 Calendar Conditions")

        col1, col2, col3 = st.columns(3)

        with col1:
            weekday = st.selectbox(
                "Day of Week",
                list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ][x]
            )

        with col2:
            holiday = st.selectbox(
                "Holiday",
                ["No Holiday", "Holiday"]
            )

        with col3:
            working_day = st.selectbox(
                "Working Day",
                ["Yes", "No"]
            )

        seoul_input = pd.DataFrame([{
            "hour": hour,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "visibility": visibility,
            "dew_point": dew_point,
            "solar_radiation": solar_radiation,
            "rainfall": rainfall,
            "snowfall": snowfall,
            "Weekday": weekday,
            "day_of_week": weekday,
            "is_weekend": 1 if weekday >= 5 else 0,
            "month_from_date": month,
            "year_from_date": year,
            "holiday_No Holiday": 1 if holiday == "No Holiday" else 0,
            "working_day_Yes": 1 if working_day == "Yes" else 0,
            "season_from_date_Spring": 1 if season == "Spring" else 0,
            "season_from_date_Summer": 1 if season == "Summer" else 0,
            "season_from_date_Winter": 1 if season == "Winter" else 0
        }])

        seoul_input = seoul_input.reindex(
            columns=seoul_model.feature_names_in_,
            fill_value=0
        )

        if st.button("🔮 Predict Bike Demand", use_container_width=True):

            prediction = seoul_model.predict(seoul_input)[0]
            prediction = max(0, prediction)

            st.success("Prediction completed successfully!")

            st.markdown("### 🚲 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated Bike Rentals",
                    f"{prediction:,.0f}"
                )

            with col2:
                st.metric(
                    "Selected Hour",
                    f"{hour:02d}:00"
                )

            st.info(
                f"Estimated bike-sharing demand in Seoul "
                f"for {month}/{year} at {hour:02d}:00."
            )

# =========================================================
# CITY COMPARISON TAB
# =========================================================
with tab_comparison:

    st.header("🏙️ City Model Comparison")

    st.markdown("""
    Compare the predictive performance of the **Random Forest models**
    trained using bike-sharing data from **Washington DC and Seoul**.
    """)

    dc_results = MODEL_RESULTS["Washington DC"]
    seoul_results = MODEL_RESULTS["Seoul"]

    st.subheader("📊 Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🇺🇸 Washington DC")
        st.metric("R² Score", f"{dc_results['R²']:.4f}")
        st.metric("MAE", f"{dc_results['MAE']:.2f}")
        st.metric("RMSE", f"{dc_results['RMSE']:.2f}")

    with col2:
        st.markdown("### 🇰🇷 Seoul")
        st.metric("R² Score", f"{seoul_results['R²']:.4f}")
        st.metric("MAE", f"{seoul_results['MAE']:.2f}")
        st.metric("RMSE", f"{seoul_results['RMSE']:.2f}")

    st.divider()

    st.subheader("R² Score Comparison")

    r2_df = pd.DataFrame({
        "City": ["Washington DC", "Seoul"],
        "R² Score": [
            dc_results["R²"],
            seoul_results["R²"]
        ]
    }).set_index("City")

    st.bar_chart(r2_df)
    st.caption(
        "Washington DC achieves a higher R² score, indicating that "
        "its Random Forest model explains more variation in bike-sharing "
        "demand than the Seoul model on its evaluation data."
    )

    st.divider()

    st.subheader("Model Interpretation")

    st.markdown(
        f"""
        - **Washington DC R²:** {dc_results['R²']:.4f}
        - **Seoul R²:** {seoul_results['R²']:.4f}

        The R² score measures how much variation in bike rental demand
        is explained by the model. A higher R² generally indicates
        better model performance on the evaluated data.
        """
    )

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "Urban Mobility Analytics | Bike-Sharing Demand Prediction | "
    "Machine Learning Project"
)