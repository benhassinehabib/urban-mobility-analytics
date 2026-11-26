"""
Urban Mobility Analytics — interactive Streamlit dashboard.

Loads the Random Forest models trained by src/train.py and lets users
explore model performance, feature importance and live demand predictions
for Washington DC and Seoul.

Run from the project root:

    streamlit run dashboard/app.py
"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Urban Mobility Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette (matches the project presentation)
NAVY = "#0D2230"
CARD = "#163443"
CARD_HOVER = "#1B3E50"
BORDER = "#24485A"
TEAL = "#1EBEB4"
AMBER = "#F5B852"
TEXT = "#F7FAFB"
MUTED = "#AEC3CC"
SLATE = "#5E7A88"

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, label {{
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
}}

/* Clean page frame */
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

/* Smooth entrance animation */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.hero, .kpi, .panel, .result {{ animation: fadeUp .55s ease both; }}

/* ---------- Hero header ---------- */
.hero {{
    background:
        radial-gradient(circle at 85% 20%, rgba(30,190,180,.28), transparent 45%),
        linear-gradient(135deg, #12304A 0%, {CARD} 55%, #0F2A3A 100%);
    border: 1px solid {BORDER};
    border-radius: 24px;
    padding: 34px 38px;
    margin-bottom: 26px;
    box-shadow: 0 18px 40px rgba(0,0,0,.25);
}}
.hero .eyebrow {{
    color: {TEAL}; font-size: .74rem; font-weight: 700;
    letter-spacing: .22em; text-transform: uppercase; margin-bottom: 10px;
}}
.hero h1 {{
    color: {TEXT}; font-size: 2.4rem; font-weight: 800;
    margin: 0 0 8px 0; letter-spacing: -.02em; line-height: 1.15;
}}
.hero p {{ color: {MUTED}; font-size: 1.05rem; margin: 0 0 18px 0; }}
.chip {{
    display: inline-block; padding: 6px 14px; margin: 0 8px 6px 0;
    border-radius: 999px; font-size: .8rem; font-weight: 600;
    color: {TEXT}; background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.10);
}}
.chip b {{ color: {TEAL}; }}

/* ---------- KPI cards ---------- */
.kpi {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 20px 22px;
    height: 100%;
    transition: transform .25s ease, border-color .25s ease, background .25s ease;
}}
.kpi:hover {{ transform: translateY(-3px); border-color: {TEAL}; background: {CARD_HOVER}; }}
.kpi .label {{
    color: {MUTED}; font-size: .72rem; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
}}
.kpi .value {{
    color: {TEXT}; font-size: 2.1rem; font-weight: 800;
    letter-spacing: -.02em; margin: 6px 0 2px 0;
}}
.kpi .sub {{ color: {MUTED}; font-size: .85rem; }}
.kpi .sub b {{ color: {TEAL}; font-weight: 600; }}
.kpi.accent .value {{ color: {TEAL}; }}

/* ---------- Panels and section titles ---------- */
.section-title {{
    color: {TEXT}; font-size: 1.25rem; font-weight: 700;
    margin: 30px 0 4px 0; letter-spacing: -.01em;
}}
.section-sub {{ color: {MUTED}; font-size: .92rem; margin-bottom: 14px; }}
.panel {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 18px; padding: 20px 24px;
}}
.panel h4 {{ color: {TEAL}; font-size: .78rem; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase; margin: 0 0 10px 0; }}
.panel ul {{ margin: 0; padding-left: 1.1rem; color: {TEXT}; }}
.panel li {{ margin-bottom: 8px; line-height: 1.5; }}
.panel li b {{ color: {TEAL}; }}

/* ---------- Prediction result ---------- */
.result {{
    background: linear-gradient(145deg, #13394A, {CARD});
    border: 1px solid {TEAL};
    border-radius: 22px;
    padding: 26px 28px;
    box-shadow: 0 0 0 4px rgba(30,190,180,.08), 0 16px 36px rgba(0,0,0,.25);
}}
.result .label {{ color: {MUTED}; font-size: .74rem; font-weight: 700;
    letter-spacing: .16em; text-transform: uppercase; }}
.result .big {{ color: {TEXT}; font-size: 3.6rem; font-weight: 800;
    letter-spacing: -.03em; line-height: 1.05; margin: 8px 0 2px 0;
    transition: all .3s ease; }}
.result .unit {{ color: {MUTED}; font-size: .95rem; margin-bottom: 16px; }}
.badge {{ display: inline-block; padding: 5px 12px; border-radius: 999px;
    font-size: .82rem; font-weight: 700; }}
.badge.up   {{ color: {NAVY}; background: {TEAL}; }}
.badge.down {{ color: {NAVY}; background: {AMBER}; }}
.badge.off  {{ color: {TEXT}; background: {SLATE}; }}
.result .context {{ color: {MUTED}; font-size: .88rem; margin-top: 14px; line-height: 1.5; }}

/* ---------- Tabs as a segmented control ---------- */
.stTabs [role="tablist"] {{
    gap: 6px !important;
    width: fit-content !important;
    padding: 6px !important;
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.25) !important;
}}
.stTabs [role="tab"] {{
    height: 44px !important;
    margin: 0 !important;
    padding: 0 22px !important;
    border: none !important;
    border-radius: 12px !important;
    background: transparent !important;
    color: {MUTED} !important;
    transition: background .25s ease, color .25s ease,
                box-shadow .25s ease, transform .15s ease !important;
}}
.stTabs [role="tab"] p {{
    color: inherit !important;
    font-size: .95rem !important;
    font-weight: 600 !important;
}}
.stTabs [role="tab"]:hover {{
    color: {TEXT} !important;
    background: rgba(255,255,255,.06) !important;
}}
.stTabs [role="tab"]:active {{ transform: scale(.97); }}
.stTabs [role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #2AD6C8, {TEAL}) !important;
    color: {NAVY} !important;
    box-shadow: 0 6px 18px rgba(30,190,180,.35) !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 12px; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{ background: #0A1B26; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .side-brand {{
    color: {TEXT}; font-weight: 800; font-size: 1.15rem; margin-bottom: 2px;
}}
[data-testid="stSidebar"] .side-sub {{ color: {MUTED}; font-size: .8rem; margin-bottom: 18px; }}
[data-testid="stSidebar"] .side-label {{
    color: {TEAL}; font-size: .7rem; font-weight: 700;
    letter-spacing: .16em; text-transform: uppercase; margin: 18px 0 6px 0;
}}

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div {{
    background: {CARD}; border-color: {BORDER}; border-radius: 10px;
    transition: border-color .2s ease;
}}
div[data-baseweb="select"] > div:hover {{ border-color: {TEAL}; }}
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 14px; background: {CARD}; }}
hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# PROJECT PATHS AND SETTINGS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_FILES = {
    "Washington DC": MODELS_DIR / "washington_rf_model.pkl",
    "Seoul": MODELS_DIR / "seoul_rf_model.pkl",
}

# Years covered by each training dataset. Random Forest cannot extrapolate,
# so any later year would behave exactly like the last year in the data.
DATA_YEARS = {"Washington DC": [2011, 2012], "Seoul": [2017, 2018]}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

# Readable names for model features (used in the importance chart)
FEATURE_LABELS = {
    "hour": "Hour of day",
    "temperature": "Temperature",
    "feels_like_temperature": "Feels-like temp.",
    "humidity": "Humidity",
    "wind_speed": "Wind speed",
    "weather": "Weather situation",
    "workingday": "Working day",
    "holiday": "Holiday",
    "year": "Year (index)",
    "year_from_date": "Year",
    "month": "Month",
    "month_from_date": "Month (date)",
    "weekday": "Weekday (Sun = 0)",
    "day_of_week": "Day of week",
    "Weekday": "Weekday",
    "is_weekend": "Weekend",
    "season": "Season (code)",
    "season_from_date_Spring": "Season: spring",
    "season_from_date_Summer": "Season: summer",
    "season_from_date_Winter": "Season: winter",
    "visibility": "Visibility",
    "dew_point": "Dew point",
    "solar_radiation": "Solar radiation",
    "rainfall": "Rainfall",
    "snowfall": "Snowfall",
    "holiday_No Holiday": "No holiday",
    "functioning_day_Yes": "System operating",
}


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_resource
def load_models():
    """Load both trained Random Forest models, or stop with setup help."""
    missing = [p.name for p in MODEL_FILES.values() if not p.exists()]
    if missing:
        st.error(
            f"Model files not found: {', '.join(missing)}.\n\n"
            "Train them first from the project root:\n\n"
            "`python src/train.py`"
        )
        st.stop()
    return {city: joblib.load(path) for city, path in MODEL_FILES.items()}


@st.cache_data
def load_results():
    """Load the evaluation metrics written by src/train.py."""
    return pd.read_csv(REPORTS_DIR / "model_comparison.csv")


@st.cache_data
def load_hourly_profiles():
    """Average historical rentals per hour of day, for each city.

    Used to put a prediction in context ("compared with a typical hour").
    Seoul hours when the system was not operating are excluded.
    """
    dc = pd.read_csv(RAW_DIR / "hour.csv")
    seoul = pd.read_csv(RAW_DIR / "SeoulBikeData.csv", encoding="unicode_escape")
    seoul = seoul[seoul["Functioning Day"] == "Yes"]
    return {
        "Washington DC": dc.groupby("hr")["cnt"].mean(),
        "Seoul": seoul.groupby("Hour")["Rented Bike Count"].mean(),
    }


models = load_models()
results_df = load_results()
profiles = load_hourly_profiles()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_metrics(city, model_name="Random Forest"):
    """Return R², MAE and RMSE for one city/model from the results file."""
    row = results_df[(results_df["City"] == city)
                     & (results_df["Model"] == model_name)].iloc[0]
    return {"R²": row["R2"], "MAE": row["MAE"], "RMSE": row["RMSE"]}


def get_season(month):
    """Meteorological season, matching the training pipeline."""
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    return "Winter"


def season_dummies(season):
    """One-hot season columns (Autumn is the dropped reference category)."""
    return {
        "season_from_date_Spring": int(season == "Spring"),
        "season_from_date_Summer": int(season == "Summer"),
        "season_from_date_Winter": int(season == "Winter"),
    }


def kpi(label, value, sub="", accent=False):
    """Render one KPI card as HTML."""
    cls = "kpi accent" if accent else "kpi"
    st.markdown(
        f'<div class="{cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def section(title, subtitle=""):
    """Render a section heading with an optional subtitle."""
    st.markdown(f'<div class="section-title">{title}</div>'
                f'<div class="section-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def style_chart(chart, height):
    """Apply the dashboard's dark styling to an Altair chart."""
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=BORDER,
            domainColor=BORDER, tickColor=BORDER,
            labelFont="Inter", titleFont="Inter", labelFontSize=12,
        )
        .configure_legend(
            labelColor=TEXT, titleColor=MUTED, labelFont="Inter",
            orient="top", title=None, labelFontSize=12,
        )
    )


def importance_chart(model):
    """Horizontal bar chart of the model's top 10 features."""
    df = (
        pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in model.feature_names_in_],
            "Importance": model.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .head(10)
    )
    base = alt.Chart(df).encode(
        y=alt.Y("Feature:N", sort="-x", title=None),
        x=alt.X("Importance:Q", title=None, axis=alt.Axis(format=".0%", grid=False,
                                                          labels=False, ticks=False)),
        tooltip=[alt.Tooltip("Feature:N"),
                 alt.Tooltip("Importance:Q", format=".1%")],
    )
    bars = base.mark_bar(color=TEAL, cornerRadiusEnd=6, size=22)
    labels = base.mark_text(align="left", dx=6, color=TEXT,
                            font="Inter", fontWeight=600).encode(
        text=alt.Text("Importance:Q", format=".0%"))
    return style_chart(bars + labels, 360)


def comparison_chart():
    """Grouped bar chart: Random Forest vs Linear Regression R² per city."""
    df = results_df.copy()
    base = alt.Chart(df).encode(
        x=alt.X("City:N", title=None, sort=["Washington DC", "Seoul"],
                axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        xOffset=alt.XOffset("Model:N", sort=["Random Forest", "Linear Regression"]),
        y=alt.Y("R2:Q", title="R² score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Model:N",
            sort=["Random Forest", "Linear Regression"],
            scale=alt.Scale(domain=["Random Forest", "Linear Regression"],
                            range=[TEAL, SLATE]),
        ),
        tooltip=["City:N", "Model:N", alt.Tooltip("R2:Q", format=".4f"),
                 alt.Tooltip("MAE:Q", format=".2f"),
                 alt.Tooltip("RMSE:Q", format=".2f")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=46)
    labels = base.mark_text(dy=-10, color=TEXT, font="Inter",
                            fontWeight=700).encode(text=alt.Text("R2:Q", format=".2f"))
    return style_chart(bars + labels, 340)


def profile_chart(city, selected_hour, prediction):
    """Typical demand by hour, with the current prediction highlighted."""
    profile = profiles[city]
    df = pd.DataFrame({"Hour": profile.index, "Average rentals": profile.values})
    y_max = max(profile.max(), prediction) * 1.15   # keep the scenario dot visible
    area = alt.Chart(df).mark_area(
        line={"color": TEAL, "strokeWidth": 2.5},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="rgba(30,190,180,0.35)", offset=0),
                   alt.GradientStop(color="rgba(30,190,180,0.0)", offset=1)],
            x1=1, x2=1, y1=0, y2=1,
        ),
        interpolate="monotone",
    ).encode(
        x=alt.X("Hour:Q", title="Hour of day",
                scale=alt.Scale(domain=[0, 23], nice=False), axis=alt.Axis(tickCount=12)),
        y=alt.Y("Average rentals:Q", title="Rentals per hour",
                scale=alt.Scale(domain=[0, y_max])),
        tooltip=[alt.Tooltip("Hour:Q"),
                 alt.Tooltip("Average rentals:Q", format=",.0f")],
    )
    point_df = pd.DataFrame({"Hour": [selected_hour],
                             "Average rentals": [prediction],
                             "Label": [f"Your scenario: {prediction:,.0f}"]})
    rule = alt.Chart(point_df).mark_rule(color=AMBER, strokeDash=[4, 4]).encode(
        x="Hour:Q")
    point = alt.Chart(point_df).mark_point(
        filled=True, size=180, color=AMBER, stroke=NAVY, strokeWidth=2
    ).encode(x="Hour:Q", y="Average rentals:Q",
             tooltip=[alt.Tooltip("Label:N", title="Prediction")])
    return style_chart(area + rule + point, 300)


def show_result(city, prediction, selected_hour, operating=True):
    """Hero card with the prediction and its comparison to a typical hour."""
    typical = profiles[city].loc[selected_hour]
    if not operating:
        badge = '<span class="badge off">System not operating</span>'
        context = ("The bike system is shut down, so no rentals are possible. "
                   "In the Seoul data, every non-operating hour has zero rentals.")
    else:
        change = (prediction - typical) / typical * 100 if typical else 0
        cls = "up" if change >= 0 else "down"
        arrow = "▲" if change >= 0 else "▼"
        badge = (f'<span class="badge {cls}">{arrow} {abs(change):.0f}% vs a '
                 f'typical {selected_hour:02d}:00</span>')
        context = (f"An average {selected_hour:02d}:00 in the historical data has "
                   f"<b>{typical:,.0f}</b> rentals. This is a model estimate based "
                   "on historical patterns, not a guaranteed rental count.")
    st.markdown(
        f'<div class="result"><div class="label">Estimated demand</div>'
        f'<div class="big">{prediction:,.0f}</div>'
        f'<div class="unit">bike rentals in this hour</div>'
        f'{badge}<div class="context">{context}</div></div>',
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR — SCENARIO SETTINGS
# =========================================================
with st.sidebar:
    st.markdown('<div class="side-brand">🚲 Urban Mobility</div>'
                '<div class="side-sub">Bike-sharing demand explorer</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="side-label">City</div>', unsafe_allow_html=True)
    city = st.selectbox("City", ["Washington DC", "Seoul"],
                        label_visibility="collapsed")

    st.markdown('<div class="side-label">When</div>', unsafe_allow_html=True)
    year = st.selectbox("Year", DATA_YEARS[city], index=len(DATA_YEARS[city]) - 1,
                        help="Only years covered by the training data. Random "
                             "Forest models cannot extrapolate to later years.")
    month = st.selectbox("Month", list(range(1, 13)), index=5,
                         format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
    weekday = st.selectbox("Day of week", list(range(7)),
                           format_func=lambda d: WEEKDAYS[d])
    hour = st.slider("Hour of day", 0, 23, 8, format="%d:00")

    season = get_season(month)
    st.caption(f"Season: **{season}**")

model = models[city]
rf = get_metrics(city)
lr = get_metrics(city, "Linear Regression")

# =========================================================
# HERO HEADER
# =========================================================
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Smart City · Mobility Intelligence</div>
  <h1>Urban Mobility Analytics</h1>
  <p>Predicting hourly bike-sharing demand from weather and time,
     for Washington DC and Seoul.</p>
  <span class="chip"><b>2</b> cities</span>
  <span class="chip"><b>26,139</b> hourly records</span>
  <span class="chip">Random Forest vs <b>linear baseline</b></span>
  <span class="chip">Tested on the <b>most recent 20%</b></span>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_overview, tab_compare = st.tabs([
    "🔮  Demand Prediction", "📊  Model Overview", "🏙️  City Comparison"])

# =========================================================
# PREDICTION TAB
# =========================================================
with tab_predict:
    section(f"Weather scenario · {city}",
            "Move the sliders — the prediction updates instantly.")

    if city == "Washington DC":
        c1, c2, c3 = st.columns(3)
        temp_c = c1.slider("🌡️ Temperature (°C)", -8.0, 39.0, 22.0, 0.5)
        feels_c = c2.slider("🤒 Feels-like (°C)", -16.0, 50.0, 24.0, 0.5)
        humidity_pct = c3.slider("💧 Humidity (%)", 0, 100, 55)
        c4, c5, c6 = st.columns(3)
        wind_kmh = c4.slider("🌬️ Wind speed (km/h)", 0.0, 57.0, 13.0, 0.5)
        weather = c5.selectbox(
            "☁️ Weather", [1, 2, 3, 4],
            format_func=lambda x: {1: "Clear / few clouds", 2: "Mist / cloudy",
                                   3: "Light rain / snow", 4: "Heavy rain / snow"}[x])
        holiday = c6.selectbox("🎉 Public holiday", ["No", "Yes"])

        is_weekend = int(weekday >= 5)
        is_holiday = int(holiday == "Yes")
        workingday = int(not is_weekend and not is_holiday)

        # Encode inputs exactly as in the original dataset:
        # season 1 = winter … 4 = fall; year 0 = 2011, 1 = 2012;
        # weekday 0 = Sunday; weather values normalised by the dataset authors.
        features = {
            "season": {"Winter": 1, "Spring": 2, "Summer": 3, "Autumn": 4}[season],
            "year": year - 2011,
            "month": month,
            "hour": hour,
            "holiday": is_holiday,
            "weekday": (weekday + 1) % 7,
            "workingday": workingday,
            "weather": weather,
            "temperature": (temp_c + 8) / 47,
            "feels_like_temperature": (feels_c + 16) / 66,
            "humidity": humidity_pct / 100,
            "wind_speed": wind_kmh / 67,
            "day_of_week": weekday,
            "is_weekend": is_weekend,
            "month_from_date": month,
            "year_from_date": year,
            **season_dummies(season),
        }
        operating = True
        st.caption(f"Working day: **{'Yes' if workingday else 'No'}** "
                   "(derived from weekday and holiday, as in the dataset)")

    else:
        c1, c2, c3 = st.columns(3)
        temperature = c1.slider("🌡️ Temperature (°C)", -18.0, 40.0, 20.0, 0.5)
        humidity = c2.slider("💧 Humidity (%)", 0, 100, 55)
        wind_speed = c3.slider("🌬️ Wind speed (m/s)", 0.0, 7.5, 1.5, 0.1)
        c4, c5, c6 = st.columns(3)
        rainfall = c4.slider("🌧️ Rainfall (mm)", 0.0, 35.0, 0.0, 0.5)
        holiday = c5.selectbox("🎉 Public holiday", ["No Holiday", "Holiday"])
        functioning = c6.selectbox(
            "🚦 Bike system operating?", ["Yes", "No"],
            help="'Functioning Day' in the Seoul dataset. When the system was "
                 "not operating, rentals were always zero.")

        with st.expander("More weather conditions"):
            e1, e2, e3, e4 = st.columns(4)
            visibility_m = e1.slider("Visibility (m)", 0, 20000, 15000, 100)
            dew_point = e2.slider("Dew point (°C)", -31.0, 28.0, 10.0, 0.5)
            solar_radiation = e3.slider("Solar radiation (MJ/m²)", 0.0, 3.6, 1.0, 0.05)
            snowfall = e4.slider("Snowfall (cm)", 0.0, 9.0, 0.0, 0.1)

        features = {
            "hour": hour,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "visibility": visibility_m / 10,       # dataset unit is 10 m
            "dew_point": dew_point,
            "solar_radiation": solar_radiation,
            "rainfall": rainfall,
            "snowfall": snowfall,
            "Weekday": weekday,
            "day_of_week": weekday,
            "is_weekend": int(weekday >= 5),
            "month_from_date": month,
            "year_from_date": year,
            "holiday_No Holiday": int(holiday == "No Holiday"),
            "functioning_day_Yes": int(functioning == "Yes"),
            **season_dummies(season),
        }
        operating = functioning == "Yes"

    # Live prediction. When Seoul's system is shut down, rentals are always 0
    # in the data; only 72 such hours are in the training period, so the
    # model has not fully learned this rule and it is applied directly.
    X = pd.DataFrame([features]).reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = max(0.0, float(model.predict(X)[0])) if operating else 0.0

    st.write("")
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        show_result(city, prediction, hour, operating)
    with right:
        st.markdown(f'<div class="section-sub" style="margin:0 0 4px 0">'
                    f'Your scenario vs the average day in {city}</div>',
                    unsafe_allow_html=True)
        st.altair_chart(profile_chart(city, hour, prediction),
                        use_container_width=True, theme=None)

# =========================================================
# MODEL OVERVIEW TAB
# =========================================================
with tab_overview:
    section(f"Model performance · {city}",
            "Random Forest trained on the first 80% of the timeline and "
            "tested on the most recent 20%.")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("R² score", f"{rf['R²']:.3f}",
            f"Linear baseline <b>{lr['R²']:.3f}</b>", accent=True)
    with k2:
        kpi("MAE", f"{rf['MAE']:,.0f}", "Average error, bikes per hour")
    with k3:
        kpi("RMSE", f"{rf['RMSE']:,.0f}", "Penalises large errors more")
    with k4:
        improvement = (rf["R²"] - lr["R²"]) / lr["R²"] * 100
        kpi("Gain vs baseline", f"+{improvement:.0f}%", "Improvement in R²")

    st.write("")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        section("What drives demand?",
                "Top 10 features by Random Forest importance.")
        st.altair_chart(importance_chart(model), use_container_width=True, theme=None)
    with right:
        section("How to read this", "")
        driver = ("the <b>hour of day</b> — demand follows commuting peaks"
                  if city == "Washington DC" else
                  "<b>temperature</b>, closely followed by the hour of day")
        st.markdown(f"""
        <div class="panel">
          <h4>Key insight</h4>
          <ul>
            <li>In {city}, the strongest driver is {driver}.</li>
            <li>Importance shows <b>predictive contribution</b> within this
                model. It does not prove causation.</li>
            <li>Some features overlap (for example two year columns), so their
                importance is shared between them.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# CITY COMPARISON TAB
# =========================================================
with tab_compare:
    section("Random Forest vs linear baseline",
            "R² on each city's test period. Higher is better.")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.altair_chart(comparison_chart(), use_container_width=True, theme=None)
    with right:
        st.markdown("""
        <div class="panel">
          <h4>What the comparison tells us</h4>
          <ul>
            <li><b>Random Forest wins in both cities</b> — demand has
                non-linear patterns such as morning and evening peaks.</li>
            <li><b>Washington DC scores higher.</b> Its demand is dominated by
                regular commuting hours, which are easy to learn.</li>
            <li><b>Seoul is harder.</b> Its test period (19 Sep – 30 Nov 2018)
                includes October and November, months never seen
                in training.</li>
            <li><b>Don't compare MAE or RMSE across cities</b> — the two systems
                have very different demand scales.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    section("Full results", "Test-set metrics for every city and model.")
    table = results_df.rename(columns={"R2": "R²"})
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "R²": st.column_config.ProgressColumn(
                "R²", format="%.4f", min_value=0.0, max_value=1.0),
            "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
        },
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f'<div style="text-align:center;color:{SLATE};font-size:.8rem;margin-top:40px">'
    "Urban Mobility Analytics · Habib Iben Hassine · MSc Computer Science "
    "Engineering, Széchenyi István University</div>",
    unsafe_allow_html=True,
)