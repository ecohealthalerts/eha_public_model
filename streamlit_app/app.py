"""
EHA Disease Risk Predictor — Streamlit App
"""
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import json
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict
import math, os
# Climate data fetcher
from climate_fetcher import fetch_and_compute_features, get_county_coords, COUNTY_COORDS
# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
CONFIG_DIR = ROOT / "serving" / "config"
DATA_DIR   = ROOT / "data" / "processed"

# ── Constants ────────────────────────────────────────────────────────
FEATURES_A = [
    "total_rainfall_mm", "mean_rainfall_mm", "max_rainfall_mm",
    "rainfall_variability", "mean_temperature_c", "max_temperature_c",
    "min_temperature_c", "temp_range_c", "peak_rainfall_month",
]
FEATURES_B = [
    "avg_rainfall_mm", "avg_rainfall_mm_lag1", "avg_rainfall_mm_lag2",
    "avg_rainfall_mm_roll3", "mean_temperature_celcius",
    "mean_temperature_celcius_lag1", "mean_temperature_celcius_lag2",
    "mean_temperature_celcius_roll3", "month_sin", "month_cos",
]

SLIDER_CFG_A = {
    "total_rainfall_mm":    ("Total Rainfall (mm/yr)", 100.0, 2600.0, 1387.0, 10.0),
    "mean_rainfall_mm":     ("Mean Monthly Rainfall (mm)", 10.0, 250.0, 115.0, 1.0),
    "max_rainfall_mm":      ("Max Monthly Rainfall (mm)", 40.0, 600.0, 254.0, 5.0),
    "rainfall_variability": ("Rainfall Variability (std)", 10.0, 160.0, 70.0, 1.0),
    "mean_temperature_c":   ("Mean Temperature (°C)", 18.0, 35.0, 25.2, 0.1),
    "max_temperature_c":    ("Max Temperature (°C)", 20.0, 42.0, 29.0, 0.1),
    "min_temperature_c":    ("Min Temperature (°C)", 12.0, 28.0, 21.3, 0.1),
    "temp_range_c":         ("Temperature Range (°C)", 2.0, 16.0, 7.8, 0.1),
    "peak_rainfall_month":  ("Peak Rainfall Month", 1, 12, 4, 1),
}

SLIDER_CFG_B = {
    "avg_rainfall_mm":                ("Avg Rainfall (mm)", 0.0, 400.0, 80.0, 1.0),
    "avg_rainfall_mm_lag1":           ("Rainfall Lag‑1 (mm)", 0.0, 400.0, 75.0, 1.0),
    "avg_rainfall_mm_lag2":           ("Rainfall Lag‑2 (mm)", 0.0, 400.0, 70.0, 1.0),
    "avg_rainfall_mm_roll3":          ("Rainfall 3‑mo Avg (mm)", 0.0, 400.0, 75.0, 1.0),
    "mean_temperature_celcius":       ("Avg Temp (°C)", 15.0, 38.0, 25.0, 0.1),
    "mean_temperature_celcius_lag1":  ("Temp Lag‑1 (°C)", 15.0, 38.0, 25.0, 0.1),
    "mean_temperature_celcius_lag2":  ("Temp Lag‑2 (°C)", 15.0, 38.0, 25.0, 0.1),
    "mean_temperature_celcius_roll3": ("Temp 3‑mo Avg (°C)", 15.0, 38.0, 25.0, 0.1),
}

COUNTIES = [
    "Baringo","Bomet","Bungoma","Busia","Elgeyo-Marakwet","Embu","Garissa",
    "Homa Bay","Isiolo","Kajiado","Kakamega","Kericho","Kiambu","Kilifi",
    "Kirinyaga","Kisii","Kisumu","Kitui","Kwale","Laikipia","Lamu",
    "Machakos","Makueni","Mandera","Marsabit","Meru","Migori","Mombasa",
    "Murang'a","Nairobi City","Nakuru","Nandi","Narok","Nyamira",
    "Nyandarua","Nyeri","Samburu","Siaya","Taita Taveta","Tana River",
    "Tharaka-Nithi","Trans Nzoia","Turkana","Uasin Gishu","Vihiga","Wajir",
    "West Pokot",
]

DISCLAIMER = (
    "Model-generated risk signal for early-warning purposes only. "
    "Always validate with field surveillance data before taking action."
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EHA Disease Risk Predictor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, .stApp, .stApp * { font-family: 'Inter', sans-serif !important; }

/* header bar */
.hero-bar {
    background: linear-gradient(135deg, #0d253f 0%, #1a3a5c 40%, #2E86AB 100%);
    border-radius: 14px; padding: 28px 32px; margin-bottom: 24px;
    border: 1px solid rgba(46,134,171,0.25);
}
.hero-bar h1 { color: #fff; margin: 0 0 4px 0; font-size: 1.6rem; font-weight: 700; }
.hero-bar p  { color: #94b8d4; margin: 0; font-size: 0.95rem; }

/* result cards */
.metric-card {
    background: rgba(255,255,255,0.04); backdrop-filter: blur(12px);
    border-radius: 14px; padding: 22px 26px;
    border: 1px solid rgba(255,255,255,0.08); text-align: center;
}
.risk-high { border-left: 4px solid #E84855; }
.risk-low  { border-left: 4px solid #3BB273; }
.badge-high {
    display: inline-block; padding: 6px 20px; border-radius: 8px;
    background: rgba(232,72,85,0.15); color: #E84855;
    font-weight: 700; font-size: 1.3rem; letter-spacing: 1px;
}
.badge-low {
    display: inline-block; padding: 6px 20px; border-radius: 8px;
    background: rgba(59,178,115,0.15); color: #3BB273;
    font-weight: 700; font-size: 1.3rem; letter-spacing: 1px;
}
.stat-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase;
              letter-spacing: 1px; margin-bottom: 4px; }
.stat-value { color: #e6edf3; font-size: 1.5rem; font-weight: 600; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 10px 20px;
    background: rgba(255,255,255,0.03);
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0E1117 100%);
}

/* chat */
.chat-user { background: rgba(46,134,171,0.1); border-radius: 12px;
             padding: 14px 18px; margin: 8px 0; border-left: 3px solid #2E86AB; }
.chat-ai   { background: rgba(255,255,255,0.04); border-radius: 12px;
             padding: 14px 18px; margin: 8px 0; border-left: 3px solid #3BB273; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models, explainers, thresholds, features = {}, {}, {}, {}
    for disease in ("cholera", "malaria"):
        for option in ("a", "b"):
            key = f"{disease}_{option}"
            mp = MODELS_DIR / f"xgb_{key}.json"
            tp = CONFIG_DIR / f"threshold_{key}.json"
            if not mp.exists():
                continue
            m = xgb.XGBClassifier()
            m.load_model(str(mp))
            models[key] = m
            explainers[key] = shap.TreeExplainer(m)
            features[key] = FEATURES_A if option == "a" else FEATURES_B
            t = 0.5
            if tp.exists():
                with open(tp) as f:
                    t = json.load(f)["threshold"]
            thresholds[key] = t
    return models, explainers, thresholds, features

@st.cache_data
def load_historical():
    path = DATA_DIR / "option_a_features.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

MODELS, EXPLAINERS, THRESHOLDS, FEAT_MAPS = load_models()


# ── Prediction engine ────────────────────────────────────────────────
def predict(feat_dict: Dict[str, float], disease: str, option: str):
    key = f"{disease}_{option}"
    if key not in MODELS:
        return None
    model = MODELS[key]
    explainer = EXPLAINERS[key]
    threshold = THRESHOLDS[key]
    feat_cols = FEAT_MAPS[key]
    X = np.array([[feat_dict[f] for f in feat_cols]])
    prob = float(model.predict_proba(X)[0][1])
    pred = int(prob >= threshold)
    sv = explainer.shap_values(X)[0]
    top_idx = np.argsort(np.abs(sv))[::-1][:5]
    drivers = [
        {"feature": feat_cols[i],
         "impact": "increases risk" if sv[i] > 0 else "decreases risk",
         "shap": round(float(sv[i]), 4)}
        for i in top_idx
    ]
    dist = abs(prob - threshold)
    conf = "High" if dist >= 0.30 else ("Medium" if dist >= 0.15 else "Low")
    return {
        "risk_level": "HIGH" if pred else "LOW",
        "probability": round(prob, 4),
        "confidence": conf,
        "threshold": round(threshold, 4),
        "drivers": drivers,
        "key": key,
    }


# ── LLM helpers ──────────────────────────────────────────────────────
def build_extraction_prompt(user_text, option):
    feats = FEATURES_A if option == "a" else FEATURES_B
    return f"""You are a climate-data extraction assistant for a Kenya disease risk model.

Given the user's description, extract values for these features: {feats}

Also extract: county (one of the 47 Kenyan counties), disease ("cholera" or "malaria"), year (integer).

Rules:
- Extract any values explicitly stated.
- For missing values, infer reasonable estimates for Kenya based on context (county, season, description). Use typical Kenya climate data.
- Return ONLY valid JSON, nothing else.

Output format:
{{"county":"...","year":2025,"disease":"cholera","features":{{"feature_name":value,...}},"reasoning":"brief note on inferred values"}}

User input: {user_text}"""


def build_explanation_prompt(result, county, year, disease, language):
    """Build a concise outbreak-focused prompt in English or Kiswahili."""
    drivers_text = ""
    for d in result["drivers"]:
        feat_nice = d["feature"].replace("_", " ")
        drivers_text += f"  - {feat_nice}: SHAP value = {d['shap']:.4f} ({d['impact']})\n"

    lang_instruction = (
        "Write your entire response in Kiswahili. Use clear, professional Kiswahili that a county health officer in Kenya would understand."
        if language == "Kiswahili"
        else "Write your entire response in clear, professional English."
    )

    return f"""You are a public health outbreak analyst for Kenya.

Only answer outbreak-related questions. Be concise and focus on the most important points.

Prediction summary for {disease} risk in {county} county for {year}:
- Risk Level: {result['risk_level']}
- Risk Probability: {result['probability']:.1%}
- Threshold: {result['threshold']:.3f} ({result['threshold']:.1%})
- Confidence: {result['confidence']}
- Model key: {result['key']}

Top drivers:
{drivers_text}

Explain in a short, practical way:
1. What the risk means for outbreak response in {county}.
2. Why the threshold matters and how it changes the HIGH/LOW decision.
3. How accurate the model is expected to be, based on threshold and confidence.
4. Which top climate drivers are pushing risk up or down.

Keep the response brief and useful. Do not add unrelated details.

{lang_instruction}"""


AI_PROVIDERS = [
    "Gemini", "Claude", "Anthropic", "OpenRouter", "DeepSeek", "Kimi"
]

PROVIDER_HELP = {
    "Gemini": "Get one free key at aistudio.google.com",
    "Claude": "Use your Anthropic Claude API key",
    "Anthropic": "Use your Anthropic API key",
    "OpenRouter": "Use your OpenRouter API key and specify any model name below",
    "DeepSeek": "Use your DeepSeek API key",
    "Kimi": "Use your Kimi API key",
}


def call_gemini(prompt, api_key):
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "Gemini support requires the google-generativeai package. "
            "Install it with `pip install google-generativeai`.") from e

    genai.configure(api_key=api_key)
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "not found" in str(e).lower():
                continue
            raise
    raise RuntimeError(
        "All Gemini models exhausted quota. Please try again later or check your API key at aistudio.google.com.")


def call_openai(prompt, api_key):
    import requests
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 650,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_anthropic(prompt, api_key):
    import requests
    url = "https://api.anthropic.com/v1/chat/completions"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": "claude-3.5",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 650,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt, api_key, model="gpt-4o-mini"):
    import requests
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://eha-predictor.streamlit.app",  # Required by OpenRouter
        "X-Title": "EHA Disease Risk Predictor",  # Required by OpenRouter
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 650,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_deepseek(prompt, api_key):
    import requests
    url = "https://api.deepseek.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 650,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_kimi(prompt, api_key):
    import requests
    url = "https://api.kimi.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 650,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_llm(prompt, provider, api_key, model=None):
    provider = provider.lower()
    if provider == "gemini":
        return call_gemini(prompt, api_key)
    if provider == "openai":
        return call_openai(prompt, api_key)
    if provider == "anthropic" or provider == "claude":
        return call_anthropic(prompt, api_key)
    if provider == "openrouter":
        return call_openrouter(prompt, api_key, model or "gpt-4o-mini")
    if provider == "deepseek":
        return call_deepseek(prompt, api_key)
    if provider == "kimi":
        return call_kimi(prompt, api_key)
    raise RuntimeError(f"Unsupported provider: {provider}")


def generate_explanation(result, county, year, disease, language, provider, api_key, model=None):
    """Generate a bilingual natural language explanation of the prediction."""
    if not api_key:
        return None
    try:
        prompt = build_explanation_prompt(result, county, year, disease, language)
        return call_llm(prompt, provider, api_key, model)
    except Exception as e:
        return f"⚠️ Could not generate explanation: {e}"


# ── Display helpers ──────────────────────────────────────────────────
def render_results(result, county, year, disease):
    is_high = result["risk_level"] == "HIGH"
    badge = "badge-high" if is_high else "badge-low"
    card = "risk-high" if is_high else "risk-low"

    st.markdown("---")
    st.markdown(f"### Prediction Results — {county.title()}, {year}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card {card}">
            <div class="stat-label">Risk Level</div>
            <div class="{badge}">{result['risk_level']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="stat-label">Probability</div>
            <div class="stat-value">{result['probability']:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="stat-label">Confidence</div>
            <div class="stat-value">{result['confidence']}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="stat-label">Threshold</div>
            <div class="stat-value">{result['threshold']:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge chart
    gc1, gc2 = st.columns([1, 1])
    with gc1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability"] * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#e6edf3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": "#E84855" if is_high else "#3BB273"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "threshold": {
                    "line": {"color": "#F6AE2D", "width": 3},
                    "value": result["threshold"] * 100,
                },
                "steps": [
                    {"range": [0, result["threshold"] * 100], "color": "rgba(59,178,115,0.1)"},
                    {"range": [result["threshold"] * 100, 100], "color": "rgba(232,72,85,0.1)"},
                ],
            },
            title={"text": f"{disease.title()} Risk Probability", "font": {"size": 16, "color": "#8b949e"}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=60, b=20, l=30, r=30),
            font={"color": "#e6edf3"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP drivers
    with gc2:
        drivers = result["drivers"]
        colors = ["#E84855" if d["shap"] > 0 else "#3BB273" for d in drivers]
        fig2 = go.Figure(go.Bar(
            x=[d["shap"] for d in drivers],
            y=[d["feature"].replace("_", " ").title() for d in drivers],
            orientation="h",
            marker_color=colors,
            text=[d["impact"] for d in drivers],
            textposition="auto",
            textfont={"size": 11, "color": "#e6edf3"},
        ))
        fig2.update_layout(
            title={"text": "Top Risk Drivers (SHAP)", "font": {"size": 16, "color": "#8b949e"}},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=60, b=20, l=10, r=10),
            xaxis={"title": "SHAP Value", "color": "#8b949e", "gridcolor": "rgba(255,255,255,0.05)"},
            yaxis={"color": "#e6edf3"},
            font={"color": "#e6edf3"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"⚠️ {DISCLAIMER}")
    return result  # return so callers can generate explanation


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## EHA Predictor")
    st.caption("Disease Risk Early-Warning System for Kenya")
    st.markdown("---")

    disease = st.radio("Disease", ["cholera", "malaria"], format_func=str.title)
    option = st.radio("Model Option", ["a", "b"],
                       format_func=lambda x: "A — Annual (Primary)" if x == "a" else "B — Monthly (Research)")

    st.markdown("---")
    language = st.radio("Explanation Language", ["English", "Kiswahili"],
                         help="The AI will explain prediction results in this language")

    st.markdown("---")
    with st.expander("AI Settings"):
        provider = st.selectbox("AI Provider", AI_PROVIDERS, index=0)
        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            help=PROVIDER_HELP.get(provider, "Enter your provider API key."),
        )
        if provider == "OpenRouter":
            openrouter_model = st.text_input(
                "OpenRouter Model",
                value="gpt-4o-mini",
                help="Enter any OpenRouter model name (e.g., gpt-4o-mini, anthropic/claude-3.5-sonnet, etc.)",
            )
        st.caption("Required for AI explanations & AI Assistant tab")

    st.markdown("---")
    st.markdown(f"**Models loaded:** {', '.join(MODELS.keys()) if MODELS else 'None'}")
    st.caption("v2.0 · EHA24-7-MVP")


# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-bar">
    <h1>🦠 EHA Disease Risk Predictor</h1>
    <p>AI-powered early-warning system for cholera and malaria risk in Kenya's 47 counties</p>
</div>
""", unsafe_allow_html=True)

tab_real, tab_manual, tab_ai, tab_history = st.tabs(["🌍 Actual Climate Data", "🎛️ Custom Scenario", "🤖 AI Assistant", "📊 Historical Data"])

# ── Tab 1: Actual Climate Data (Primary Workflow) ─────────────────────
with tab_real:
    st.markdown(
        "**Fetch real climate data for any Kenyan county and time period. "
        "The system automatically pulls actual conditions and runs the analysis.**"
    )
    st.markdown("")
    
    arc1, arc2, arc3 = st.columns([1, 1, 1])
    with arc1:
        arc_county = st.selectbox("Select County", COUNTIES, index=COUNTIES.index("Nakuru"), key="arc_county")
    with arc2:
        from datetime import datetime
        current_year = datetime.now().year
        arc_year = st.number_input("Select Year", min_value=1940, max_value=current_year, value=2020, key="arc_year")
        if arc_year == current_year:
            st.caption(f"📅 Current year: Data available up to {datetime.now().strftime('%B %Y')}")
    with arc3:
        if option == "b":
            arc_month = st.number_input("Select Month (Option B only)", min_value=1, max_value=12, value=6, key="arc_month")
        else:
            arc_month = None
    
    # Fetch button
    if st.button("🌦️ Fetch & Predict", type="primary", use_container_width=True, key="btn_real"):
        with st.spinner("Fetching climate data from Open-Meteo…"):
            features = fetch_and_compute_features(arc_county, arc_year, option, arc_month)
        
        if features is None:
            from datetime import datetime
            current_year = datetime.now().year
            if arc_year == current_year:
                st.error(
                    f"❌ Could not fetch climate data for {arc_county} in {arc_year}.\n\n"
                    f"For current year ({current_year}), data is only available up to present month. "
                    f"Try a past year (1940-{current_year-1}) or wait for more data to become available."
                )
            else:
                st.error(
                    f"❌ Could not fetch climate data for {arc_county} in {arc_year}.\n\n"
                    f"Open-Meteo covers 1940-present. Please try a different year or county."
                )
        else:
            # Show extracted features
            with st.expander("📋 Extracted Climate Features", expanded=False):
                feat_df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
                st.dataframe(feat_df, use_container_width=True)
            
            # Run prediction
            result = predict(features, disease, option)
            if result:
                render_results(result, arc_county, arc_year, disease)
                # Generate explanation if API key available
                if api_key:
                    with st.spinner("🤖 Generating explanation…" if language == "English" else "🤖 Inaandaa maelezo…"):
                        model_param = openrouter_model if provider == "OpenRouter" and 'openrouter_model' in locals() else None
                        explanation = generate_explanation(result, arc_county, arc_year, disease, language, provider, api_key, model_param)
                        if explanation:
                            st.markdown(f'<div class="chat-ai">{explanation}</div>', unsafe_allow_html=True)
                else:
                    st.info("💡 Add an API key in the sidebar to get AI-powered explanations of results.")
            else:
                st.error(f"Model `{disease}_{option}` not loaded. Check data/models/.")


# ── Tab 2: Custom Scenario (Manual Sliders) ──────────────────────────
with tab_manual:
    st.markdown(
        "**Create custom 'what-if' scenarios** by manually adjusting climate parameters. "
        "Use this for sensitivity testing and exploring hypothetical conditions."
    )
    st.markdown("")
    
    mc1, mc2 = st.columns([1, 2])
    with mc1:
        county = st.selectbox("County", COUNTIES, index=COUNTIES.index("Nakuru"), key="custom_county")
        year = st.number_input("Year", min_value=2020, max_value=2030, value=2025, key="custom_year")

    with mc2:
        st.markdown("#### Climate Features")
        cfg = SLIDER_CFG_A if option == "a" else SLIDER_CFG_B
        feat_vals = {}
        cols = st.columns(3)
        for i, (feat, (label, mn, mx, default, step)) in enumerate(cfg.items()):
            with cols[i % 3]:
                if isinstance(mn, int):
                    feat_vals[feat] = st.slider(label, mn, mx, default, step, key=f"m_{feat}")
                else:
                    feat_vals[feat] = st.slider(label, mn, mx, default, step, key=f"m_{feat}")

        # For option B: compute month_sin/cos from a month picker
        if option == "b":
            month = st.slider("Month", 1, 12, 6, key="m_month")
            feat_vals["month_sin"] = round(math.sin(2 * math.pi * month / 12), 6)
            feat_vals["month_cos"] = round(math.cos(2 * math.pi * month / 12), 6)

    if st.button("🔬 Predict Risk", type="primary", use_container_width=True, key="btn_manual"):
        result = predict(feat_vals, disease, option)
        if result:
            render_results(result, county, year, disease)
            # Generate bilingual LLM explanation
            if api_key:
                with st.spinner("🤖 Generating explanation…" if language == "English" else "🤖 Inaandaa maelezo…"):
                    model_param = openrouter_model if provider == "OpenRouter" and 'openrouter_model' in locals() else None
                    explanation = generate_explanation(result, county, year, disease, language, provider, api_key, model_param)
                    if explanation:
                        label = "🤖 AI Explanation" if language == "English" else "🤖 Maelezo ya AI"
                        st.markdown(f'<div class="chat-ai">{explanation}</div>', unsafe_allow_html=True)
            else:
                st.info("💡 Add an API key in the sidebar and choose a provider to get AI-powered explanations of results.")
        else:
            st.error(f"Model `{disease}_{option}` not loaded. Check data/models/.")


# ── Tab 2: AI Assistant ──────────────────────────────────────────────
with tab_ai:
    intro_text = (
        "Describe a scenario in plain English or Kiswahili and the AI will extract climate features, "
        "run the prediction, and explain the results."
    )
    st.markdown(intro_text)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    placeholder_text = (
        "e.g. What's the cholera risk in Mombasa? It's been an extremely wet rainy season "
        "with about 1200mm of rain, temperatures around 28°C."
        if language == "English" else
        "mfano: Je, hatari ya kipindupindu ni ipi Mombasa? Kumekuwa na mvua nyingi sana, "
        "karibu 1200mm, na joto la wastani wa 28°C."
    )

    user_input = st.text_area(
        "Describe the scenario" if language == "English" else "Eleza hali ya hewa",
        placeholder=placeholder_text,
        height=100, key="ai_input",
    )

    btn_label = "🤖 Analyse" if language == "English" else "🤖 Changanua"
    if st.button(btn_label, type="primary", use_container_width=True, key="btn_ai"):
            if not api_key:
                st.warning("Please enter your provider API key in the sidebar." if language == "English"
                           else "Tafadhali ingiza ufunguo wa API wa mtumaji kwenye upau wa pembeni.")
            elif not user_input.strip():
                st.warning("Please describe a scenario." if language == "English"
                           else "Tafadhali eleza hali.")
            else:
                spinner_msg = "AI is extracting climate features…" if language == "English" else "AI inachambua data ya hali ya hewa…"
                with st.spinner(spinner_msg):
                    try:
                        prompt = build_extraction_prompt(user_input, option)
                        model_param = openrouter_model if provider == "OpenRouter" and 'openrouter_model' in locals() else None
                        raw = call_llm(prompt, provider, api_key, model_param)
                        # Strip markdown fences if present
                        clean = raw.strip()
                        if clean.startswith("```"):
                            clean = clean.split("\n", 1)[1]
                            clean = clean.rsplit("```", 1)[0]
                        parsed = json.loads(clean)

                        ai_county = parsed.get("county", "Nakuru").title()
                        ai_year = parsed.get("year", 2025)
                        ai_disease = parsed.get("disease", disease).lower()
                        ai_features = parsed.get("features", {})
                        reasoning = parsed.get("reasoning", "")

                        st.markdown(f'<div class="chat-user">💬 {user_input}</div>', unsafe_allow_html=True)

                        with st.expander("🔍 Extracted Features" if language == "English" else "🔍 Data Iliyochukuliwa", expanded=False):
                            st.json(ai_features)
                            if reasoning:
                                st.caption(f"ℹ️ {reasoning}")

                        result = predict(ai_features, ai_disease, option)
                        if result:
                            render_results(result, ai_county, ai_year, ai_disease)

                            # Generate detailed bilingual explanation
                            with st.spinner("🤖 Generating explanation…" if language == "English" else "🤖 Inaandaa maelezo…"):
                                model_param = openrouter_model if provider == "OpenRouter" and 'openrouter_model' in locals() else None
                                explanation = generate_explanation(result, ai_county, ai_year, ai_disease, language, provider, api_key, model_param)
                                st.markdown(f'<div class="chat-ai">{explanation}</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"Model `{ai_disease}_{option}` not loaded.")

                    except json.JSONDecodeError:
                        st.error("AI returned invalid JSON. Please try rephrasing." if language == "English"
                                 else "AI haikuweza kuchambua. Tafadhali jaribu tena.")
                        st.code(raw)
                    except Exception as e:
                        st.error(f"Error: {e}")


# ── Tab 3: Historical Data ───────────────────────────────────────────
with tab_history:
    st.markdown("Run predictions on actual historical climate data from your training dataset.")
    df_hist = load_historical()

    if df_hist is not None and option == "a":
        hc1, hc2 = st.columns(2)
        with hc1:
            h_county = st.selectbox("Select County", sorted(df_hist["county"].unique()),
                                     key="hist_county")
        with hc2:
            avail_years = sorted(df_hist[df_hist["county"] == h_county]["year"].unique())
            h_year = st.selectbox("Select Year", avail_years, key="hist_year")

        row = df_hist[(df_hist["county"] == h_county) & (df_hist["year"] == h_year)]

        if not row.empty:
            feat_dict = {f: float(row.iloc[0][f]) for f in FEATURES_A if f in row.columns}
            with st.expander("📋 Feature Values", expanded=False):
                st.dataframe(pd.DataFrame([feat_dict]).T.rename(columns={0: "Value"}))

            if st.button("🔬 Predict from Historical Data", type="primary",
                         use_container_width=True, key="btn_hist"):
                result = predict(feat_dict, disease, "a")
                if result:
                    render_results(result, h_county, h_year, disease)
                    # Generate bilingual LLM explanation
                    if api_key:
                        with st.spinner("🤖 Generating explanation…" if language == "English" else "🤖 Inaandaa maelezo…"):
                            model_param = openrouter_model if provider == "OpenRouter" and 'openrouter_model' in locals() else None
                            explanation = generate_explanation(result, h_county, h_year, disease, language, provider, api_key, model_param)
                            if explanation:
                                st.markdown(f'<div class="chat-ai">{explanation}</div>', unsafe_allow_html=True)
                    else:
                        st.info("💡 Add an API key in the sidebar and choose a provider to get AI-powered explanations.")

            # ── Historical Data Visualization ──────────────────────────────
            st.markdown("---")
            st.subheader("📊 Historical Data Visualization")

            # Get all years for this county
            county_data = df_hist[df_hist["county"] == h_county].sort_values("year")

            if len(county_data) > 1:
                # Chart 1: Rainfall vs Outbreaks Over Time
                st.markdown("**Rainfall and Outbreak Trends**")
                fig1 = go.Figure()

                # Add rainfall data
                fig1.add_trace(go.Scatter(
                    x=county_data["year"],
                    y=county_data["total_rainfall_mm"],
                    name="Total Rainfall (mm)",
                    mode="lines+markers",
                    line=dict(color="blue", width=2),
                    yaxis="y1"
                ))

                # Add outbreak data
                fig1.add_trace(go.Bar(
                    x=county_data["year"],
                    y=county_data["cholera_cases"],
                    name="Cholera Cases",
                    marker_color="red",
                    opacity=0.7,
                    yaxis="y2"
                ))

                fig1.add_trace(go.Bar(
                    x=county_data["year"],
                    y=county_data["malaria_cases"],
                    name="Malaria Cases",
                    marker_color="orange",
                    opacity=0.7,
                    yaxis="y2"
                ))

                fig1.update_layout(
                    title=f"Rainfall and Disease Cases in {h_county.title()}",
                    xaxis=dict(title="Year"),
                    yaxis=dict(
                        title=dict(text="Total Rainfall (mm)", font=dict(color="blue")),
                        tickfont=dict(color="blue")
                    ),
                    yaxis2=dict(
                        title=dict(text="Disease Cases", font=dict(color="red")),
                        tickfont=dict(color="red"),
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(x=0.1, y=1.1, orientation="h")
                )

                st.plotly_chart(fig1, use_container_width=True)

                # Chart 2: Temperature Trends
                st.markdown("**Temperature Trends and Outbreaks**")
                fig2 = go.Figure()

                # Temperature data
                fig2.add_trace(go.Scatter(
                    x=county_data["year"],
                    y=county_data["mean_temperature_c"],
                    name="Mean Temperature (°C)",
                    mode="lines+markers",
                    line=dict(color="green", width=2),
                    yaxis="y1"
                ))

                fig2.add_trace(go.Scatter(
                    x=county_data["year"],
                    y=county_data["max_temperature_c"],
                    name="Max Temperature (°C)",
                    mode="lines+markers",
                    line=dict(color="red", width=2, dash="dash"),
                    yaxis="y1"
                ))

                fig2.add_trace(go.Scatter(
                    x=county_data["year"],
                    y=county_data["min_temperature_c"],
                    name="Min Temperature (°C)",
                    mode="lines+markers",
                    line=dict(color="blue", width=2, dash="dash"),
                    yaxis="y1"
                ))

                # Outbreak data
                fig2.add_trace(go.Bar(
                    x=county_data["year"],
                    y=county_data["cholera_cases"],
                    name="Cholera Cases",
                    marker_color="purple",
                    opacity=0.6,
                    yaxis="y2"
                ))

                fig2.update_layout(
                    title=f"Temperature and Cholera Cases in {h_county.title()}",
                    xaxis=dict(title="Year"),
                    yaxis=dict(
                        title=dict(text="Temperature (°C)", font=dict(color="green")),
                        tickfont=dict(color="green")
                    ),
                    yaxis2=dict(
                        title=dict(text="Cholera Cases", font=dict(color="purple")),
                        tickfont=dict(color="purple"),
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(x=0.1, y=1.1, orientation="h")
                )

                st.plotly_chart(fig2, use_container_width=True)

                # Chart 3: Rainfall Variability vs Cases
                st.markdown("**Rainfall Variability and Disease Relationships**")
                fig3 = go.Figure()

                fig3.add_trace(go.Scatter(
                    x=county_data["rainfall_variability"],
                    y=county_data["cholera_cases"],
                    mode="markers",
                    name="Cholera vs Rainfall Variability",
                    marker=dict(
                        size=county_data["total_rainfall_mm"]/50,
                        color=county_data["year"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Year")
                    ),
                    text=[f"Year: {y}<br>Rainfall: {r:.1f}mm<br>Cases: {c}"
                          for y, r, c in zip(county_data["year"], county_data["total_rainfall_mm"], county_data["cholera_cases"])]
                ))

                fig3.update_layout(
                    title=f"Cholera Cases vs Rainfall Variability in {h_county.title()}",
                    xaxis=dict(title="Rainfall Variability"),
                    yaxis=dict(title="Cholera Cases"),
                    showlegend=False
                )

                st.plotly_chart(fig3, use_container_width=True)

                # Chart 4: Temperature Range vs Malaria
                st.markdown("**Temperature Range and Malaria Cases**")
                fig4 = go.Figure()

                fig4.add_trace(go.Scatter(
                    x=county_data["temp_range_c"],
                    y=county_data["malaria_cases"],
                    mode="markers+text",
                    name="Malaria vs Temperature Range",
                    marker=dict(
                        size=county_data["mean_temperature_c"],
                        sizemode="area",
                        sizeref=2.*max(county_data["mean_temperature_c"])/(40.**2),
                        sizemin=4,
                        color=county_data["year"],
                        colorscale="Plasma",
                        showscale=True,
                        colorbar=dict(title="Year")
                    ),
                    text=[f"Year: {y}<br>Temp Range: {tr:.1f}°C<br>Cases: {c}"
                          for y, tr, c in zip(county_data["year"], county_data["temp_range_c"], county_data["malaria_cases"])],
                    textposition="top center"
                ))

                fig4.update_layout(
                    title=f"Malaria Cases vs Temperature Range in {h_county.title()}",
                    xaxis=dict(title="Temperature Range (°C)"),
                    yaxis=dict(title="Malaria Cases"),
                    showlegend=False
                )

                st.plotly_chart(fig4, use_container_width=True)

                # Chart 5: Monthly Rainfall Pattern
                st.markdown("**Peak Rainfall Month Distribution**")
                peak_month_counts = county_data["peak_rainfall_month"].value_counts().sort_index()

                fig5 = go.Figure()

                fig5.add_trace(go.Bar(
                    x=[f"Month {i}" for i in peak_month_counts.index],
                    y=peak_month_counts.values,
                    marker_color="lightblue",
                    name="Peak Rainfall Month"
                ))

                fig5.update_layout(
                    title=f"Peak Rainfall Months in {h_county.title()}",
                    xaxis=dict(title="Month"),
                    yaxis=dict(title="Frequency"),
                    showlegend=False
                )

                st.plotly_chart(fig5, use_container_width=True)

            else:
                st.info("📊 Visualization requires data from multiple years. This county only has data for one year.")

    elif option == "b":
        st.info("Historical data view is available for Option A only. Option B data can be explored via manual input.")
    else:
        st.warning("Historical data file not found at `data/processed/option_a_features.csv`.")
