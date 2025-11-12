import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Beverage Price Prediction", layout="wide", initial_sidebar_state="collapsed")

# Load CSS
css_path = Path("assets/style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

PIPELINE_FILE = "best_pipeline_XGBoost.pkl"
LABEL_ENCODER_FILE = "label_encoder_target.pkl"

@st.cache_resource
def load_model():
    pipeline = joblib.load(PIPELINE_FILE)
    le_target = joblib.load(LABEL_ENCODER_FILE)
    return pipeline, le_target

pipeline, le_target = load_model()

def categorize_age_group(age):
    if 18 <= age <= 25: return "18-25"
    if 26 <= age <= 35: return "26-35"
    if 36 <= age <= 45: return "36-45"
    if 46 <= age <= 55: return "46-55"
    if 56 <= age <= 70: return "56-70"
    if age > 70: return "70+"
    return "Unknown"

# ---- Hero Header ----
st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">🥤 Beverage Price Prediction</h1>
            <p class="hero-subtitle">Transform consumer insights into optimal pricing strategies with machine learning</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ===============================================
# --- Unified Form Container ---
# ===============================================
st.markdown('<div class="form-container">', unsafe_allow_html=True)

# Tab-style navigation
st.markdown("""
    <div class="tab-navigation">
        <div class="tab-item active">
            <div class="tab-icon">🎯</div>
            <div class="tab-label">Demographics</div>
        </div>
        <div class="tab-connector"></div>
        <div class="tab-item active">
            <div class="tab-icon">💳</div>
            <div class="tab-label">Behavior Patterns</div>
        </div>
        <div class="tab-connector"></div>
        <div class="tab-item active">
            <div class="tab-icon">⚡</div>
            <div class="tab-label">Market Position</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---- Demographics Section ----
st.markdown("""
    <div class="input-section demographics-section">
        <div class="section-header">
            <span class="section-icon">🎯</span>
            <div>
                <h2 class="section-heading">Consumer Demographics</h2>
                <p class="section-description">Core profile attributes for market segmentation</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.number_input("👤 Age", min_value=18, max_value=100, value=25, key="age_input", help="Consumer age in years")

with col2:
    gender = st.selectbox("⚧ Gender", ["Male","Female","Other"], key="gender_select", help="Gender identity")

with col3:
    zone = st.selectbox("🏙️ Geographic Zone", ["Urban", "Metro", "Rural", "Semi-Urban"], key="zone_select", help="Primary location type")

with col4:
    income_levels = st.selectbox("💰 Annual Income", ["<10L","10L-15L","16L-25L","26L-35L",">35L","Not Reported"], key="income_select", help="Household income bracket")

col5, col6 = st.columns(2)
with col5:
    occupation = st.selectbox("💼 Occupation Type", ["Student","Employed","Entrepreneur","Unemployed"], key="occupation_select", help="Primary occupation")

with col6:
    health_concerns = st.selectbox("🏃 Wellness Focus", ["Low","Medium","High"], key="health_select", help="Health consciousness level")

age_group = categorize_age_group(age)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---- Behavior Patterns Section ----
st.markdown("""
    <div class="input-section behavior-section">
        <div class="section-header">
            <span class="section-icon">💳</span>
            <div>
                <h2 class="section-heading">Purchase Behavior Patterns</h2>
                <p class="section-description">Consumption habits and preferences analysis</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

col7, col8, col9 = st.columns(3)
with col7:
    consume_frequency = st.selectbox("📊 Weekly Frequency", ["0-2 times","3-4 times","5-7 times"], key="freq_select", help="Average weekly consumption")

with col8:
    typical_consumption_situations = st.selectbox("📍 Primary Location", ["Home","Work","Outdoors","Other"], key="situation_select", help="Most common consumption setting")

with col9:
    preferable_consumption_size = st.selectbox("📏 Preferred Volume", ["Small","Medium","Large"], key="size_select", help="Typical purchase size")

col10, col11, col12 = st.columns(3)
with col10:
    packaging_preference = st.selectbox("📦 Package Type", ["Bottle","Can","Tetra Pack","Other"], key="package_select", help="Preferred packaging format")

with col11:
    purchase_channel = st.selectbox("🛍️ Purchase Channel", ["Online","Retail","Supermarket","Other"], key="channel_select", help="Primary buying platform")

with col12:
    flavor_preference = st.selectbox("🍹 Flavor Profile", ["Sweet","Sour","Bitter","Mixed","Other"], key="flavor_select", help="Taste preference")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---- Market Position Section ----
st.markdown("""
    <div class="input-section market-section">
        <div class="section-header">
            <span class="section-icon">⚡</span>
            <div>
                <h2 class="section-heading">Market Position & Brand Dynamics</h2>
                <p class="section-description">Brand perception and competitive landscape</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

col13, col14, col15 = st.columns(3)
with col13:
    current_brand = st.selectbox("🏆 Brand Category", ["Established","Newcomer"], key="brand_type_select", help="Current brand market position")

with col14:
    brand_awareness = st.selectbox("📢 Brand Recognition", ["Low","Medium","High"], key="awareness_select", help="Consumer brand awareness level")

with col15:
    primary_selection_factor = st.selectbox("🎯 Decision Driver", ["Price","Quality","Other"], key="factor_select", help="Primary purchase factor")

st.markdown('</div>', unsafe_allow_html=True)

# ---- Feature Engineering (LOGIC UNTOUCHED) ----
frequency_mapping = {"0-2 times":1, "3-4 times":2, "5-7 times":3}
awareness_mapping = {"Low":1, "Medium":2, "High":3}
zone_mapping = {"Urban":3, "Metro":4, "Rural":1, "Semi-Urban":2}
income_mapping = {"<10L":1, "10L-15L":2, "16L-25L":3, "26L-35L":4, ">35L":5, "Not Reported":0}

cf_val = frequency_mapping[consume_frequency]
ab_val = awareness_mapping[brand_awareness]
zone_val = zone_mapping[zone]
income_val = income_mapping[income_levels]

cf_ab_score = round(cf_val / (cf_val + ab_val), 2)
zas_score = zone_val * income_val
bsi = int((current_brand != "Established") and (primary_selection_factor in ["Price","Quality"]))

# Build raw DataFrame
input_raw = pd.DataFrame([{
    "age": age,
    "age_group": age_group,
    "zone": zone,
    "income_levels": income_levels,
    "gender": gender,
    "occupation": occupation,
    "health_concerns": health_concerns,
    "consume_frequency(weekly)": consume_frequency,
    "typical_consumption_situations": typical_consumption_situations,
    "packaging_preference": packaging_preference,
    "preferable_consumption_size": preferable_consumption_size,
    "purchase_channel": purchase_channel,
    "flavor_preference": flavor_preference,
    "current_brand": current_brand,
    "brand_awareness": brand_awareness,
    "awareness_of_other_brands": brand_awareness,
    "reasons_for_choosing_brands": primary_selection_factor,
    "cf_ab_score": cf_ab_score,
    "zas_score": zas_score,
    "bsi": bsi
}])

# ---- Ensure correct dtypes (LOGIC UNTOUCHED) ----
preprocessor = pipeline.named_steps['preprocessor']
num_cols, cat_cols = [], []

for name, transformer, cols in preprocessor.transformers_:
    if name=='num': num_cols = cols
    else: cat_cols.extend(cols)

for col in num_cols:
    if col in input_raw.columns: input_raw[col] = pd.to_numeric(input_raw[col], errors='coerce').fillna(0)
for col in cat_cols:
    if col in input_raw.columns: input_raw[col] = input_raw[col].astype(str).fillna("Unknown")

# ---- Prediction CTA ----
st.markdown('<div class="cta-section"></div>', unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    if st.button("⚡ GENERATE PREDICTION", key="predict_button", use_container_width=True):
        try:
            with st.spinner('🔄 Analyzing...'):
                preds = pipeline.predict(input_raw)
                pred_label = le_target.inverse_transform(preds)[0]
            
            st.markdown(f"""
                <div class="result-container">
                    <div class="result-content">
                        <div class="result-label">OPTIMAL PRICE RANGE</div>
                        <div class="result-value">{pred_label}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")


# ---- Advanced Analytics Panel ----
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
    <div class="analytics-header">
        <h3>📈 Advanced Analytics</h3>
        <p>Review complete input data</p>
    </div>
""", unsafe_allow_html=True)

with st.expander("🔍 View Complete Data Profile", expanded=False):
    st.dataframe(input_raw.T, use_container_width=True)

# Footer
st.markdown("""
    <div class="footer-container">
        <div class="footer-content">
            <div class="footer-brand">
                <h4>Beverage Price Prediction & Analysis</h4>
                <p>Powered by XGBoost • Built with Streamlit</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)