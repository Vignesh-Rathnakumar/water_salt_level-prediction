import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==============================
# 🎯 Load Model Files
# ==============================
model = joblib.load("model/salt_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

# ==============================
# 🌊 Streamlit Page Config
# ==============================
st.set_page_config(page_title="💧 Water Salt Level Predictor", layout="wide")

# ==============================
# 💧 Title & Description
# ==============================
st.markdown("""
    <div style="text-align:center">
        <h1 style="color:#2E86C1;">💧 Water Salt Level Prediction</h1>
        <h4 style="color:#5D6D7E;">Predict water salinity and drinkability using key water quality parameters</h4>
        <hr style="border:1px solid #2E86C1;">
    </div>
""", unsafe_allow_html=True)

# ==============================
# 🧪 Input Section
# ==============================
st.subheader("🌿 Enter Water Properties")

cols = st.columns(3)
input_data = {}

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.01)

# ==============================
# 🔮 Prediction Button
# ==============================
if st.button("🔮 Predict Salt Level", use_container_width=True, type="primary"):
    try:
        df_input = pd.DataFrame([input_data])
        X_scaled = scaler.transform(df_input)
        pred = model.predict(X_scaled)[0]

        # Drinkability threshold
        drinkable = pred < 500

        # ==============================
        # 📊 Result Display Section
        # ==============================
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🧾 Prediction Summary")
            st.success(f"**Predicted Salt Level:** {pred:.2f} mg/L")

            if drinkable:
                st.markdown("<h3 style='color:green;'>✅ Water is Drinkable</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color:red;'>🚫 Water is Not Drinkable</h3>", unsafe_allow_html=True)

            # ---- Progress / Gauge style bar ----
            st.markdown("### ⚙️ Salt Level Indicator")
            progress = min(pred / 1000, 1.0)
            st.progress(progress)
            st.caption(f"Salt Concentration Level: {progress*100:.1f}% of unsafe limit")

        # ==============================
        # 📈 Graph Section
        # ==============================
        with col2:
            st.markdown("### 📊 Feature Overview")
            fig, ax = plt.subplots(figsize=(4, 4))
            top_features = dict(list(input_data.items())[:5])  # Top 5 features
            ax.barh(list(top_features.keys()), list(top_features.values()), color='#5DADE2')
            ax.set_xlabel("Value")
            ax.set_title("Top 5 Input Parameters")
            st.pyplot(fig)

        # ==============================
        # 💬 Extra Info
        # ==============================
        st.markdown("""
        ---
        <div style="text-align:center; color:grey;">
        <b>Note:</b> This prediction is based on machine learning trained with physical & chemical water properties.  
        Ensure laboratory testing for accurate analysis.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ==============================
# 🧭 Footer
# ==============================
st.markdown("""
---
<div style="text-align:center; color:#85929E;">
Developed with ❤️ using <b>Streamlit</b> and <b>Machine Learning</b>  
</div>
""", unsafe_allow_html=True)
