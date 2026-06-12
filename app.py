import streamlit as st
from utils.recommend_crop import recommend_crop, get_allowed_values, get_shap_explanation, get_logistic_explanation

st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Crop Recommendation System")

# ---------------- LOAD DROPDOWN VALUES ----------------
allowed_values = get_allowed_values()

# ---------------- USER INPUT ----------------
st.header("Enter Farm Details")

col1, col2 = st.columns(2)

with col1:
    soil = st.selectbox("Soil Type", allowed_values["soil"])
    season = st.selectbox("Season", allowed_values["season"])
    cropduration = st.number_input("Crop Duration (days)", value=120.0)
    # sown = st.selectbox("Sown Month", allowed_values["sown"])
    # harvested = st.selectbox("Harvest Month", allowed_values["harvested"])

with col2:
    water_source = st.selectbox("Water Source", allowed_values["water_source"])
    soil_ph = st.number_input("Soil pH", value=6.5)
    # cropduration = st.number_input("Crop Duration (days)", value=120.0)
    temp = st.number_input("Temperature (°C)", value=25.0)

st.subheader("Soil Nutrients")

col3, col4, col5 = st.columns(3)

with col3:
    N_level = st.selectbox(
        "Nitrogen (N)",
        ["Low (20-60)", "Medium (61-120)", "High (121-200)"]
    )

with col4:
    P_level = st.selectbox(
        "Phosphorus (P)",
        ["Low (20-40)", "Medium (41-70)", "High (71-100)"]
    )

with col5:
    K_level = st.selectbox(
        "Potassium (K)",
        ["Low (20-50)", "Medium (51-90)", "High (91-150)"]
    )
    
relative_humidity = st.slider("Relative Humidity (%)", 0, 100, 70)
waterrequired = st.number_input("Water Required (litres)", value=500.0)

# ---------------- MODEL SELECTION ----------------
model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "Logistic Regression", "Decision Tree"]
)

model_map = {
    "Random Forest": "rf",
    "Logistic Regression": "log",
    "Decision Tree": "dt"
}

def convert_n(val):
    if "Low" in val:
        return 40
    elif "Medium" in val:
        return 90
    else:
        return 160

def convert_p(val):
    if "Low" in val:
        return 30
    elif "Medium" in val:
        return 55
    else:
        return 85

def convert_k(val):
    if "Low" in val:
        return 35
    elif "Medium" in val:
        return 70
    else:
        return 120

N = convert_n(N_level)
P = convert_p(P_level)
K = convert_k(K_level)

# ---------------- PREDICTION ----------------
if st.button("🌱 Recommend Crop"):

    input_data = {
        "SOIL": soil,
        "SEASON": season,
        "WATER_SOURCE": water_source,
        "SOIL_PH": soil_ph,
        "CROPDURATION": cropduration,
        "TEMP": temp,
        "WATERREQUIRED": waterrequired,
        "RELATIVE_HUMIDITY": relative_humidity,
        "N": N,
        "P": P,
        "K": K
    }

    result = recommend_crop(input_data, model_map[model_choice])

    st.success("Top 3 Recommended Crops")

    for i, item in enumerate(result["top_3"], start=1):
        st.write(f"{i}. 🌾 {item['crop']} → {item['probability']}%")

    st.markdown("---")

    st.write(f"**Confidence Gap:** {result['confidence_gap']}%")
    st.write(f"**Confidence Level:** {result['confidence_level']}")

    import matplotlib.pyplot as plt

# ---------------- MODEL-SPECIFIC EXPLANATION ----------------
    if model_map[model_choice] in ["rf", "dt"]:

        st.markdown("### 📊 Why this crop was recommended (SHAP)")
        shap_df = get_shap_explanation(input_data, model_map[model_choice])

    else:

        st.markdown("### 📊 Feature Influence (Logistic Regression)")
        shap_df = get_logistic_explanation(input_data)


# ---------------- COMMON GRAPH ----------------
    top_features = shap_df.head(8)

    fig, ax = plt.subplots()

    colors = ["green" if val > 0 else "red" for val in top_features["impact"]]

    ax.barh(top_features["feature"], top_features["impact"], color=colors)
    ax.set_xlabel("Impact on Prediction")
    ax.set_title("Feature Influence")

    st.pyplot(fig)

    st.info("Green = positive impact, Red = negative impact")