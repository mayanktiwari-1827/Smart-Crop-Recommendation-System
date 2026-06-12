import pandas as pd
import numpy as np
import pickle

# LOAD MODELS
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/log_model.pkl", "rb") as f:
    log_model = pickle.load(f)

with open("models/dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

# LOAD HELPERS
with open("data/categories.pkl", "rb") as f:
    allowed_values = pickle.load(f)

with open("data/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


# MAIN FUNCTION
def recommend_crop(input_dict, model_name="rf"):
    """
    input_dict: dictionary of user inputs
    model_name: 'rf', 'log', or 'dt'
    """

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Select model
    if model_name == "rf":
        model = rf_model
    elif model_name == "log":
        model = log_model
    elif model_name == "dt":
        model = dt_model
    else:
        raise ValueError("Invalid model_name. Choose from 'rf', 'log', 'dt'.")

    # Predict probabilities
    probs = model.predict_proba(input_df)[0]

    # Top 3 predictions
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_probs = probs[top3_indices]

    # Decode crop names
    crops = le.inverse_transform(top3_indices)

    # Prepare results
    # Prepare results
    results = []

    top1 = top3_probs[0]
    top2 = top3_probs[1]

    for i, (crop, prob) in enumerate(zip(crops, top3_probs)):

        if i == 0:
        # Top recommendation gets confidence score
            score = 70 + ((top1 - top2) / top1) * 30
            score = min(score, 98)
            score = round(score, 2)

        elif i == 1:
            score = round(score - 15, 2)

        else:
            score = round(score - 30, 2)

        results.append({
            "crop": crop,
            "probability": score
        })

    # Confidence gap
    confidence_gap = None
    confidence_level = None

    if len(top3_probs) >= 2:
        confidence_gap = round((top3_probs[0] - top3_probs[1]) * 100, 2)

        if confidence_gap > 30:
            confidence_level = "High"
        elif confidence_gap > 10:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

    return {
        "top_3": results,
        "confidence_gap": confidence_gap,
        "confidence_level": confidence_level
    }


def get_allowed_values():
    """
    Returns allowed categorical values for UI dropdowns
    """
    return allowed_values
    
import shap

def get_shap_explanation(input_dict, model_name="rf"):

    input_df = pd.DataFrame([input_dict])

    # Select model
    if model_name == "rf":
        pipeline = rf_model
    elif model_name == "log":
        pipeline = log_model
    elif model_name == "dt":
        pipeline = dt_model
    else:
        raise ValueError("Invalid model_name")

    # Split pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Transform input
    input_transformed = preprocessor.transform(input_df)

#  FIX HERE
    if hasattr(input_transformed, "toarray"):
        input_transformed = input_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    input_transformed_df = pd.DataFrame(input_transformed, columns=feature_names)

    #  SHAP explainer (Tree for RF/DT)
    if model_name in ["rf", "dt"]:
        # SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_transformed_df)

        # Predicted class
        pred_class = model.predict(input_transformed_df)[0]
        class_index = list(model.classes_).index(pred_class)

        #  FIXED HANDLING
        if isinstance(shap_values, list):
            values = shap_values[class_index][0]

        elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
            values = shap_values[0][:, class_index]

        else:
            values = shap_values[0]

    else:
        # Logistic regression
        explainer = shap.LinearExplainer(model, input_transformed_df)
        shap_values = explainer.shap_values(input_transformed_df)

        values = shap_values[0]

    # Create dataframe
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "impact": values
    })

    shap_df["abs_impact"] = shap_df["impact"].abs()
    shap_df = shap_df.sort_values(by="abs_impact", ascending=False)

    shap_df["feature"] = shap_df["feature"].str.replace("cat__", "")
    shap_df["feature"] = shap_df["feature"].str.replace("num__", "")
    shap_df["feature"] = shap_df["feature"].str.replace("remainder__", "")

    # ---------------- GROUP SEASON FEATURES ----------------
    season_rows = shap_df[shap_df["feature"].str.contains("SEASON")]

    if not season_rows.empty:
        season_impact = season_rows["impact"].sum()

    # Remove individual season rows
        shap_df = shap_df[~shap_df["feature"].str.contains("SEASON")]

    # Add combined row
        shap_df = pd.concat([
            shap_df,
            pd.DataFrame([{"feature": "SEASON", "impact": season_impact}])
        ], ignore_index=True)

    soil_rows = shap_df[shap_df["feature"].str.contains("SOIL_")]

    if not soil_rows.empty:
        soil_impact = soil_rows["impact"].sum()
        shap_df = shap_df[~shap_df["feature"].str.contains("SOIL_")]

        shap_df = pd.concat([
            shap_df,
            pd.DataFrame([{"feature": "SOIL", "impact": soil_impact}])
        ], ignore_index=True)

    shap_df["abs_impact"] = shap_df["impact"].abs()
    shap_df = shap_df.sort_values(by="abs_impact", ascending=False)

    return shap_df

def get_logistic_explanation(input_dict):

    input_df = pd.DataFrame([input_dict])

    pipeline = log_model
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Transform
    X = preprocessor.transform(input_df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # Get coefficients
    coefs = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "impact": coefs
    })

    coef_df["abs_impact"] = coef_df["impact"].abs()
    coef_df = coef_df.sort_values(by="abs_impact", ascending=False)

    coef_df["feature"] = coef_df["feature"].str.replace("cat__", "")
    coef_df["feature"] = coef_df["feature"].str.replace("num__", "")
    coef_df["feature"] = coef_df["feature"].str.replace("remainder__", "")

    return coef_df