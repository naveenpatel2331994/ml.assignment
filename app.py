import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Marketing Model Explorer", layout="wide")

ROOT = Path(__file__).parent

RESULTS_PATH = ROOT / "results_all.json"
MODELS_DIR = ROOT / "model"  # For Streamlit Cloud deployment
ROC_DIR = ROOT  # ROC curves are at root level


def interpret_mcc(mcc_value):
    if mcc_value >= 0.8:
        return "Excellent agreement (0.8+)"
    elif mcc_value >= 0.6:
        return "Strong agreement (0.6-0.8)"
    elif mcc_value >= 0.4:
        return "Moderate agreement (0.4-0.6)"
    elif mcc_value >= 0.2:
        return "Fair agreement (0.2-0.4)"
    elif mcc_value >= 0:
        return "Slight agreement (0-0.2)"
    else:
        return "Poor/Inverse agreement (<0)"


@st.cache_data
def load_results():
    try:
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        st.error(f"Error loading results file: {e}")
        return {}


@st.cache_resource
def load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


# Expected columns for model input (features only, excluding target 'y')
EXPECTED_COLUMNS = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                    'housing', 'loan', 'contact', 'day', 'month', 'duration', 
                    'campaign', 'pdays', 'previous', 'poutcome']


def validate_columns(df):
    """Validate that uploaded CSV has expected columns."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    
    if missing:
        return False, f"Missing required columns: {sorted(missing)}"
    if extra:
        return False, f"Unknown columns (will be ignored): {sorted(extra)}"
    return True, "All required columns present"


def get_sample_csv():
    """Generate a sample CSV for users to download."""
    sample_data = {
        'age': [30, 45, 35],
        'job': ['management', 'blue-collar', 'technician'],
        'marital': ['married', 'married', 'single'],
        'education': ['tertiary', 'secondary', 'tertiary'],
        'default': ['no', 'no', 'no'],
        'balance': [1000, 500, 2000],
        'housing': ['yes', 'no', 'yes'],
        'loan': ['no', 'yes', 'no'],
        'contact': ['cellular', 'telephone', 'cellular'],
        'day': [5, 10, 15],
        'month': ['may', 'jun', 'jul'],
        'duration': [100, 200, 150],
        'campaign': [2, 1, 3],
        'pdays': [-1, 30, 90],
        'previous': [0, 1, 2],
        'poutcome': ['unknown', 'failure', 'success']
    }
    return pd.DataFrame(sample_data).to_csv(index=False)


def main():
    st.title("Bank Marketing — Model Explorer")

    results = load_results()

    if not results:
        st.error("results_all.json not found in repo. Upload reports/models_all folder to GitHub.")
        st.stop()

    model_names = list(results.keys())
    selected = st.sidebar.selectbox("Select model", model_names)

    metrics = results[selected]
    st.header(selected.replace("_", " ").title())

    st.subheader("Metrics")
    metrics_display = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    st.json(metrics_display)

    if "mcc" in metrics:
        st.success(f"MCC: {metrics['mcc']:.4f} — {interpret_mcc(metrics['mcc'])}")

    if "confusion_matrix" in metrics:
        st.subheader("Confusion Matrix")
        fig = plot_confusion(metrics["confusion_matrix"])
        st.pyplot(fig)

    roc_path = ROC_DIR / f"roc_{selected}.png"
    if roc_path.exists():
        st.subheader("ROC Curve")
        st.image(str(roc_path))
    else:
        st.warning("ROC image not found")

    model_file = MODELS_DIR / f"{selected}.joblib"
    if model_file.exists():
        st.subheader("Batch Predictions")
        
        # Show expected columns
        with st.expander("Expected CSV columns", expanded=False):
            st.write("Your CSV must have these columns:")
            st.code(", ".join(EXPECTED_COLUMNS))
            st.download_button(
                "Download sample CSV",
                get_sample_csv(),
                "sample_bank_data.csv",
                "text/csv"
            )
        
        uploaded = st.file_uploader("Upload CSV with features", type=["csv"])

        if uploaded:
            df = pd.read_csv(uploaded)
            
            # Validate columns
            is_valid, message = validate_columns(df)
            
            if not is_valid:
                st.error(f"Column validation failed: {message}")
                st.info("Please upload a CSV with the correct columns or download the sample CSV above.")
                st.stop()
            elif "Unknown columns" in message:
                st.warning(message + " - these columns will be ignored.")
            
            # Select only expected columns (in correct order)
            df = df[EXPECTED_COLUMNS]
            
            model = load_model(model_file)

            if model:
                try:
                    preds = model.predict(df)

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(df)[:, 1]
                        out = pd.DataFrame({"prediction": preds, "probability": probs})
                    else:
                        out = pd.DataFrame({"prediction": preds})

                    st.dataframe(out.head(50))
                    st.download_button(
                        "Download predictions",
                        out.to_csv(index=False),
                        "predictions.csv",
                    )
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info("This may be due to incorrect data types. Check that your values match the expected format.")
    else:
        st.warning("Model file not found in repo")


if __name__ == "__main__":
    main()
