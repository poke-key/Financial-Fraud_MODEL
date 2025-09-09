import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- Paths ---
MODEL_PATH = "models/fraud_xgb_tuned.pkl"
CATEGORICAL_COLS = ['ProductCD', 'card4', 'card6', 'DeviceType', 'id_30', 'id_31', 'P_emaildomain', 'R_emaildomain']

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# --- Encode Categorical Columns ---
def encode_categoricals(df):
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# --- Prediction Function ---
def predict_fraud(df, model):
    df = encode_categoricals(df)
    predictions = model.predict(df)
    df['prediction'] = predictions
    return df

# --- Streamlit App ---
st.title("💳 Financial Fraud Detection")
st.write("Upload a CSV file and the model will predict fraudulent transactions.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:")
    st.dataframe(input_df.head())

    try:
        model = load_model()
        result_df = predict_fraud(input_df, model)

        st.write("📊 Predictions:")
        st.dataframe(result_df)

        # Visualization
        fig, ax = plt.subplots()
        result_df['prediction'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xticklabels(['Not Fraud (0)', 'Fraud (1)'], rotation=0)
        ax.set_ylabel("Count")
        ax.set_title("Fraud vs Not Fraud Predictions")
        st.pyplot(fig)

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
