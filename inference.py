import argparse
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Paths
MODEL_PATH = "models/fraud_xgb_tuned.pkl"
CATEGORICAL_COLS = ['ProductCD', 'card4', 'card6', 'DeviceType', 'id_30', 'id_31', 'P_emaildomain', 'R_emaildomain']

def encode_categoricals(df):
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def predict_fraud(df):
    # Load model
    model = joblib.load(MODEL_PATH)

    # Preprocess
    df = encode_categoricals(df)

    # Predict
    predictions = model.predict(df)
    df['prediction'] = predictions
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    args = parser.parse_args()

    input_path = args.input
    print(f"📂 Reading input from: {input_path}")

    try:
        df = pd.read_csv(input_path)
        result_df = predict_fraud(df)

        output_path = "predictions.csv"
        result_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()
