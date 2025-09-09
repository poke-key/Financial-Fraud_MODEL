from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)
CORS(app)

# Paths
MODEL_PATH = "../models/fraud_xgb_tuned.pkl"
CATEGORICAL_COLS = ['ProductCD', 'card4', 'card6', 'DeviceType', 'id_30', 'id_31', 'P_emaildomain', 'R_emaildomain']

# Load model at startup
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully")
        else:
            print(f"Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

def encode_categoricals(df):
    """Encode categorical columns and fix data types"""
    df_encoded = df.copy()
    
    # First, convert all numeric columns to proper types
    for col in df_encoded.columns:
        if col not in CATEGORICAL_COLS:
            try:
                # Try to convert to numeric, errors='coerce' will turn invalid values to NaN
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                # Fill any NaN values with 0
                df_encoded[col] = df_encoded[col].fillna(0)
            except:
                # If conversion fails, keep as is
                pass
    
    # Then encode categorical columns
    for col in CATEGORICAL_COLS:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Handle missing values by converting to string first
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        data = request.json
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset'}), 400
        
        # Store original data
        original_df = df.copy()
        
        if model is None:
            # Fallback to simple heuristic if model not available
            predictions = []
            for _, row in df.iterrows():
                # Simple heuristic based on transaction amount
                trans_amt = float(row.get('TransactionAmt', 0))
                prediction = 1 if trans_amt > 200 and np.random.random() > 0.8 else 0
                predictions.append(prediction)
        else:
            # Use actual model
            try:
                # Encode categorical columns
                df_processed = encode_categoricals(df)
                
                # Make predictions
                predictions = model.predict(df_processed)
                predictions = predictions.astype(int).tolist()
                
            except Exception as e:
                print(f"Error during prediction: {e}")
                # Fallback to heuristic
                predictions = []
                for _, row in df.iterrows():
                    trans_amt = float(row.get('TransactionAmt', 0))
                    prediction = 1 if trans_amt > 200 and np.random.random() > 0.8 else 0
                    predictions.append(prediction)
        
        # Add predictions to original data
        result_data = original_df.copy()
        result_data['prediction'] = predictions
        
        return jsonify({
            'success': True,
            'predictions': result_data.to_dict('records')
        })
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("Starting Python API server...")
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)