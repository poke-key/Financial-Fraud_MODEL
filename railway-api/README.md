# Fraud Detection API - Railway Deployment

A Flask API service for financial fraud detection using XGBoost machine learning model, designed for Railway deployment.

## Features

- **Fraud Prediction**: REST API endpoint for batch fraud detection
- **Health Check**: Monitoring endpoint for deployment health
- **Model Loading**: Automatic XGBoost model loading with fallback logic
- **CORS Enabled**: Cross-origin requests supported for web applications
- **Railway Ready**: Configured for seamless Railway deployment

## API Endpoints

### `GET /`
Returns API information and status.

**Response:**
```json
{
  "message": "Fraud Detection API",
  "status": "running", 
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `GET /health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`
Main prediction endpoint for fraud detection.

**Request Body:**
```json
{
  "data": [
    {
      "TransactionAmt": 100.50,
      "ProductCD": "C",
      "card4": "visa",
      "card6": "debit",
      "DeviceType": "desktop",
      "id_30": "Android",
      "id_31": "samsung",
      "P_emaildomain": "gmail.com",
      "R_emaildomain": "yahoo.com"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "TransactionAmt": 100.50,
      "ProductCD": "C",
      "card4": "visa",
      "prediction": 0
    }
  ],
  "model_used": true
}
```

## Model Information

- **Model Type**: XGBoost Classifier
- **File**: `models/fraud_xgb_tuned.pkl`
- **Features**: Supports categorical encoding for financial transaction features
- **Fallback**: Simple heuristic-based prediction if model loading fails

## Deployment Structure

```
fraud-detection-api-railway/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Procfile                  # Railway/Heroku process file
├── railway.toml              # Railway deployment configuration
├── models/
│   └── fraud_xgb_tuned.pkl   # XGBoost model file
├── data/
│   ├── sample_data_5.csv     # Sample test data
│   ├── sample_data_100.csv   # Larger test dataset
│   └── sample_data_100_display.csv
└── README.md                 # This file
```

## Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

## Railway Deployment

### Prerequisites
- Railway account
- Git repository with this code

### Deploy Steps

1. **Create new Railway project:**
   ```bash
   railway login
   railway new
   ```

2. **Connect to repository:**
   - Link your GitHub repository containing this code
   - Railway will auto-detect Python and use the configuration

3. **Environment Variables:**
   - `PORT`: Automatically set by Railway (default: 8000)
   - No additional environment variables required

4. **Deployment:**
   - Railway will automatically deploy using `railway.toml` configuration
   - Health check endpoint: `/health`
   - The API will be available at: `https://your-app.railway.app`

### Railway Configuration

The `railway.toml` file configures:
- **Builder**: Nixpacks for Python applications
- **Health Check**: `/health` endpoint with 100s timeout
- **Restart Policy**: Restart on failure with max 10 retries
- **Port**: 8000 (standard for Railway Python apps)

## Integration with Next.js Frontend

After deploying to Railway, update your Next.js application:

1. **Create `.env.local` in your Next.js project:**
   ```env
   PYTHON_API_URL=https://your-railway-app.railway.app
   ```

2. **The Next.js API route will automatically use the Railway URL:**
   ```typescript
   const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:5000'
   ```

## Testing

Test the deployed API:

```bash
# Health check
curl https://your-railway-app.railway.app/health

# Prediction test
curl -X POST https://your-railway-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "TransactionAmt": 150.0,
        "ProductCD": "C",
        "card4": "visa"
      }
    ]
  }'
```

## Monitoring

- **Health Check**: Available at `/health` for uptime monitoring
- **Logs**: View in Railway dashboard for debugging
- **Metrics**: Railway provides automatic performance monitoring

## Security

- **CORS**: Configured to allow cross-origin requests
- **Environment**: Production-ready with gunicorn WSGI server
- **Error Handling**: Graceful error responses without exposing internals

## Cost Optimization

- **Free Tier**: Railway provides generous free tier for small applications
- **Sleep Mode**: Inactive services automatically sleep to save resources
- **Scaling**: Automatic scaling based on request volume