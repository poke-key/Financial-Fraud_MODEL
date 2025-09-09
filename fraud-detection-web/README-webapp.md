# Financial Fraud Detection Web App

A Next.js web application that replicates the functionality of the original Streamlit fraud detection app, built with modern web technologies and shadcn/ui components.

## Features

- **File Upload**: CSV file upload with drag-and-drop support
- **Data Preview**: View uploaded data in a clean table format
- **Fraud Prediction**: Machine learning-based fraud detection
- **Visualization**: Interactive charts showing prediction results
- **Export Results**: Download predictions as CSV
- **Responsive Design**: Modern UI built with shadcn/ui and Tailwind CSS

## Architecture

- **Frontend**: Next.js 15 + React + TypeScript + shadcn/ui
- **Backend**: Python Flask API for ML model predictions
- **Fallback**: JavaScript-based prediction when Python API is unavailable
- **UI Components**: shadcn/ui with Tailwind CSS
- **Charts**: Recharts for data visualization

## Setup

### Prerequisites

- Node.js 18+ 
- Python 3.8+
- pip

### Installation

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Install Python dependencies:**
   ```bash
   npm run install-python
   # or manually:
   pip install -r requirements.txt
   ```

3. **Copy the ML model:**
   Make sure the `fraud_xgb_tuned.pkl` model file is available in `../models/` directory relative to the project root.

### Running the Application

#### Option 1: Run both servers simultaneously (Recommended)
```bash
npm run dev-full
```

This will start:
- Next.js development server on http://localhost:3000
- Python API server on http://localhost:5000

#### Option 2: Run servers separately

**Terminal 1 - Next.js Frontend:**
```bash
npm run dev
```

**Terminal 2 - Python API:**
```bash
npm run python-api
# or manually:
python python_api.py
```

## Usage

1. **Open the application** in your browser at http://localhost:3000

2. **Upload a CSV file** containing transaction data with the required columns:
   - TransactionAmt
   - ProductCD, card4, card6, DeviceType, id_30, id_31, P_emaildomain, R_emaildomain (categorical features)
   - Other transaction features as needed

3. **Click "Predict Fraud"** to run the model on your data

4. **View Results:**
   - Data preview table
   - Predictions table with fraud/not fraud classifications
   - Bar chart visualization of results

5. **Download Results** as CSV using the download button

## API Endpoints

### POST /api/predict
Processes uploaded data and returns fraud predictions.

**Request Body:**
```json
{
  "data": [
    {
      "TransactionAmt": 100.50,
      "ProductCD": "C",
      "card4": "visa",
      // ... other features
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
  ]
}
```

## Components

### UI Components (shadcn/ui)
- `Card` - Container components
- `Button` - Action buttons
- `Input` - File upload
- `Table` - Data display
- `Alert` - Error messages
- `Progress` - Loading states

### Main Components
- `FraudDetectionApp` - Main application component
- File upload handler
- Prediction API integration
- Chart visualization
- CSV export functionality

## Differences from Streamlit App

While maintaining the same core functionality, the web app offers:

- **Better Performance**: Client-side processing and caching
- **Modern UI**: Clean, professional interface with shadcn/ui
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Better Error Handling**: Graceful fallbacks and user feedback
- **API Architecture**: Separation of concerns with REST API
- **Export Features**: Enhanced CSV download functionality

## Development

### Project Structure
```
fraud-detection-web/
├── src/
│   ├── app/
│   │   ├── api/predict/route.ts    # Prediction API endpoint
│   │   ├── globals.css             # Global styles
│   │   └── page.tsx                # Main app component
│   ├── components/ui/              # shadcn/ui components
│   └── lib/utils.ts                # Utility functions
├── python_api.py                   # Python Flask API server  
├── requirements.txt                # Python dependencies
└── package.json                    # Node.js dependencies
```

### Adding New Features

1. **UI Components**: Add new shadcn/ui components with `npx shadcn@latest add [component]`
2. **API Routes**: Create new route handlers in `src/app/api/`
3. **Python Integration**: Extend `python_api.py` for additional ML functionality

## Troubleshooting

### Python API Connection Issues
- Ensure Python API is running on port 5000
- Check firewall settings
- The app will fallback to JavaScript-based predictions if Python API is unavailable

### Model Loading Issues
- Verify `fraud_xgb_tuned.pkl` exists in `../models/` directory
- Check Python dependencies are installed correctly
- Review console logs for specific error messages

### CSV Upload Issues
- Ensure CSV has proper headers
- Check file size limits
- Verify required columns are present

## License

This project maintains the same license as the original fraud detection repository.