import { NextRequest, NextResponse } from 'next/server'

const PYTHON_API_URL = process.env.FINANCIAL_SERVICE_URL || 'http://localhost:5000'

export async function POST(request: NextRequest) {
  try {
    console.log('Using API URL:', PYTHON_API_URL)
    console.log('Environment variable FINANCIAL_SERVICE_URL:', process.env.FINANCIAL_SERVICE_URL)
    const { data } = await request.json()
    
    if (!data || !Array.isArray(data) || data.length === 0) {
      return NextResponse.json(
        { error: 'Invalid data provided' },
        { status: 400 }
      )
    }

    try {
      // Try to call the Python API first with timeout handling
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 second timeout
      
      console.log('Attempting to fetch from:', `${PYTHON_API_URL}/predict`)
      
      const response = await fetch(`${PYTHON_API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data }),
        signal: controller.signal
      })

      clearTimeout(timeoutId)
      
      console.log('Response status:', response.status)
      console.log('Response ok:', response.ok)

      if (response.ok) {
        const result = await response.json()
        console.log('Python API success, returning result')
        return NextResponse.json(result)
      } else {
        console.log('Response not ok, status text:', response.statusText)
        const errorText = await response.text()
        console.log('Error response body:', errorText)
      }
    } catch (pythonApiError) {
      console.error('Python API error details:', pythonApiError)
      console.warn('Python API unavailable, falling back to simple prediction:', pythonApiError)
    }

    // Fallback: Simple prediction logic when Python API is not available
    const predictions = data.map((row: Record<string, unknown>) => {
      // Simple heuristic based on transaction amount and other factors
      const transactionAmt = parseFloat(String(row.TransactionAmt)) || 0
      const cardType = String(row.card4) || 'unknown'
      
      // More sophisticated fallback logic
      let riskScore = 0
      
      // High transaction amounts are riskier
      if (transactionAmt > 500) riskScore += 0.3
      else if (transactionAmt > 200) riskScore += 0.2
      else if (transactionAmt > 100) riskScore += 0.1
      
      // Certain card types might be riskier (simulate)
      if (cardType === 'visa') riskScore += 0.1
      else if (cardType === 'mastercard') riskScore += 0.05
      
      // Add some randomness to simulate model uncertainty
      riskScore += Math.random() * 0.3
      
      const prediction = riskScore > 0.5 ? 1 : 0
      
      return {
        ...row,
        prediction
      }
    })

    return NextResponse.json({
      success: true,
      predictions,
      fallback: true,
      message: 'Using fallback prediction (Python API unavailable)'
    })

  } catch (error) {
    console.error('Prediction error:', error)
    return NextResponse.json(
      { error: 'Internal server error during prediction' },
      { status: 500 }
    )
  }
}