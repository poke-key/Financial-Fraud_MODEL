'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import Papa from 'papaparse'
import { Download, CreditCard, AlertCircle, CheckCircle } from 'lucide-react'

interface PredictionData {
  [key: string]: string | number
}

interface ChartData {
  name: string
  count: number
}

export default function FraudDetectionApp() {
  const [uploadedData, setUploadedData] = useState<PredictionData[]>([])
  const [predictions, setPredictions] = useState<PredictionData[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [, setDataLoaded] = useState(false)

  // Load sample data on component mount
  useEffect(() => {
    const loadSampleData = async () => {
      try {
        const response = await fetch('/sample_data_100.csv')
        const csvText = await response.text()
        
        Papa.parse(csvText, {
          complete: (results) => {
            if (results.errors.length > 0) {
              setError('Error loading sample data')
              return
            }
            
            const data = results.data as PredictionData[]
            setUploadedData(data.slice(0, -1)) // Remove last empty row if exists
            setDataLoaded(true)
          },
          header: true,
          skipEmptyLines: true
        })
      } catch {
        setError('Failed to load sample data')
      }
    }

    loadSampleData()
  }, [])

  const handlePredict = async () => {
    if (uploadedData.length === 0) {
      setError('No data available for prediction')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: uploadedData }),
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const result = await response.json()
      setPredictions(result.predictions)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during prediction')
    } finally {
      setIsLoading(false)
    }
  }

  const downloadResults = () => {
    if (predictions.length === 0) return

    const csv = Papa.unparse(predictions)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob)
      link.setAttribute('href', url)
      link.setAttribute('download', 'predictions.csv')
      link.style.visibility = 'hidden'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const getChartData = (): ChartData[] => {
    if (predictions.length === 0) return []

    const fraudCount = predictions.filter(p => p.prediction === 1).length
    const notFraudCount = predictions.filter(p => p.prediction === 0).length

    return [
      { name: 'Not Fraud (0)', count: notFraudCount },
      { name: 'Fraud (1)', count: fraudCount }
    ]
  }


  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center space-x-2">
            <CreditCard className="h-8 w-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-900">Financial Fraud Detection</h1>
          </div>
          <p className="text-gray-600">Sample transaction data loaded. Click below to run fraud detection analysis.</p>
        </div>

        {/* Action Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <span>Sample Data Loaded</span>
            </CardTitle>
            <CardDescription>
              100 sample transactions are ready for fraud detection analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-sm text-green-600">
                <CheckCircle className="h-4 w-4" />
                <span>Data loaded: sample_data_100.csv ({uploadedData.length} transactions)</span>
              </div>
              <Button onClick={handlePredict} disabled={isLoading || uploadedData.length === 0}>
                {isLoading ? 'Analyzing...' : 'Run Fraud Detection'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Uploaded Data Preview */}
        {uploadedData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <span>Uploaded Data Preview</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {Object.keys(uploadedData[0] || {}).slice(0, 10).map((key) => (
                        <TableHead key={key}>{key}</TableHead>
                      ))}
                      {Object.keys(uploadedData[0] || {}).length > 10 && (
                        <TableHead>...</TableHead>
                      )}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {uploadedData.slice(0, 5).map((row, index) => (
                      <TableRow key={index}>
                        {Object.values(row).slice(0, 10).map((value, cellIndex) => (
                          <TableCell key={cellIndex}>{String(value)}</TableCell>
                        ))}
                        {Object.values(row).length > 10 && (
                          <TableCell>...</TableCell>
                        )}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              {uploadedData.length > 5 && (
                <p className="text-sm text-gray-500 mt-2">
                  Showing first 5 rows of {uploadedData.length} total rows
                </p>
              )}
            </CardContent>
          </Card>
        )}

        {/* Predictions */}
        {predictions.length > 0 && (
          <>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center space-x-2">
                    <span>📊 Predictions</span>
                  </CardTitle>
                  <Button onClick={downloadResults} variant="outline" size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    Download Results as CSV
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {Object.keys(predictions[0] || {}).slice(0, 10).map((key) => (
                          <TableHead key={key}>{key}</TableHead>
                        ))}
                        {Object.keys(predictions[0] || {}).length > 10 && (
                          <TableHead>...</TableHead>
                        )}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {predictions.slice(0, 10).map((row, index) => (
                        <TableRow key={index}>
                          {Object.entries(row).slice(0, 10).map(([key, value], cellIndex) => (
                            <TableCell 
                              key={cellIndex}
                              className={key === 'prediction' ? 
                                (value === 1 ? 'text-red-600 font-semibold' : 'text-green-600 font-semibold') 
                                : ''
                              }
                            >
                              {String(value)}
                            </TableCell>
                          ))}
                          {Object.entries(row).length > 10 && (
                            <TableCell>...</TableCell>
                          )}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                {predictions.length > 10 && (
                  <p className="text-sm text-gray-500 mt-2">
                    Showing first 10 rows of {predictions.length} total predictions
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Visualization */}
            <Card>
              <CardHeader>
                <CardTitle>Fraud vs Not Fraud Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count">
                        {getChartData().map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.name === 'Fraud (1)' ? '#ef4444' : '#3b82f6'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
