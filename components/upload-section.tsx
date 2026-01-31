"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, Image as ImageIcon, Loader2, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { AnalysisResult } from "@/app/page"
import Image from "next/image"

interface UploadSectionProps {
  onAnalysisComplete: (result: AnalysisResult) => void
  isAnalyzing: boolean
  setIsAnalyzing: (value: boolean) => void
}

const CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

function generateMockAnalysis(imageUrl: string): AnalysisResult {
  // Simulate different tumor types with weighted random selection
  const randomIndex = Math.floor(Math.random() * CLASS_NAMES.length)
  const tumorType = CLASS_NAMES[randomIndex]
  
  // Generate realistic probabilities
  const baseProbability = 0.7 + Math.random() * 0.25
  const remaining = 1 - baseProbability
  const otherProbs = CLASS_NAMES.filter(c => c !== tumorType).map(() => remaining / 3 + (Math.random() - 0.5) * 0.1)
  
  const probabilities: { [key: string]: number } = {}
  let probIndex = 0
  CLASS_NAMES.forEach(className => {
    if (className === tumorType) {
      probabilities[className] = baseProbability * 100
    } else {
      probabilities[className] = Math.max(0, otherProbs[probIndex++] * 100)
    }
  })
  
  const confidence = probabilities[tumorType]
  const affectedArea = tumorType === 'notumor' ? 0 : 5 + Math.random() * 15
  
  let severity = "None"
  if (tumorType !== 'notumor') {
    if (affectedArea < 8) severity = "Low"
    else if (affectedArea < 15) severity = "Moderate"
    else severity = "High"
  }
  
  return {
    tumorType,
    confidence,
    probabilities,
    severity,
    affectedArea,
    originalImage: imageUrl,
  }
}

export function UploadSection({ onAnalysisComplete, isAnalyzing, setIsAnalyzing }: UploadSectionProps) {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>("")

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setFileName(file.name)
      const reader = new FileReader()
      reader.onload = () => {
        setUploadedImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png']
    },
    multiple: false
  })

  const handleAnalyze = async () => {
    if (!uploadedImage) return
    
    setIsAnalyzing(true)
    
    // Simulate analysis delay
    await new Promise(resolve => setTimeout(resolve, 2500))
    
    const result = generateMockAnalysis(uploadedImage)
    onAnalysisComplete(result)
    setIsAnalyzing(false)
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight text-foreground">Brain Tumor Analysis</h2>
        <p className="text-muted-foreground">
          AI-Powered Medical Imaging Analysis for Early Detection
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Upload Area */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Upload MRI Scan</CardTitle>
            <CardDescription>
              Upload a brain MRI scan in JPG or PNG format for analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              {...getRootProps()}
              className={cn(
                "relative flex min-h-[280px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors",
                isDragActive 
                  ? "border-primary bg-primary/5" 
                  : "border-border hover:border-primary/50 hover:bg-muted/50"
              )}
            >
              <input {...getInputProps()} />
              {uploadedImage ? (
                <div className="relative w-full h-full min-h-[280px] p-4">
                  <Image
                    src={uploadedImage}
                    alt="Uploaded MRI scan"
                    fill
                    className="object-contain rounded-lg"
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-4 p-8 text-center">
                  <div className="rounded-full bg-primary/10 p-4">
                    <Upload className="h-8 w-8 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium text-foreground">
                      {isDragActive ? "Drop the file here" : "Drag & drop your MRI scan"}
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      or click to browse files
                    </p>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Supports: JPG, JPEG, PNG
                  </p>
                </div>
              )}
            </div>
            
            {uploadedImage && (
              <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <ImageIcon className="h-4 w-4" />
                  <span className="truncate max-w-[200px]">{fileName}</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    setUploadedImage(null)
                    setFileName("")
                  }}
                >
                  Remove
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Instructions Card */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">How It Works</CardTitle>
            <CardDescription>
              Follow these steps to analyze your MRI scan
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <Step 
              number={1} 
              title="Upload Your Scan"
              description="Upload a brain MRI image in JPG or PNG format. The image should be clear and properly oriented."
            />
            <Step 
              number={2} 
              title="AI Analysis"
              description="Our deep learning model analyzes the scan using Convolutional Neural Networks trained on thousands of MRI images."
            />
            <Step 
              number={3} 
              title="Review Results"
              description="View the classification results, confidence scores, and visual heatmap showing areas of interest."
            />
            <Step 
              number={4} 
              title="Download Report"
              description="Download a detailed analysis report for your records or to share with healthcare professionals."
            />
          </CardContent>
        </Card>
      </div>

      {/* Analyze Button */}
      <Button
        size="lg"
        className="w-full max-w-md mx-auto flex gap-2"
        disabled={!uploadedImage || isAnalyzing}
        onClick={handleAnalyze}
      >
        {isAnalyzing ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            Analyzing Scan...
          </>
        ) : (
          <>
            <Sparkles className="h-5 w-5" />
            Analyze Scan
          </>
        )}
      </Button>
    </div>
  )
}

function Step({ number, title, description }: { number: number; title: string; description: string }) {
  return (
    <div className="flex gap-4">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-semibold text-primary-foreground">
        {number}
      </div>
      <div>
        <h4 className="font-medium text-foreground">{title}</h4>
        <p className="text-sm text-muted-foreground mt-1">{description}</p>
      </div>
    </div>
  )
}
