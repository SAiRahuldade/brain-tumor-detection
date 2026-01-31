"use client"

import { useState } from "react"
import { Header } from "@/components/header"
import { Sidebar } from "@/components/sidebar"
import { UploadSection } from "@/components/upload-section"
import { AnalysisResults } from "@/components/analysis-results"
import { ModelPerformance } from "@/components/model-performance"
import { Architecture } from "@/components/architecture"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export interface AnalysisResult {
  tumorType: string
  confidence: number
  probabilities: { [key: string]: number }
  severity: string
  affectedArea: number
  originalImage: string
  heatmapImage?: string
  overlayImage?: string
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("analysis")
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  return (
    <div className="min-h-screen bg-background">
      <div className="flex">
        <Sidebar />
        <main className="flex-1 ml-0 lg:ml-72">
          <Header />
          <div className="p-6 lg:p-8">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full max-w-md grid-cols-3 mb-8">
                <TabsTrigger value="analysis">Analysis</TabsTrigger>
                <TabsTrigger value="performance">Performance</TabsTrigger>
                <TabsTrigger value="architecture">Architecture</TabsTrigger>
              </TabsList>
              
              <TabsContent value="analysis" className="space-y-8">
                <UploadSection 
                  onAnalysisComplete={setAnalysisResult}
                  isAnalyzing={isAnalyzing}
                  setIsAnalyzing={setIsAnalyzing}
                />
                {analysisResult && (
                  <AnalysisResults result={analysisResult} />
                )}
              </TabsContent>
              
              <TabsContent value="performance">
                <ModelPerformance />
              </TabsContent>
              
              <TabsContent value="architecture">
                <Architecture />
              </TabsContent>
            </Tabs>
          </div>
        </main>
      </div>
    </div>
  )
}
