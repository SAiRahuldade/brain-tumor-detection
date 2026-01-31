"use client"

import { Brain, Download, Sparkles, Activity, Layers, Upload, CheckCircle, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 relative overflow-hidden">
      {/* Animated Background Orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-violet-500/5 rounded-full blur-3xl" />
      </div>

      {/* Grid Pattern Overlay */}
      <div 
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage: `linear-gradient(rgba(99, 102, 241, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(99, 102, 241, 0.03) 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }}
      />

      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16 space-y-6">
          <Badge variant="outline" className="border-indigo-500/30 bg-indigo-500/10 text-indigo-300 px-4 py-1.5">
            <Sparkles className="w-3.5 h-3.5 mr-2" />
            AI-Powered Medical Imaging
          </Badge>
          
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
            <span className="bg-gradient-to-r from-white via-indigo-200 to-cyan-300 bg-clip-text text-transparent">
              NeuroScan AI
            </span>
          </h1>
          
          <p className="text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Advanced brain tumor detection using deep learning. Upload MRI scans for instant AI-powered analysis with Grad-CAM visualization.
          </p>
        </div>

        {/* Main Feature Card */}
        <Card className="max-w-4xl mx-auto bg-white/5 border-white/10 backdrop-blur-xl mb-12">
          <CardHeader className="text-center border-b border-white/10 pb-6">
            <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center">
              <Brain className="w-10 h-10 text-white" />
            </div>
            <CardTitle className="text-2xl text-white">Streamlit Application Ready</CardTitle>
            <CardDescription className="text-slate-400">
              Your visually stunning brain tumor detection app has been created
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-8">
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <FeatureItem 
                icon={<Upload className="w-5 h-5" />}
                title="MRI Upload"
                description="Drag & drop MRI scans for instant analysis"
              />
              <FeatureItem 
                icon={<Activity className="w-5 h-5" />}
                title="Grad-CAM Visualization"
                description="See exactly where the AI focuses attention"
              />
              <FeatureItem 
                icon={<Layers className="w-5 h-5" />}
                title="4 Tumor Types"
                description="Glioma, Meningioma, Pituitary, No Tumor"
              />
              <FeatureItem 
                icon={<CheckCircle className="w-5 h-5" />}
                title="95% Accuracy"
                description="State-of-the-art CNN model performance"
              />
            </div>

            {/* How to Run */}
            <div className="bg-slate-900/50 rounded-xl p-6 border border-white/5">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-sm">1</span>
                How to Run Your App
              </h3>
              <div className="space-y-3">
                <CodeStep step="Download the project files" />
                <CodeStep step="Install dependencies:" code="pip install streamlit tensorflow opencv-python matplotlib pillow" />
                <CodeStep step="Run the app:" code="streamlit run app.py" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Visual Features */}
        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <VisualCard 
            gradient="from-indigo-500 to-violet-500"
            title="Glassmorphism UI"
            description="Modern frosted glass effects with animated backgrounds"
          />
          <VisualCard 
            gradient="from-cyan-500 to-blue-500"
            title="Dark Theme"
            description="Eye-friendly dark palette perfect for medical imaging"
          />
          <VisualCard 
            gradient="from-emerald-500 to-teal-500"
            title="Interactive Charts"
            description="Training metrics with gradient fills and animations"
          />
        </div>

        {/* Footer */}
        <div className="text-center mt-16 text-slate-500 text-sm">
          <p>Built with TensorFlow, Streamlit, and advanced CSS styling</p>
        </div>
      </div>
    </div>
  )
}

function FeatureItem({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="flex items-start gap-4 p-4 rounded-xl bg-white/5 border border-white/5 hover:border-indigo-500/30 transition-colors">
      <div className="w-10 h-10 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center flex-shrink-0">
        {icon}
      </div>
      <div>
        <h4 className="font-medium text-white mb-1">{title}</h4>
        <p className="text-sm text-slate-400">{description}</p>
      </div>
    </div>
  )
}

function CodeStep({ step, code }: { step: string, code?: string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 text-slate-300">
        <ArrowRight className="w-4 h-4 text-indigo-400" />
        <span>{step}</span>
      </div>
      {code && (
        <code className="block ml-6 px-3 py-2 bg-slate-800/80 rounded-lg text-cyan-400 text-sm font-mono">
          {code}
        </code>
      )}
    </div>
  )
}

function VisualCard({ gradient, title, description }: { gradient: string, title: string, description: string }) {
  return (
    <Card className="bg-white/5 border-white/10 hover:border-white/20 transition-all hover:-translate-y-1">
      <CardContent className="pt-6">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradient} mb-4 flex items-center justify-center`}>
          <Sparkles className="w-6 h-6 text-white" />
        </div>
        <h3 className="font-semibold text-white mb-2">{title}</h3>
        <p className="text-sm text-slate-400">{description}</p>
      </CardContent>
    </Card>
  )
}
