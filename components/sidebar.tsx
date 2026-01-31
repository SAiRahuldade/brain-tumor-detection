"use client"

import { Brain, AlertTriangle, Activity, BarChart3, Info } from "lucide-react"
import { cn } from "@/lib/utils"

interface SidebarProps {
  isMobile?: boolean
}

export function Sidebar({ isMobile }: SidebarProps) {
  return (
    <aside className={cn(
      "fixed inset-y-0 left-0 z-50 w-72 border-r border-border bg-card",
      !isMobile && "hidden lg:block"
    )}>
      <div className="flex h-full flex-col">
        {/* Logo */}
        <div className="flex h-16 items-center gap-3 border-b border-border px-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
            <Brain className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="font-semibold text-foreground">Brain Tumor</h1>
            <p className="text-xs text-muted-foreground">Analysis System</p>
          </div>
        </div>
        
        {/* About Section */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <div>
            <div className="flex items-center gap-2 text-sm font-medium text-foreground mb-3">
              <Info className="h-4 w-4" />
              About
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              This advanced AI-powered system analyzes brain MRI scans to detect and classify tumors using deep learning technology.
            </p>
          </div>
          
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Features</p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                High-accuracy tumor detection
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Detailed medical analysis
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Visual heatmap analysis
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Downloadable reports
              </li>
            </ul>
          </div>
          
          {/* Warning */}
          <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-4">
            <div className="flex items-center gap-2 text-amber-500 mb-2">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-sm font-medium">Important Notice</span>
            </div>
            <p className="text-xs text-amber-500/80 leading-relaxed">
              This tool is for educational purposes only. Always consult qualified medical professionals for diagnosis and treatment.
            </p>
          </div>
          
          {/* Classifications */}
          <div>
            <div className="flex items-center gap-2 text-sm font-medium text-foreground mb-3">
              <Activity className="h-4 w-4" />
              Supported Classifications
            </div>
            <div className="space-y-2">
              <ClassificationBadge color="bg-red-500" label="Glioma" />
              <ClassificationBadge color="bg-orange-500" label="Meningioma" />
              <ClassificationBadge color="bg-yellow-500" label="Pituitary Adenoma" />
              <ClassificationBadge color="bg-emerald-500" label="No Tumor" />
            </div>
          </div>
          
          {/* Model Performance */}
          <div>
            <div className="flex items-center gap-2 text-sm font-medium text-foreground mb-3">
              <BarChart3 className="h-4 w-4" />
              Model Performance
            </div>
            <div className="space-y-2">
              <MetricRow label="Accuracy" value="95.2%" />
              <MetricRow label="Precision" value="94.8%" />
              <MetricRow label="F1-Score" value="94.6%" />
              <MetricRow label="Training Epochs" value="30" />
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}

function ClassificationBadge({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className={cn("h-2 w-2 rounded-full", color)} />
      <span className="text-sm text-muted-foreground">{label}</span>
    </div>
  )
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground">{value}</span>
    </div>
  )
}
