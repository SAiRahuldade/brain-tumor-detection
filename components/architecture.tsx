"use client"

import { ArrowRight } from "lucide-react"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { cn } from "@/lib/utils"

const layers = [
  { name: "Input", detail: "224x224x3", color: "bg-blue-500" },
  { name: "Conv2D", detail: "64 filters", color: "bg-red-500" },
  { name: "MaxPool", detail: "2x2", color: "bg-amber-500" },
  { name: "Conv2D", detail: "128 filters", color: "bg-red-500" },
  { name: "MaxPool", detail: "2x2", color: "bg-amber-500" },
  { name: "Conv2D", detail: "256 filters", color: "bg-red-500" },
  { name: "Flatten", detail: "", color: "bg-purple-500" },
  { name: "Dense", detail: "512 units", color: "bg-emerald-500" },
  { name: "Output", detail: "4 classes", color: "bg-cyan-500" },
]

const legendItems = [
  { name: "Input/Output", color: "bg-blue-500" },
  { name: "Convolutional", color: "bg-red-500" },
  { name: "Pooling", color: "bg-amber-500" },
  { name: "Dense", color: "bg-emerald-500" },
]

const trainingConfig = [
  { label: "Optimizer", value: "Adam" },
  { label: "Loss Function", value: "Categorical Crossentropy" },
  { label: "Batch Size", value: "32" },
  { label: "Learning Rate", value: "0.001" },
  { label: "Epochs", value: "30" },
]

const dataAugmentation = [
  "Rotation (up to 20 degrees)",
  "Horizontal flip",
  "Zoom (up to 20%)",
  "Width/Height shift",
  "Brightness adjustment",
]

export function Architecture() {
  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight text-foreground">CNN Architecture</h2>
        <p className="text-muted-foreground">
          Visual representation of the deep learning architecture used for tumor classification
        </p>
      </div>

      {/* Architecture Diagram */}
      <Card>
        <CardHeader>
          <CardTitle>Network Architecture</CardTitle>
          <CardDescription>
            Sequential CNN model for brain tumor classification
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Layer Visualization */}
          <div className="overflow-x-auto pb-4">
            <div className="flex items-center gap-2 min-w-max p-4">
              {layers.map((layer, index) => (
                <div key={index} className="flex items-center gap-2">
                  <div className="flex flex-col items-center">
                    <div
                      className={cn(
                        "w-20 h-24 rounded-lg flex flex-col items-center justify-center text-white shadow-lg",
                        layer.color
                      )}
                    >
                      <span className="text-xs font-semibold text-center px-1">{layer.name}</span>
                      {layer.detail && (
                        <span className="text-[10px] opacity-80 text-center px-1 mt-1">{layer.detail}</span>
                      )}
                    </div>
                  </div>
                  {index < layers.length - 1 && (
                    <ArrowRight className="h-5 w-5 text-muted-foreground shrink-0" />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-4 mt-6 pt-6 border-t border-border">
            <span className="text-sm font-medium text-foreground">Layer Types:</span>
            {legendItems.map((item) => (
              <div key={item.name} className="flex items-center gap-2">
                <div className={cn("h-3 w-3 rounded", item.color)} />
                <span className="text-sm text-muted-foreground">{item.name}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Details Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Architecture Details */}
        <Card>
          <CardHeader>
            <CardTitle>Architecture Details</CardTitle>
            <CardDescription>
              Key components of the CNN model
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <DetailSection 
              title="Input Layer"
              description="Accepts 224x224 RGB images (3 channels). Images are normalized to [0, 1] range before processing."
            />
            <DetailSection 
              title="Convolutional Layers"
              description="Extract hierarchical features using 3x3 filters with ReLU activation. Filter depths increase (64, 128, 256) to capture complex patterns."
            />
            <DetailSection 
              title="Pooling Layers"
              description="Max-pooling with 2x2 windows reduces spatial dimensions while retaining important features and providing translation invariance."
            />
            <DetailSection 
              title="Dense Layers"
              description="Fully connected layers with 512 neurons perform final classification. Dropout (0.5) prevents overfitting."
            />
            <DetailSection 
              title="Output Layer"
              description="Softmax activation for 4-class probability distribution: Glioma, Meningioma, No Tumor, Pituitary."
            />
          </CardContent>
        </Card>

        {/* Training Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Training Configuration</CardTitle>
            <CardDescription>
              Hyperparameters and training settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              {trainingConfig.map((config) => (
                <div key={config.label} className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">{config.label}</span>
                  <span className="text-sm font-medium text-foreground bg-muted px-3 py-1 rounded">
                    {config.value}
                  </span>
                </div>
              ))}
            </div>

            <div className="pt-4 border-t border-border">
              <h4 className="text-sm font-medium text-foreground mb-3">Data Augmentation</h4>
              <ul className="space-y-2">
                {dataAugmentation.map((item) => (
                  <li key={item} className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Technical Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Technical Summary</CardTitle>
          <CardDescription>
            Overview of the model implementation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-3">
            <SummaryCard
              title="Total Parameters"
              value="~2.5M"
              description="Trainable weights in the model"
            />
            <SummaryCard
              title="Training Time"
              value="~45 min"
              description="On NVIDIA GPU (RTX 3080)"
            />
            <SummaryCard
              title="Inference Time"
              value="<100ms"
              description="Per image prediction"
            />
          </div>
        </CardContent>
      </Card>

      {/* Grad-CAM Explanation */}
      <Card>
        <CardHeader>
          <CardTitle>Grad-CAM Visualization</CardTitle>
          <CardDescription>
            How the model explains its predictions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Gradient-weighted Class Activation Mapping (Grad-CAM) is used to produce visual explanations 
            for predictions made by the CNN. It highlights the regions of the input image that are most 
            important for the model's classification decision.
          </p>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-lg border border-border p-4">
              <h4 className="font-medium text-foreground mb-2">Step 1: Forward Pass</h4>
              <p className="text-sm text-muted-foreground">
                The image passes through the network, and we capture the feature maps from the last convolutional layer.
              </p>
            </div>
            <div className="rounded-lg border border-border p-4">
              <h4 className="font-medium text-foreground mb-2">Step 2: Gradient Computation</h4>
              <p className="text-sm text-muted-foreground">
                Gradients of the predicted class score are computed with respect to the feature maps.
              </p>
            </div>
            <div className="rounded-lg border border-border p-4">
              <h4 className="font-medium text-foreground mb-2">Step 3: Heatmap Generation</h4>
              <p className="text-sm text-muted-foreground">
                Gradients are pooled and used to weight the feature maps, creating a class-specific heatmap.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function DetailSection({ title, description }: { title: string; description: string }) {
  return (
    <div className="space-y-1">
      <h4 className="text-sm font-medium text-foreground">{title}</h4>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  )
}

function SummaryCard({ title, value, description }: { title: string; value: string; description: string }) {
  return (
    <div className="rounded-lg border border-border p-4 text-center">
      <p className="text-sm text-muted-foreground">{title}</p>
      <p className="text-2xl font-bold text-primary mt-1">{value}</p>
      <p className="text-xs text-muted-foreground mt-1">{description}</p>
    </div>
  )
}
