"use client"

import {
  Line,
  LineChart,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
  Pie,
  PieChart,
} from "recharts"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

// Training history data
const trainingHistory = [
  { epoch: 1, accuracy: 0.66, val_accuracy: 0.23, loss: 1.11, val_loss: 7.56 },
  { epoch: 5, accuracy: 0.86, val_accuracy: 0.79, loss: 0.40, val_loss: 0.56 },
  { epoch: 10, accuracy: 0.90, val_accuracy: 0.76, loss: 0.27, val_loss: 1.04 },
  { epoch: 15, accuracy: 0.76, val_accuracy: 0.76, loss: 0.62, val_loss: 1.13 },
  { epoch: 20, accuracy: 0.90, val_accuracy: 0.85, loss: 0.27, val_loss: 0.44 },
  { epoch: 25, accuracy: 0.94, val_accuracy: 0.84, loss: 0.17, val_loss: 0.48 },
  { epoch: 30, accuracy: 0.95, val_accuracy: 0.95, loss: 0.13, val_loss: 0.15 },
]

// Performance metrics
const performanceMetrics = [
  { name: "Accuracy", value: 95.2, color: "#8b5cf6" },
  { name: "Precision", value: 94.8, color: "#06b6d4" },
  { name: "Recall", value: 94.5, color: "#f59e0b" },
  { name: "F1-Score", value: 94.6, color: "#ef4444" },
]

// Confusion matrix data
const confusionMatrix = [
  { true: "Glioma", glioma: 245, meningioma: 12, notumor: 8, pituitary: 5 },
  { true: "Meningioma", glioma: 10, meningioma: 258, notumor: 6, pituitary: 3 },
  { true: "No Tumor", glioma: 5, meningioma: 8, notumor: 390, pituitary: 2 },
  { true: "Pituitary", glioma: 7, meningioma: 5, notumor: 3, pituitary: 285 },
]

// Dataset distribution
const datasetDistribution = [
  { name: "Glioma", value: 1321, color: "#ef4444" },
  { name: "Meningioma", value: 1339, color: "#f59e0b" },
  { name: "No Tumor", value: 1595, color: "#22c55e" },
  { name: "Pituitary", value: 1457, color: "#3b82f6" },
]

export function ModelPerformance() {
  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight text-foreground">Model Performance</h2>
        <p className="text-muted-foreground">
          Comprehensive visualization of the model's training and evaluation metrics
        </p>
      </div>

      {/* Training History */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Training Accuracy</CardTitle>
            <CardDescription>
              Accuracy progression over 30 training epochs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                accuracy: {
                  label: "Train Accuracy",
                  color: "#8b5cf6",
                },
                val_accuracy: {
                  label: "Validation Accuracy",
                  color: "#f59e0b",
                },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey="epoch" 
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 1]}
                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                    name="Train Accuracy"
                  />
                  <Line
                    type="monotone"
                    dataKey="val_accuracy"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={false}
                    name="Validation Accuracy"
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Training Loss</CardTitle>
            <CardDescription>
              Loss reduction during model training
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                loss: {
                  label: "Train Loss",
                  color: "#8b5cf6",
                },
                val_loss: {
                  label: "Validation Loss",
                  color: "#f59e0b",
                },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey="epoch" 
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                    name="Train Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={false}
                    name="Validation Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {/* Performance Metrics and Confusion Matrix */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
            <CardDescription>
              Overall model evaluation scores
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                value: {
                  label: "Score",
                  color: "#8b5cf6",
                },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={performanceMetrics} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                  <XAxis 
                    type="number" 
                    domain={[0, 100]}
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) => `${value}%`}
                  />
                  <YAxis 
                    type="category" 
                    dataKey="name"
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                    width={80}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {performanceMetrics.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Confusion Matrix</CardTitle>
            <CardDescription>
              Classification performance breakdown
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    <th className="p-2 text-left text-muted-foreground font-medium">True / Pred</th>
                    <th className="p-2 text-center text-muted-foreground font-medium">Glioma</th>
                    <th className="p-2 text-center text-muted-foreground font-medium">Meningioma</th>
                    <th className="p-2 text-center text-muted-foreground font-medium">No Tumor</th>
                    <th className="p-2 text-center text-muted-foreground font-medium">Pituitary</th>
                  </tr>
                </thead>
                <tbody>
                  {confusionMatrix.map((row, i) => (
                    <tr key={row.true}>
                      <td className="p-2 font-medium text-foreground">{row.true}</td>
                      <td className={`p-2 text-center rounded ${i === 0 ? 'bg-primary/20 text-primary font-semibold' : 'text-muted-foreground'}`}>
                        {row.glioma}
                      </td>
                      <td className={`p-2 text-center rounded ${i === 1 ? 'bg-primary/20 text-primary font-semibold' : 'text-muted-foreground'}`}>
                        {row.meningioma}
                      </td>
                      <td className={`p-2 text-center rounded ${i === 2 ? 'bg-primary/20 text-primary font-semibold' : 'text-muted-foreground'}`}>
                        {row.notumor}
                      </td>
                      <td className={`p-2 text-center rounded ${i === 3 ? 'bg-primary/20 text-primary font-semibold' : 'text-muted-foreground'}`}>
                        {row.pituitary}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Dataset Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Distribution</CardTitle>
          <CardDescription>
            Distribution of 5,712 brain MRI images across four categories
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 lg:grid-cols-2">
            <ChartContainer
              config={{
                glioma: { label: "Glioma", color: "#ef4444" },
                meningioma: { label: "Meningioma", color: "#f59e0b" },
                notumor: { label: "No Tumor", color: "#22c55e" },
                pituitary: { label: "Pituitary", color: "#3b82f6" },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={datasetDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey="name" 
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {datasetDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer
              config={{
                value: { label: "Count", color: "#8b5cf6" },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={datasetDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {datasetDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <ChartTooltip content={<ChartTooltipContent />} />
                </PieChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </CardContent>
      </Card>

      <div className="rounded-lg border border-border bg-card p-4">
        <p className="text-sm text-muted-foreground">
          <strong className="text-foreground">Note:</strong> The model was trained on a balanced dataset of 5,712 brain MRI images across four categories. 
          The validation accuracy stabilizes around 95%, indicating good generalization without significant overfitting.
        </p>
      </div>
    </div>
  )
}
