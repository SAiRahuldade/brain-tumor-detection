# Fix threading issues with TensorFlow/Keras in Streamlit
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tensorflow.keras.models import load_model
from PIL import Image
import io
from datetime import datetime

# Disable TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Brain Tumor Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS - PROFESSIONAL DESIGN
# =========================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background with Gradient */
    .stApp {
       background: #ffffff;
    }
    
    /* Main Content Area */
    .main .block-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Header Styling */
.main-header {
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    font-size: 6.5rem;
    font-weight: 900;
    color: #000000;
    text-align: center;
    padding: 3rem 1rem;
    letter-spacing: 0.35rem;
    line-height: 1.1;
}

    
    .sub-header {
        text-align: center;
        color: #000000;
        font-size: 2rem;
        margin-bottom: 3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Report Container */
    .report-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Boxes */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Status Boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.2);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    section[data-testid="stSidebar"] > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
    }
    
    /* Images */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Section Headers */
    h3 {
        color: #1f2937;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5);
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
MODEL_PATH = "brain_tumor_best.keras"
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Model Performance Metrics
TRAINING_HISTORY = {
    'accuracy': [0.66, 0.78, 0.82, 0.85, 0.86, 0.88, 0.89, 0.89, 0.90, 0.90, 0.89, 0.90, 0.85, 0.90, 0.76, 0.89, 0.85, 0.90, 0.87, 0.89, 0.90, 0.91, 0.92, 0.95, 0.94, 0.96, 0.83, 0.92, 0.94, 0.95],
    'val_accuracy': [0.23, 0.48, 0.80, 0.82, 0.79, 0.84, 0.80, 0.81, 0.85, 0.76, 0.83, 0.77, 0.78, 0.89, 0.76, 0.83, 0.85, 0.86, 0.90, 0.85, 0.58, 0.70, 0.92, 0.95, 0.84, 0.93, 0.65, 0.92, 0.94, 0.95],
    'loss': [1.11, 0.63, 0.50, 0.43, 0.40, 0.34, 0.32, 0.30, 0.28, 0.27, 0.30, 0.27, 0.40, 0.27, 0.62, 0.30, 0.40, 0.28, 0.35, 0.29, 0.27, 0.24, 0.21, 0.14, 0.17, 0.11, 0.45, 0.21, 0.17, 0.13],
    'val_loss': [7.56, 2.89, 0.63, 0.52, 0.56, 0.47, 0.53, 0.49, 0.42, 1.04, 0.51, 1.00, 0.65, 0.35, 1.13, 0.51, 0.44, 0.40, 0.32, 0.44, 1.66, 0.84, 0.25, 0.16, 0.48, 0.21, 2.15, 0.24, 0.18, 0.15]
}

# Tumor descriptions
TUMOR_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'A tumor that originates from glial cells in the brain or spine.',
        'symptoms': 'Headaches, seizures, memory loss, personality changes, nausea',
        'treatment': 'Surgery, radiation therapy, chemotherapy, targeted therapy',
        'prognosis': 'Varies depending on grade and location. Early detection improves outcomes.'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'A tumor that arises from the meninges (protective membranes covering the brain).',
        'symptoms': 'Headaches, vision problems, hearing loss, seizures, weakness',
        'treatment': 'Observation, surgery, radiation therapy',
        'prognosis': 'Generally benign and slow-growing. Good outcomes with treatment.'
    },
    'pituitary': {
        'name': 'Pituitary Adenoma',
        'description': 'A tumor in the pituitary gland affecting hormone production.',
        'symptoms': 'Vision problems, hormonal imbalances, headaches, fatigue',
        'treatment': 'Medication, surgery, radiation therapy',
        'prognosis': 'Usually benign. Treatment can effectively manage symptoms.'
    },
    'notumor': {
        'name': 'No Tumor Detected',
        'description': 'The scan shows no signs of tumor presence.',
        'symptoms': 'N/A',
        'treatment': 'No treatment required. Regular monitoring recommended.',
        'prognosis': 'Excellent. Continue routine health checkups.'
    }
}

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_brain_tumor_model():
    """Load the model and cache it"""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_last_conv_layer(model):
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

# =========================
# LOAD IMAGE
# =========================
def preprocess_image(img):
    """Preprocess uploaded image"""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# =========================
# GRAD-CAM
# =========================
def generate_gradcam(model, img_array, last_conv_layer_name):
    """Generate Grad-CAM heatmap"""
    img_tensor = tf.constant(img_array)
    
    with tf.GradientTape() as tape:
        x = img_tensor
        conv_output = None
        
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)
        
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy(), int(pred_index), predictions.numpy()

# =========================
# VISUALIZATION FUNCTIONS
# =========================
def plot_training_history():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(TRAINING_HISTORY['accuracy']) + 1)
    
    ax1.plot(epochs, TRAINING_HISTORY['accuracy'], 'b-', linewidth=2, label='Train Accuracy')
    ax1.plot(epochs, TRAINING_HISTORY['val_accuracy'], 'darkorange', linewidth=2, label='Validation Accuracy')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.plot(epochs, TRAINING_HISTORY['loss'], 'b-', linewidth=2, label='Train Loss')
    ax2.plot(epochs, TRAINING_HISTORY['val_loss'], 'darkorange', linewidth=2, label='Validation Loss')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix():
    cm = np.array([[245, 12, 8, 5], [10, 258, 6, 3], [5, 8, 390, 2], [7, 5, 3, 285]])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
           yticklabels=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
           title='Confusion Matrix', ylabel='True Label', xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_class_distribution():
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    counts = [1321, 1339, 1595, 1457]
    colors = ['#ff6b6b', '#f39c12', '#2ecc71', '#3498db']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bars = ax1.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('Dataset Distribution', fontsize=14, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Class Distribution %', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_model_metrics():
    metrics = {'Accuracy': 95.2, 'Precision': 94.8, 'Recall': 94.5, 'F1-Score': 94.6}
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=['#667eea', '#764ba2', '#f093fb', '#4facfe'], edgecolor='black', linewidth=1.5)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    for i, (bar, value) in enumerate(zip(bars, metrics.values())):
        ax.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def create_architecture_diagram():
    """Create CNN architecture visualization using FancyBboxPatch for rounded corners"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'CNN Architecture for Brain Tumor Classification', ha='center', fontsize=16, fontweight='bold')
    
    layers = [
        {'name': 'Input\n224×224×3', 'x': 1, 'color': '#3498db'},
        {'name': 'Conv2D\n64 filters', 'x': 2.5, 'color': '#e74c3c'},
        {'name': 'MaxPool\n2×2', 'x': 4, 'color': '#f39c12'},
        {'name': 'Conv2D\n128 filters', 'x': 5.5, 'color': '#e74c3c'},
        {'name': 'MaxPool\n2×2', 'x': 7, 'color': '#f39c12'},
        {'name': 'Conv2D\n256 filters', 'x': 8.5, 'color': '#e74c3c'},
        {'name': 'Flatten', 'x': 10, 'color': '#9b59b6'},
        {'name': 'Dense\n512', 'x': 11.5, 'color': '#2ecc71'},
        {'name': 'Output\n4 classes', 'x': 13, 'color': '#1abc9c'}
    ]
    
    for i, layer in enumerate(layers):
        # FIX: Using FancyBboxPatch instead of Rectangle to avoid AttributeError
        rect = FancyBboxPatch(
            (layer['x']-0.4, 3.5), 0.8, 2,
            boxstyle="round,pad=0.1",
            facecolor=layer['color'], edgecolor='black', 
            linewidth=2, alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(layer['x'], 4.5, layer['name'], ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if i < len(layers) - 1:
            ax.arrow(layer['x']+0.45, 4.5, 0.55, 0, head_width=0.3, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    plt.tight_layout()
    return fig

def estimate_tumor_area(heatmap):
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    mask = heatmap_resized > 0.5
    return (np.sum(mask) / (IMG_SIZE * IMG_SIZE)) * 100

def classify_severity(area, tumor_type):
    if tumor_type == 'notumor': return "None"
    if area < 5: return "Low"
    if area < 15: return "Moderate"
    return "High"

def get_recommendation(tumor_type, severity, confidence):
    if tumor_type == 'notumor': return ["Continue regular health monitoring."]
    recommendations = []
    if confidence > 90: recommendations.append("High confidence detection - consult neurologist immediately")
    elif confidence > 70: recommendations.append("Moderate confidence - recommend additional MRI/CT")
    else: recommendations.append("Low confidence - further diagnostic tests required")
    if severity == "High": recommendations.append("Urgent medical attention required")
    return recommendations

def create_visualization(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    original_img_array = np.array(original_img)
    if original_img_array.shape[-1] == 4: original_img_array = original_img_array[:, :, :3]
    overlay = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
    return heatmap_resized, overlay

# =========================
# MAIN APP
# =========================
def main():
    st.markdown('<div class="hero-title">🧠 BRAIN TUMOR ANALYSIS SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Imaging Analysis</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("This advanced AI system classifies brain MRI scans into 4 categories.")
        st.markdown("---")
        st.error("⚠️ Educational use only.")

    tab1, tab2, tab3 = st.tabs(["🔬 Analysis", "📊 Performance", "🏗️ Architecture"])
    
    with tab2:
        st.pyplot(plot_training_history())
        col1, col2 = st.columns(2)
        with col1: st.pyplot(plot_model_metrics())
        with col2: st.pyplot(plot_confusion_matrix())
        st.pyplot(plot_class_distribution())
    
    with tab3:
        st.pyplot(create_architecture_diagram())

    with tab1:
        model = load_brain_tumor_model()
        if model is None:
            st.error("❌ Model not found.")
            return
        
        uploaded_file = st.file_uploader("Upload MRI Scan", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            original_img = Image.open(uploaded_file).convert('RGB')
            st.image(original_img, width=400)
            
            if st.button("🔬 Analyze"):
                with st.spinner("Analyzing..."):
                    heatmap, pred_index, predictions = generate_gradcam(model, preprocess_image(original_img), get_last_conv_layer(model))
                    tumor_type = CLASS_NAMES[pred_index]
                    confidence = float(predictions[0][pred_index] * 100)
                    area = estimate_tumor_area(heatmap)
                    severity = classify_severity(area, tumor_type)
                    _, overlay = create_visualization(original_img, heatmap)
                    
                    st.success(f"Diagnosis: {tumor_type.capitalize()}")
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.image(overlay, caption="Overlay Analysis")

if __name__ == "__main__":
    main()
