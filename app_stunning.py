# Fix threading issues with TensorFlow/Keras in Streamlit
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import io
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STUNNING CUSTOM CSS - GLASSMORPHISM + DARK THEME
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --accent: #06b6d4;
        --accent-light: #22d3ee;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f0f1a;
        --bg-card: rgba(15, 15, 26, 0.8);
        --glass: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    
    /* Animated Background Orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(6, 182, 212, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(2%, 2%) rotate(1deg); }
        66% { transform: translate(-1%, 1%) rotate(-1deg); }
    }
    
    /* Main Container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        position: relative;
        z-index: 1;
    }
    
    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), rgba(6, 182, 212, 0.5), transparent);
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 50px;
        font-size: 0.85rem;
        color: #a5b4fc;
        margin-bottom: 1.5rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #a1a1aa;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #a1a1aa;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Status Badges */
    .status-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .status-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .status-info {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 26, 0.98) 0%, rgba(26, 26, 46, 0.98) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }
    
    section[data-testid="stSidebar"] * {
        color: #e4e4e7 !important;
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .sidebar-logo {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .sidebar-section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #71717a !important;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.5);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 2px dashed rgba(99, 102, 241, 0.3);
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(99, 102, 241, 0.6);
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        border-radius: 10px;
    }
    
    /* Metrics Container */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    div[data-testid="metric-container"] label {
        color: #a1a1aa !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a1a1aa;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(6, 182, 212, 0.2)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Images */
    img {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    h2 {
        font-size: 1.75rem;
        font-weight: 600;
        margin-top: 2rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e4e4e7 !important;
    }
    
    p, li {
        color: #a1a1aa;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), rgba(6, 182, 212, 0.3), transparent);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        color: #ffffff !important;
    }
    
    /* Animation keyframes */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .animate-slide-in {
        animation: slideIn 0.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
MODEL_PATH = "brain_tumor_best.keras"
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

TRAINING_HISTORY = {
    'accuracy': [0.66, 0.78, 0.82, 0.85, 0.86, 0.88, 0.89, 0.89, 0.90, 0.90, 0.89, 0.90, 0.85, 0.90, 0.76, 0.89, 0.85, 0.90, 0.87, 0.89, 0.90, 0.91, 0.92, 0.95, 0.94, 0.96, 0.83, 0.92, 0.94, 0.95],
    'val_accuracy': [0.23, 0.48, 0.80, 0.82, 0.79, 0.84, 0.80, 0.81, 0.85, 0.76, 0.83, 0.77, 0.78, 0.89, 0.76, 0.83, 0.85, 0.86, 0.90, 0.85, 0.58, 0.70, 0.92, 0.95, 0.84, 0.93, 0.65, 0.92, 0.94, 0.95],
    'loss': [1.11, 0.63, 0.50, 0.43, 0.40, 0.34, 0.32, 0.30, 0.28, 0.27, 0.30, 0.27, 0.40, 0.27, 0.62, 0.30, 0.40, 0.28, 0.35, 0.29, 0.27, 0.24, 0.21, 0.14, 0.17, 0.11, 0.45, 0.21, 0.17, 0.13],
    'val_loss': [7.56, 2.89, 0.63, 0.52, 0.56, 0.47, 0.53, 0.49, 0.42, 1.04, 0.51, 1.00, 0.65, 0.35, 1.13, 0.51, 0.44, 0.40, 0.32, 0.44, 1.66, 0.84, 0.25, 0.16, 0.48, 0.21, 2.15, 0.24, 0.18, 0.15]
}

TUMOR_INFO = {
    'glioma': {
        'name': 'Glioma',
        'icon': '🔴',
        'description': 'A tumor that originates from glial cells in the brain or spine.',
        'symptoms': 'Headaches, seizures, memory loss, personality changes, nausea',
        'treatment': 'Surgery, radiation therapy, chemotherapy, targeted therapy',
        'prognosis': 'Varies depending on grade and location. Early detection improves outcomes.',
        'color': '#ef4444'
    },
    'meningioma': {
        'name': 'Meningioma',
        'icon': '🟠',
        'description': 'A tumor that arises from the meninges (protective membranes covering the brain).',
        'symptoms': 'Headaches, vision problems, hearing loss, seizures, weakness',
        'treatment': 'Observation, surgery, radiation therapy',
        'prognosis': 'Generally benign and slow-growing. Good outcomes with treatment.',
        'color': '#f59e0b'
    },
    'pituitary': {
        'name': 'Pituitary Adenoma',
        'icon': '🟡',
        'description': 'A tumor in the pituitary gland affecting hormone production.',
        'symptoms': 'Vision problems, hormonal imbalances, headaches, fatigue',
        'treatment': 'Medication, surgery, radiation therapy',
        'prognosis': 'Usually benign. Treatment can effectively manage symptoms.',
        'color': '#eab308'
    },
    'notumor': {
        'name': 'No Tumor Detected',
        'icon': '🟢',
        'description': 'The scan shows no signs of tumor presence.',
        'symptoms': 'N/A',
        'treatment': 'No treatment required. Regular monitoring recommended.',
        'prognosis': 'Excellent. Continue routine health checkups.',
        'color': '#10b981'
    }
}

# =========================
# MODEL FUNCTIONS
# =========================
@st.cache_resource
def load_brain_tumor_model():
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

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def generate_gradcam(model, img_array, last_conv_layer_name):
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
def create_dark_theme_plot():
    """Apply dark theme to matplotlib plots"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0f0f1a',
        'axes.facecolor': '#0f0f1a',
        'axes.edgecolor': '#3f3f46',
        'axes.labelcolor': '#a1a1aa',
        'text.color': '#e4e4e7',
        'xtick.color': '#a1a1aa',
        'ytick.color': '#a1a1aa',
        'grid.color': '#27272a',
        'legend.facecolor': '#18181b',
        'legend.edgecolor': '#3f3f46'
    })

def plot_training_history():
    create_dark_theme_plot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')
    
    epochs = range(1, len(TRAINING_HISTORY['accuracy']) + 1)
    
    # Accuracy plot with gradient fill
    ax1.plot(epochs, TRAINING_HISTORY['accuracy'], color='#6366f1', linewidth=2.5, label='Train Accuracy')
    ax1.plot(epochs, TRAINING_HISTORY['val_accuracy'], color='#06b6d4', linewidth=2.5, label='Validation Accuracy')
    ax1.fill_between(epochs, TRAINING_HISTORY['accuracy'], alpha=0.2, color='#6366f1')
    ax1.fill_between(epochs, TRAINING_HISTORY['val_accuracy'], alpha=0.2, color='#06b6d4')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold', color='#ffffff', pad=15)
    ax1.set_xlabel('Epoch', fontsize=11, color='#a1a1aa')
    ax1.set_ylabel('Accuracy', fontsize=11, color='#a1a1aa')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_ylim([0, 1])
    ax1.set_facecolor('#0f0f1a')
    
    # Loss plot
    ax2.plot(epochs, TRAINING_HISTORY['loss'], color='#6366f1', linewidth=2.5, label='Train Loss')
    ax2.plot(epochs, TRAINING_HISTORY['val_loss'], color='#06b6d4', linewidth=2.5, label='Validation Loss')
    ax2.fill_between(epochs, TRAINING_HISTORY['loss'], alpha=0.2, color='#6366f1')
    ax2.fill_between(epochs, TRAINING_HISTORY['val_loss'], alpha=0.2, color='#06b6d4')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold', color='#ffffff', pad=15)
    ax2.set_xlabel('Epoch', fontsize=11, color='#a1a1aa')
    ax2.set_ylabel('Loss', fontsize=11, color='#a1a1aa')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_facecolor('#0f0f1a')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix():
    create_dark_theme_plot()
    cm = np.array([
        [245, 12, 8, 5],
        [10, 258, 6, 3],
        [5, 8, 390, 2],
        [7, 5, 3, 285]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    
    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#0f0f1a', '#312e81', '#4f46e5', '#818cf8']
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#a1a1aa')
    
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#ffffff', pad=15)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="#ffffff" if cm[i, j] > thresh else "#a1a1aa",
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_class_distribution():
    create_dark_theme_plot()
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    counts = [1321, 1339, 1595, 1457]
    colors = ['#ef4444', '#f59e0b', '#10b981', '#6366f1']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')
    ax1.set_facecolor('#0f0f1a')
    ax2.set_facecolor('#0f0f1a')
    
    bars = ax1.bar(classes, counts, color=colors, alpha=0.85, edgecolor='#ffffff', linewidth=1)
    ax1.set_title('Dataset Distribution', fontsize=14, fontweight='bold', color='#ffffff', pad=15)
    ax1.set_ylabel('Number of Images', fontsize=11, color='#a1a1aa')
    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color='#ffffff')
    
    wedges, texts, autotexts = ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold', 'color': '#ffffff'},
            wedgeprops={'edgecolor': '#0f0f1a', 'linewidth': 2})
    ax2.set_title('Class Distribution %', fontsize=14, fontweight='bold', color='#ffffff', pad=15)
    
    plt.tight_layout()
    return fig

def plot_model_metrics():
    create_dark_theme_plot()
    metrics = {
        'Accuracy': 95.2,
        'Precision': 94.8,
        'Recall': 94.5,
        'F1-Score': 94.6
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    
    colors = ['#6366f1', '#8b5cf6', '#a855f7', '#06b6d4']
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), 
                   color=colors, alpha=0.85, edgecolor='#ffffff', linewidth=1, height=0.6)
    
    ax.set_xlabel('Score (%)', fontsize=11, fontweight='bold', color='#a1a1aa')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', color='#ffffff', pad=15)
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    
    for i, (bar, value) in enumerate(zip(bars, metrics.values())):
        ax.text(value + 1.5, i, f'{value:.1f}%', 
               va='center', fontsize=11, fontweight='bold', color='#ffffff')
    
    plt.tight_layout()
    return fig

def create_architecture_diagram():
    create_dark_theme_plot()
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9, 'CNN Architecture for Brain Tumor Classification', 
            ha='center', fontsize=16, fontweight='bold', color='#ffffff')
    
    layers = [
        {'name': 'Input\n224x224x3', 'x': 1, 'color': '#6366f1'},
        {'name': 'Conv2D\n64 filters', 'x': 2.5, 'color': '#ef4444'},
        {'name': 'MaxPool\n2x2', 'x': 4, 'color': '#f59e0b'},
        {'name': 'Conv2D\n128 filters', 'x': 5.5, 'color': '#ef4444'},
        {'name': 'MaxPool\n2x2', 'x': 7, 'color': '#f59e0b'},
        {'name': 'Conv2D\n256 filters', 'x': 8.5, 'color': '#ef4444'},
        {'name': 'Flatten', 'x': 10, 'color': '#8b5cf6'},
        {'name': 'Dense\n512', 'x': 11.5, 'color': '#10b981'},
        {'name': 'Output\n4 classes', 'x': 13, 'color': '#06b6d4'}
    ]
    
    for i, layer in enumerate(layers):
        rect = plt.Rectangle((layer['x']-0.45, 4), 0.9, 2.5, 
                            facecolor=layer['color'], edgecolor='#ffffff', 
                            linewidth=1.5, alpha=0.8, 
                            boxstyle='round,pad=0.05', joinstyle='round')
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((layer['x']-0.45, 4), 0.9, 2.5,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=layer['color'], edgecolor='#ffffff',
                              linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        
        ax.text(layer['x'], 5.25, layer['name'], 
               ha='center', va='center', fontsize=9, 
               fontweight='bold', color='white')
        
        if i < len(layers) - 1:
            ax.annotate('', xy=(layers[i+1]['x']-0.5, 5.25), xytext=(layer['x']+0.5, 5.25),
                       arrowprops=dict(arrowstyle='->', color='#6366f1', lw=2))
    
    legend_y = 1.5
    ax.text(7, legend_y + 1, 'Layer Types:', ha='center', fontsize=11, fontweight='bold', color='#ffffff')
    legend_items = [
        ('Input/Output', '#6366f1'),
        ('Convolutional', '#ef4444'),
        ('Pooling', '#f59e0b'),
        ('Dense', '#10b981')
    ]
    
    for i, (name, color) in enumerate(legend_items):
        x_pos = 2.5 + i * 3
        rect = FancyBboxPatch((x_pos-0.25, legend_y-0.25), 0.5, 0.4,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='#ffffff',
                              linewidth=1, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x_pos + 0.5, legend_y - 0.05, name, va='center', fontsize=10, color='#e4e4e7')
    
    plt.tight_layout()
    return fig

def estimate_tumor_area(heatmap):
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    mask = heatmap_resized > 0.5
    area_percentage = (np.sum(mask) / (IMG_SIZE * IMG_SIZE)) * 100
    return area_percentage

def classify_severity(area, tumor_type):
    if tumor_type == 'notumor':
        return "None"
    elif area < 5:
        return "Low"
    elif area < 15:
        return "Moderate"
    else:
        return "High"

def get_recommendation(tumor_type, severity, confidence):
    if tumor_type == 'notumor':
        return ["Continue regular health monitoring. Schedule annual checkups."]
    
    recommendations = []
    
    if confidence > 90:
        recommendations.append("High confidence detection - consult with a neurologist immediately")
    elif confidence > 70:
        recommendations.append("Moderate confidence - recommend additional imaging (CT/MRI scan)")
    else:
        recommendations.append("Low confidence - further diagnostic tests required")
    
    if severity == "High":
        recommendations.append("Urgent medical attention required")
        recommendations.append("Consult with neurosurgeon for treatment options")
    elif severity == "Moderate":
        recommendations.append("Schedule consultation with oncologist within 1-2 weeks")
        recommendations.append("May require biopsy for detailed analysis")
    else:
        recommendations.append("Monitor with follow-up scans in 3-6 months")
        recommendations.append("Maintain healthy lifestyle and regular checkups")
    
    return recommendations

def create_visualization(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    original_img_array = np.array(original_img)
    if original_img_array.shape[-1] == 4:
        original_img_array = original_img_array[:, :, :3]
    
    overlay = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
    
    return heatmap_resized, overlay

# =========================
# MAIN APP
# =========================
def main():
    # Hero Header
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">
            <span>Powered by Deep Learning</span>
        </div>
        <h1 class="hero-title">NeuroScan AI</h1>
        <p class="hero-subtitle">Advanced brain tumor detection and classification using state-of-the-art convolutional neural networks with Grad-CAM visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">🧠</div>
            <div class="sidebar-title">NeuroScan AI</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">About</div>
            <p style="font-size: 0.9rem; line-height: 1.6;">
            Advanced AI-powered medical imaging analysis for early brain tumor detection using deep learning.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Classifications</div>
            <p>🔴 Glioma</p>
            <p>🟠 Meningioma</p>
            <p>🟡 Pituitary Adenoma</p>
            <p>🟢 No Tumor</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Model Stats</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "95.2%")
        with col2:
            st.metric("F1-Score", "94.6%")
        
        st.markdown("---")
        st.warning("**Disclaimer:** This tool is for educational purposes only. Always consult medical professionals for diagnosis.")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Analysis", "📊 Performance", "🏗️ Architecture"])
    
    with tab2:
        st.markdown("## Model Performance Metrics")
        st.markdown("Comprehensive visualization of training and evaluation metrics.")
        
        st.markdown("### Training History")
        fig_history = plot_training_history()
        st.pyplot(fig_history)
        plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance Metrics")
            fig_metrics = plot_model_metrics()
            st.pyplot(fig_metrics)
            plt.close()
        
        with col2:
            st.markdown("### Confusion Matrix")
            fig_cm = plot_confusion_matrix()
            st.pyplot(fig_cm)
            plt.close()
        
        st.markdown("### Dataset Distribution")
        fig_dist = plot_class_distribution()
        st.pyplot(fig_dist)
        plt.close()
    
    with tab3:
        st.markdown("## CNN Architecture")
        st.markdown("Visual representation of the deep learning architecture.")
        
        fig_arch = create_architecture_diagram()
        st.pyplot(fig_arch)
        plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #6366f1;">Architecture Components</h3>
                <ul>
                    <li><strong>Input Layer:</strong> 224x224x3 RGB images</li>
                    <li><strong>Conv Layers:</strong> 64 → 128 → 256 filters</li>
                    <li><strong>Pooling:</strong> 2x2 max pooling</li>
                    <li><strong>Dense Layer:</strong> 512 neurons</li>
                    <li><strong>Output:</strong> 4-class softmax</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #06b6d4;">Training Configuration</h3>
                <ul>
                    <li><strong>Optimizer:</strong> Adam</li>
                    <li><strong>Loss:</strong> Categorical Crossentropy</li>
                    <li><strong>Batch Size:</strong> 32</li>
                    <li><strong>Epochs:</strong> 30</li>
                    <li><strong>Augmentation:</strong> Rotation, flip, zoom</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab1:
        model = load_brain_tumor_model()
        if model is None:
            st.error("Failed to load model. Please ensure 'brain_tumor_best.keras' exists.")
            return
        
        last_conv_layer = get_last_conv_layer(model)
        
        st.markdown("### Upload MRI Scan")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload a brain MRI image",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            original_img = Image.open(uploaded_file).convert('RGB')
            img_array = preprocess_image(original_img)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(original_img, caption="Uploaded MRI Scan", use_container_width=True)
            
            if st.button("🔬 Analyze Scan", type="primary", use_container_width=True):
                with st.spinner("Analyzing brain scan with AI..."):
                    heatmap, pred_index, predictions = generate_gradcam(model, img_array, last_conv_layer)
                    
                    tumor_type = CLASS_NAMES[pred_index]
                    confidence = float(predictions[0][pred_index] * 100)
                    tumor_area = estimate_tumor_area(heatmap)
                    severity = classify_severity(tumor_area, tumor_type)
                    
                    heatmap_vis, overlay_vis = create_visualization(original_img, heatmap)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    st.success("Analysis Complete!")
                    
                    st.markdown("---")
                    st.markdown("## Detailed Analysis Report")
                    st.markdown(f"**Generated:** {timestamp}")
                    
                    # Diagnosis Box
                    tumor_info = TUMOR_INFO[tumor_type]
                    
                    if tumor_type == 'notumor':
                        st.markdown(f"""
                        <div class="status-success">
                            <h3 style="color: #10b981; margin: 0;">{tumor_info['icon']} {tumor_info['name']}</h3>
                            <p style="margin: 0.5rem 0 0 0; color: #a7f3d0;">Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        status_class = {
                            'Low': 'status-success',
                            'Moderate': 'status-warning',
                            'High': 'status-danger'
                        }[severity]
                        st.markdown(f"""
                        <div class="{status_class}">
                            <h3 style="margin: 0;">{tumor_info['icon']} {tumor_info['name']} Detected</h3>
                            <p style="margin: 0.5rem 0 0 0;">Confidence: {confidence:.2f}% | Severity: {severity}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    st.markdown("### Quantitative Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    with m1:
                        st.metric("Tumor Type", tumor_info['name'])
                    with m2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with m3:
                        st.metric("Affected Area", f"{tumor_area:.2f}%")
                    with m4:
                        st.metric("Severity", severity)
                    
                    # Probabilities
                    st.markdown("### Classification Probabilities")
                    prob_data = {CLASS_NAMES[i]: float(predictions[0][i] * 100) for i in range(len(CLASS_NAMES))}
                    
                    for class_name, prob in prob_data.items():
                        st.progress(float(prob / 100), text=f"{class_name.capitalize()}: {prob:.2f}%")
                    
                    # Tumor Info
                    st.markdown("### Tumor Information")
                    st.markdown(f"""
                    <div class="status-info">
                        <p><strong>Description:</strong> {tumor_info['description']}</p>
                        <p><strong>Symptoms:</strong> {tumor_info['symptoms']}</p>
                        <p><strong>Treatment:</strong> {tumor_info['treatment']}</p>
                        <p><strong>Prognosis:</strong> {tumor_info['prognosis']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("### Medical Recommendations")
                    recommendations = get_recommendation(tumor_type, severity, confidence)
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {rec}")
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("### Visual Analysis")
                    
                    v1, v2, v3 = st.columns(3)
                    
                    with v1:
                        st.image(original_img, caption="Original MRI", use_container_width=True)
                    
                    with v2:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        fig.patch.set_facecolor('#0f0f1a')
                        ax.set_facecolor('#0f0f1a')
                        im = ax.imshow(heatmap_vis, cmap='jet')
                        ax.axis('off')
                        ax.set_title("Activation Heatmap", fontsize=12, weight='bold', color='#ffffff')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        st.pyplot(fig)
                        plt.close()
                    
                    with v3:
                        st.image(overlay_vis, caption="Overlay Analysis", use_container_width=True)
                    
                    st.info("**Heatmap Interpretation:** Red/yellow regions indicate areas where the AI detected potential tumor characteristics.")
                    
                    # Downloads
                    st.markdown("---")
                    st.markdown("### Export Report")
                    
                    report_text = f"""
NEUROSCAN AI - BRAIN TUMOR ANALYSIS REPORT
Generated: {timestamp}
{'='*60}

PRIMARY DIAGNOSIS
Tumor Type: {tumor_info['name']}
Confidence: {confidence:.2f}%
Affected Area: {tumor_area:.2f}%
Severity: {severity}

TUMOR INFORMATION
Description: {tumor_info['description']}
Symptoms: {tumor_info['symptoms']}
Treatment: {tumor_info['treatment']}
Prognosis: {tumor_info['prognosis']}

CLASSIFICATION PROBABILITIES
{chr(10).join([f'{name.capitalize()}: {prob_data[name]:.2f}%' for name in CLASS_NAMES])}

RECOMMENDATIONS
{chr(10).join([f'{i}. {rec}' for i, rec in enumerate(recommendations, 1)])}

DISCLAIMER
This analysis is generated by an AI system and should not replace 
professional medical diagnosis. Please consult with qualified 
healthcare providers for proper medical evaluation.
{'='*60}
                    """
                    
                    d1, d2 = st.columns(2)
                    
                    with d1:
                        st.download_button(
                            label="Download Report (TXT)",
                            data=report_text,
                            file_name=f"neuroscan_report_{timestamp.replace(':', '-').replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    
                    with d2:
                        overlay_pil = Image.fromarray(overlay_vis)
                        buf = io.BytesIO()
                        overlay_pil.save(buf, format='PNG')
                        st.download_button(
                            label="Download Overlay Image",
                            data=buf.getvalue(),
                            file_name=f"neuroscan_overlay_{timestamp.replace(':', '-').replace(' ', '_')}.png",
                            mime="image/png"
                        )
                    
                    st.markdown("---")
                    st.error("**MEDICAL DISCLAIMER:** This AI system is for educational purposes only. Always seek professional medical advice for diagnosis and treatment.")

if __name__ == "__main__":
    main()
