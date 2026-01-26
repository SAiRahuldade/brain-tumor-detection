import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
MODEL_PATH = "brain_tumor_best.keras"
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# =========================
# LOAD MODEL
# =========================
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# FIND LAST CONV LAYER
# =========================
LAST_CONV_LAYER = None
for layer in model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        LAST_CONV_LAYER = layer.name
        break

if LAST_CONV_LAYER is None:
    raise ValueError("No Conv2D layer found in model!")

print(f"✅ Using last conv layer: {LAST_CONV_LAYER}")

# =========================
# LOAD IMAGE
# =========================
def load_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32), img

# =========================
# GRAD-CAM (MANUAL APPROACH - NO MODEL CREATION)
# =========================
def generate_gradcam(img_array):
    """
    Generate Grad-CAM without creating a new Model object.
    We'll manually forward pass and compute gradients.
    """
    img_tensor = tf.constant(img_array)
    
    # Get the target conv layer
    last_conv_layer = model.get_layer(LAST_CONV_LAYER)
    
    # Create a tape to watch intermediate activations
    with tf.GradientTape() as tape:
        # Manually forward pass through layers until conv layer
        x = img_tensor
        conv_output = None
        
        for layer in model.layers:
            x = layer(x)
            if layer.name == LAST_CONV_LAYER:
                conv_output = x
                tape.watch(conv_output)
        
        # x now contains the final predictions
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Compute gradients of the predicted class w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_output)
    
    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradients
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy(), int(pred_index), predictions.numpy()

# =========================
# TUMOR AREA ESTIMATION
# =========================
def estimate_tumor_area(heatmap):
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    mask = heatmap_resized > 0.5
    area_percentage = (np.sum(mask) / (IMG_SIZE * IMG_SIZE)) * 100
    return area_percentage

# =========================
# SEVERITY CLASSIFICATION
# =========================
def classify_severity(area):
    if area < 5:
        return "Low"
    elif area < 15:
        return "Moderate"
    else:
        return "High"

# =========================
# FINAL REPORT
# =========================
def generate_report(img_path):
    print(f"\n🔍 Analyzing: {img_path}")
    
    # Load image
    img_array, original_img = load_image(img_path)
    
    # Generate Grad-CAM
    print("🧠 Generating Grad-CAM heatmap...")
    heatmap, pred_index, preds = generate_gradcam(img_array)
    
    # Calculate metrics
    tumor_area = estimate_tumor_area(heatmap)
    severity = classify_severity(tumor_area)
    confidence = preds[0][pred_index] * 100
    
    # Print all class probabilities
    print("\n📊 Class Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        bar = "█" * int(preds[0][i] * 20)
        print(f"   {class_name:12s}: {preds[0][i]*100:6.2f}% {bar}")
    
    # Print report
    print("\n" + "="*52)
    print("         MEDICAL AI ANALYSIS REPORT")
    print("="*52)
    print(f"Tumor Type           : {CLASS_NAMES[pred_index].upper()}")
    print(f"Prediction Confidence: {confidence:.2f}%")
    print(f"Estimated Tumor Area : {tumor_area:.2f}%")
    print(f"Tumor Severity       : {severity}")
    print("="*52 + "\n")
    
    # Create visualization
    heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    original_img_array = np.array(original_img)
    overlay = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
    
    # Save individual images
    output_path = "final_prediction_result.png"
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("heatmap_only.png", heatmap_colored)
    print(f"✅ Result saved as {output_path}")
    print(f"✅ Heatmap saved as heatmap_only.png")
    
    # Display with better layout
    fig = plt.figure(figsize=(15, 5))
    
    # Original
    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original MRI Scan", fontsize=14, weight='bold', pad=10)
    plt.axis("off")
    
    # Heatmap
    ax2 = plt.subplot(1, 3, 2)
    im = plt.imshow(heatmap_resized, cmap='jet')
    plt.title("Activation Heatmap", fontsize=14, weight='bold', pad=10)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Activation', rotation=270, labelpad=15)
    
    # Overlay
    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    title_text = f"Prediction: {CLASS_NAMES[pred_index].upper()}\nConfidence: {confidence:.1f}% | Area: {tumor_area:.1f}%"
    plt.title(title_text, fontsize=14, weight='bold', pad=10)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("comparison_result.png", dpi=150, bbox_inches='tight')
    print("✅ Comparison saved as comparison_result.png")
    plt.show()
    
    return {
        'tumor_type': CLASS_NAMES[pred_index],
        'confidence': confidence,
        'tumor_area': tumor_area,
        'severity': severity,
        'all_predictions': {CLASS_NAMES[i]: float(preds[0][i]*100) for i in range(len(CLASS_NAMES))}
    }

# =========================
# BATCH ANALYSIS (BONUS)
# =========================
def analyze_multiple_images(image_paths):
    """Analyze multiple images at once"""
    results = []
    for img_path in image_paths:
        try:
            print(f"\n{'='*60}")
            result = generate_report(img_path)
            results.append({'image': img_path, **result})
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            results.append({'image': img_path, 'error': str(e)})
    return results

# =========================
# RUN
# =========================
if __name__ == "__main__":
    try:
        result = generate_report("sample_mri.jpg")
        print("\n✅ Analysis complete!")
        print(f"\n📋 Summary:")
        print(f"   🧠 Diagnosis: {result['tumor_type'].upper()}")
        print(f"   📊 Confidence: {result['confidence']:.1f}%")
        print(f"   📏 Tumor Area: {result['tumor_area']:.1f}%")
        print(f"   ⚠️  Severity: {result['severity']}")
        
    except FileNotFoundError:
        print("❌ Error: sample_mri.jpg not found!")
        print("Please ensure the image file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()