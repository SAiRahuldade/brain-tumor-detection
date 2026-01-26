import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
MODEL_PATH = "brain_tumor_best.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Force model build (VERY IMPORTANT)
dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model(dummy_input)

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Get last Conv2D layer automatically
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found")

LAST_CONV_LAYER = get_last_conv_layer(model)
print(f"✅ Using last conv layer: {LAST_CONV_LAYER}")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def generate_gradcam(img_array):
    conv_layer = model.get_layer(LAST_CONV_LAYER)

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)

        # Forward pass manually
        x = inputs
        for layer in model.layers:
            x = layer(x)
            if layer == conv_layer:
                conv_outputs = x

        predictions = x
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # SAFETY CHECK
    if grads is None:
        raise RuntimeError("Gradients are None — graph not connected")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy(), pred_index.numpy(), predictions.numpy()

def show_gradcam(img_path):
    img_array = preprocess_image(img_path)
    heatmap, pred_index, predictions = generate_gradcam(img_array)

    pred_class = class_labels[pred_index]
    confidence = np.max(predictions) * 100

    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original MRI")
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"{pred_class} ({confidence:.2f}%)")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# RUN
show_gradcam("sample_mri.jpg")
