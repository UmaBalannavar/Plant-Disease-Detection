import os
import io
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import traceback

# ─────────────────────────────────────────────
# Lazy-load TF/Keras so Flask starts even if TF
# is not yet installed in the target environment
# ─────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not found. Install it with:  pip install tensorflow")

# ── Directory layout ─────────────────────────
#   PLANT_DISEASE_DETECTION/
#   ├── backend/
#   │   ├── app.py                  ← this file
#   │   └── mobilenet_plant_mo...   ← model
#   └── frontend/
#       └── index.html
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))          # …/backend
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")            # …/frontend

app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_DIR, "static"),   # frontend/static (optional assets)
    template_folder=FRONTEND_DIR,                          # serves index.html from frontend/
)
CORS(app)   # allow frontend (e.g. live-server on :5500) to call backend on :5000

# ── Config ──────────────────────────────────
MODEL_PATH    = os.path.join(BASE_DIR, "mobilenet_plant_model.keras")
IMG_SIZE      = (224, 224)          # MobileNet default input
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Plant class labels – update to match your model ──
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Blueberry___healthy",
    "Corn_(maize)___Common_rust_",
    "Grape___Black_rot",
]

# ── Load model once at startup ───────────────
model = None
if TF_AVAILABLE:
    try:
        import tensorflow as tf
        import json
        
        # Patch the Dense layer to ignore quantization_config
        original_dense_init = tf.keras.layers.Dense.__init__
        def patched_dense_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)  # Remove problematic argument
            return original_dense_init(self, *args, **kwargs)
        tf.keras.layers.Dense.__init__ = patched_dense_init
        
        # Load the model
        model = keras.models.load_model(MODEL_PATH, compile=False)
        
        # Restore original Dense layer
        tf.keras.layers.Dense.__init__ = original_dense_init
        
        print(f"[INFO] Model loaded from {MODEL_PATH}")
        print(f"[INFO] Input shape : {model.input_shape}")
        print(f"[INFO] Output shape: {model.output_shape}")
        
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print("[INFO] Application will run but predictions will return 'Model not loaded'")
        print("[INFO] Model loading failed due to version incompatibility")

# ────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize → RGB → [0,1] → add batch dim."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)


def ndarray_to_b64_png(arr: np.ndarray) -> str:
    """Convert a HxWx3 float32 [0,1] array to base-64 PNG string."""
    clipped = np.clip(arr, 0, 1)
    uint8   = (clipped * 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_layer_outputs(input_tensor: np.ndarray) -> list[dict]:
    """
    Run a sub-model up to the first 3 Conv/BN/Activation layers
    and return the first channel of each output as a base-64 heatmap.
    """
    if model is None:
        return []

    results = []
    visited = 0
    target_types = (
        keras.layers.Conv2D,
        keras.layers.DepthwiseConv2D,
        keras.layers.BatchNormalization,
        keras.layers.ReLU,
        keras.layers.Activation,
    )

    for layer in model.layers:
        if not isinstance(layer, target_types):
            continue
        try:
            sub = keras.Model(inputs=model.input, outputs=layer.output)
            feat_map = sub.predict(input_tensor, verbose=0)   # (1, H, W, C)

            # Take channel 0 and normalise to [0,1]
            channel = feat_map[0, :, :, 0]
            mn, mx  = channel.min(), channel.max()
            if mx - mn > 1e-8:
                channel = (channel - mn) / (mx - mn)
            else:
                channel = np.zeros_like(channel)

            # Convert greyscale → RGB heatmap (viridis-like using numpy)
            rgb = greyscale_to_heatmap(channel)
            b64 = ndarray_to_b64_png(rgb)

            results.append({
                "layer_name": layer.name,
                "layer_type": type(layer).__name__,
                "shape":      list(feat_map.shape[1:]),
                "image_b64":  b64,
                "matrix_sample": channel[:8, :8].round(4).tolist(),  # 8×8 preview
            })
            visited += 1
        except Exception:
            pass   # skip layers that fail (e.g. shape mismatches)

        if visited >= 3:
            break

    return results


def greyscale_to_heatmap(grey: np.ndarray) -> np.ndarray:
    """
    Map a 2-D [0,1] array → (H, W, 3) float32 using a cool-warm palette
    without requiring matplotlib.
    """
    r = np.clip(grey * 2,       0, 1)
    g = np.clip(grey * 1.5,     0, 1)
    b = np.clip(1 - grey * 1.5, 0, 1)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────

@app.route("/")
def index():
    # Serves PLANT_DISEASE_DETECTION/frontend/index.html
    return render_template("index.html")


@app.route("/static/frontend/<path:filename>")
def frontend_static(filename):
    """Serve any static assets placed in frontend/static/."""
    return send_from_directory(os.path.join(FRONTEND_DIR, "static"), filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream)
        tensor = preprocess_image(img)

        # ── Original image (resized) as base-64 ──
        original_b64 = ndarray_to_b64_png(tensor[0])

        # ── Prediction ──
        predictions = {}
        top_class   = "Model not loaded"
        confidence  = 0.0
        top5        = []

        if model is not None:
            preds = model.predict(tensor, verbose=0)[0]      # shape: (num_classes,)
            top_idx       = int(np.argmax(preds))
            top_class     = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class {top_idx}"
            confidence    = float(preds[top_idx])

            top5_idx = np.argsort(preds)[::-1][:5]
            top5 = [
                {
                    "class": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}",
                    "score": round(float(preds[i]) * 100, 2),
                }
                for i in top5_idx
            ]

        # ── Layer preprocessing visualisations ──
        layer_visuals = get_layer_outputs(tensor)

        return jsonify({
            "prediction":    top_class,
            "confidence":    round(confidence * 100, 2),
            "top5":          top5,
            "original_b64":  original_b64,
            "layer_visuals": layer_visuals,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def model_info():
    if model is None:
        return jsonify({"error": "Model not loaded"})
    layers_info = [
        {"name": l.name, "type": type(l).__name__,
         "params": l.count_params()}
        for l in model.layers
    ]
    return jsonify({
        "total_params": model.count_params(),
        "layers":       layers_info,
        "input_shape":  list(model.input_shape),
        "output_shape": list(model.output_shape),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)