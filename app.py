import os
import io
import base64
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from skimage.feature import hog
from skimage.filters import gabor

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parking_slot_model.pkl')
model = joblib.load(MODEL_PATH)
print(f"Model loaded OK")

# Config
IMG_SIZE          = (64, 64)
HOG_ORIENTATIONS  = 9
HOG_PIXELS_CELL   = (8, 8)
HOG_CELLS_BLOCK   = (2, 2)
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS      = [0, np.pi/4, np.pi/2, 3*np.pi/4]

DEFAULT_ROI = [
    [30,  50, 80, 100],
    [120, 50, 80, 100],
    [210, 50, 80, 100],
    [300, 50, 80, 100],
    [390, 50, 80, 100],
]

def preprocess_image(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    return gray.astype(np.float64) / 255.0

def extract_gabor_features(gray_img):
    feats = []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_THETAS:
            real, imag = gabor(gray_img, frequency=freq, theta=theta)
            energy = np.sqrt(real**2 + imag**2)
            feats.append(float(energy.mean()))
            feats.append(float(energy.std()))
    return np.array(feats)

def extract_hog_features(gray_img):
    return hog(
        gray_img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_CELL,
        cells_per_block=HOG_CELLS_BLOCK,
        visualize=False,
        channel_axis=None
    )

def extract_combined_features(img_bgr):
    gray   = preprocess_image(img_bgr)
    g_feat = extract_gabor_features(gray)
    h_feat = extract_hog_features(gray)
    return np.concatenate([g_feat, h_feat])

def detect_slots(img_bgr, roi_coords):
    results    = []
    canvas     = img_bgr.copy()
    n_tersedia = 0

    for idx, roi in enumerate(roi_coords):
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        x, y = max(0, x), max(0, y)
        w = min(w, img_bgr.shape[1] - x)
        h = min(h, img_bgr.shape[0] - y)
        if w < 10 or h < 10:
            continue

        patch = img_bgr[y:y+h, x:x+w]
        feats = extract_combined_features(patch).reshape(1, -1)
        pred  = int(model.predict(feats)[0])
        prob  = model.predict_proba(feats)[0]
        conf  = float(prob[pred])

        # Jika confidence penuh < 75%, anggap tersedia (lebih sensitif)
        if pred == 1 and conf < 0.75:
            pred = 0
            conf = float(prob[0])
        status    = "Tersedia" if pred == 0 else "Penuh"
        color     = (0, 200, 0) if pred == 0 else (0, 0, 220)
        if pred == 0:
            n_tersedia += 1

        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
        label = f"S{idx+1} {status} {conf*100:.0f}%"
        cv2.putText(canvas, label, (x+2, max(8, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        results.append({
            "slot_id": idx+1,
            "status": status,
            "confidence": round(conf*100, 1),
            "roi": [x, y, w, h]
        })

    total    = len(results)
    info_str = f"Tersedia: {n_tersedia}/{total}"
    cv2.rectangle(canvas, (5, 5), (220, 38), (20, 20, 20), -1)
    cv2.putText(canvas, info_str, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return canvas, results, n_tersedia, total

def img_to_base64(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400

    roi_raw    = request.form.get('roi', None)
    roi_coords = json.loads(roi_raw) if roi_raw else DEFAULT_ROI

    file_bytes = np.frombuffer(request.files['image'].read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid image"}), 400

    canvas, results, n_tersedia, total = detect_slots(img_bgr, roi_coords)
    return jsonify({
        "success"   : True,
        "image_b64" : img_to_base64(canvas),
        "results"   : results,
        "n_tersedia": n_tersedia,
        "total"     : total,
        "n_penuh"   : total - n_tersedia
    })

@app.route('/detect/frame', methods=['POST'])
def detect_frame():
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({"error": "No frame"}), 400

    frame_data  = data['frame'].split(',')[-1]
    roi_coords  = data.get('roi', DEFAULT_ROI)
    img_bytes   = base64.b64decode(frame_data)
    img_array   = np.frombuffer(img_bytes, np.uint8)
    img_bgr     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid frame"}), 400

    canvas, results, n_tersedia, total = detect_slots(img_bgr, roi_coords)
    return jsonify({
        "success"   : True,
        "image_b64" : img_to_base64(canvas),
        "results"   : results,
        "n_tersedia": n_tersedia,
        "total"     : total,
        "n_penuh"   : total - n_tersedia
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
