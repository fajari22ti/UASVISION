"""
Sistem Monitoring Ketersediaan Slot Parkir Otomatis
Backend Flask - Rizki Fajari (2255301167)
"""

import os
import io
import base64
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from skimage.feature import hog
from skimage.filters import gabor

# ── Init Flask ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load Model ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parking_slot_model.pkl')
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# ── Config Fitur (harus sama persis dengan saat training) ─────────────────
IMG_SIZE          = (64, 64)
HOG_ORIENTATIONS  = 9
HOG_PIXELS_CELL   = (8, 8)
HOG_CELLS_BLOCK   = (2, 2)
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS      = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# ── Koordinat ROI default (dapat diubah dari frontend) ────────────────────
# Format: [[x, y, w, h], ...]
DEFAULT_ROI = [
    [30,  50, 80, 100],
    [120, 50, 80, 100],
    [210, 50, 80, 100],
    [300, 50, 80, 100],
    [390, 50, 80, 100],
]


# ── Fungsi Ekstraksi Fitur ────────────────────────────────────────────────
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
            feats.append(energy.mean())
            feats.append(energy.std())
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
    """Deteksi status setiap slot parkir."""
    results  = []
    canvas   = img_bgr.copy()
    n_tersedia = 0

    for idx, (x, y, w, h) in enumerate(roi_coords):
        # Pastikan ROI dalam batas gambar
        x, y = max(0, x), max(0, y)
        w = min(w, img_bgr.shape[1] - x)
        h = min(h, img_bgr.shape[0] - y)

        roi = img_bgr[y:y+h, x:x+w]
        if roi.size == 0 or w < 10 or h < 10:
            continue

        feats  = extract_combined_features(roi).reshape(1, -1)
        pred   = int(model.predict(feats)[0])
        proba  = model.predict_proba(feats)[0]
        conf   = float(proba[pred])

        status    = "Tersedia" if pred == 0 else "Penuh"
        box_color = (0, 200, 0) if pred == 0 else (0, 0, 220)   # BGR

        if pred == 0:
            n_tersedia += 1

        # Gambar bounding box
        cv2.rectangle(canvas, (x, y), (x+w, y+h), box_color, 2)

        # Label background
        label  = f"S{idx+1} {status} {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x, max(0, y-th-6)), (x+tw+4, y), box_color, -1)
        cv2.putText(canvas, label, (x+2, max(8, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        results.append({
            "slot_id"   : idx + 1,
            "status"    : status,
            "confidence": round(conf * 100, 1),
            "roi"       : [x, y, w, h]
        })

    # Info counter pojok kiri atas
    total    = len(results)
    info_str = f"Tersedia: {n_tersedia}/{total}"
    cv2.rectangle(canvas, (5, 5), (220, 38), (20, 20, 20), -1)
    cv2.putText(canvas, info_str, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return canvas, results, n_tersedia, total


def img_to_base64(img_bgr):
    """Konversi gambar OpenCV ke base64 string untuk dikirim ke frontend."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _, buf  = cv2.imencode('.jpg', img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect/image', methods=['POST'])
def detect_image():
    """Upload gambar → deteksi slot parkir."""
    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada file gambar"}), 400

    file    = request.files['image']
    roi_raw = request.form.get('roi', None)

    # Parse ROI dari frontend (JSON string)
    if roi_raw:
        import json
        try:
            roi_coords = json.loads(roi_raw)
        except Exception:
            roi_coords = DEFAULT_ROI
    else:
        roi_coords = DEFAULT_ROI

    # Baca gambar
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Gambar tidak valid"}), 400

    # Deteksi
    canvas, results, n_tersedia, total = detect_slots(img_bgr, roi_coords)

    return jsonify({
        "success"    : True,
        "image_b64"  : img_to_base64(canvas),
        "results"    : results,
        "n_tersedia" : n_tersedia,
        "total"      : total,
        "n_penuh"    : total - n_tersedia
    })


@app.route('/detect/frame', methods=['POST'])
def detect_frame():
    """Terima frame base64 dari kamera live → deteksi slot."""
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({"error": "Frame tidak ditemukan"}), 400

    # Decode base64 frame
    frame_data  = data['frame'].split(',')[-1]   # hapus prefix data:image/...
    roi_coords  = data.get('roi', DEFAULT_ROI)

    img_bytes   = base64.b64decode(frame_data)
    img_array   = np.frombuffer(img_bytes, np.uint8)
    img_bgr     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Frame tidak valid"}), 400

    canvas, results, n_tersedia, total = detect_slots(img_bgr, roi_coords)

    return jsonify({
        "success"    : True,
        "image_b64"  : img_to_base64(canvas),
        "results"    : results,
        "n_tersedia" : n_tersedia,
        "total"      : total,
        "n_penuh"    : total - n_tersedia
    })


@app.route('/health')
def health():
    return jsonify({"status": "ok", "model": "parking_slot_model.pkl"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
