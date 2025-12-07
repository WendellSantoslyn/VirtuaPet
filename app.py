# app.py
import os
import sys
import time
import pickle

import cv2
import numpy as np
import trimesh
import pyrender

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QPlainTextEdit
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

# -------------- Configurações --------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(BASE_DIR, "recognizer")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SSD_PROTO = os.path.join(REC_DIR, "deploy.prototxt.txt")
SSD_MODEL = os.path.join(REC_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
PICKLE_PATH = os.path.join(REC_DIR, "face_names.pickle")
LBPH_PATH = os.path.join(REC_DIR, "lbph_classifier.yml")
GLB_PATH = os.path.join(MODEL_DIR, "pet.glb")

# Thresholds
SSD_CONF_MIN = 0.7    # confidence to accept SSD detections (0..1)
LBPH_CONF_THRESHOLD = 80.0  # LBPH: lower is better; accept if <= this


# ---------------- utilities ----------------
def safe_load_pickle(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------- recognition logic ----------------
def load_recognizer(path):
    """Load LBPH recognizer from file (requires opencv-contrib)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recognizer file not found: {path}")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(path)
    return recognizer


def recognize_faces(network, face_classifier, orig_frame, face_names, threshold_val=LBPH_CONF_THRESHOLD, conf_min=SSD_CONF_MIN):
    """
    Detect faces via SSD network, then predict with face_classifier (LBPH).
    Returns: processed_frame, detection_info (dict or None)
    detection_info contains keys: pred_id (int), pred_name (str), conf (float), bbox (tuple)
    If no face detected → detection_info is None
    """
    frame = orig_frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    best_conf = 0.0
    best_bbox = None
    # find best detection
    for i in range(0, detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > conf_min and conf > best_conf:
            bb = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_conf = conf
            best_bbox = bb.astype("int")

    if best_bbox is None:
        return frame, None

    (start_x, start_y, end_x, end_y) = best_bbox
    # clamp
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(w - 1, end_x)
    end_y = min(h - 1, end_y)

    # ROI and resize for recognizer
    if end_x <= start_x or end_y <= start_y:
        return frame, None

    face_roi = gray[start_y:end_y, start_x:end_x]
    # Resize to expected size (the training script used 200x200 earlier)
    try:
        face_resized = cv2.resize(face_roi, (200, 200))
    except Exception:
        return frame, None

    detection_info = {"pred_id": None, "pred_name": None, "conf": None, "bbox": (start_x, start_y, end_x, end_y)}

    # Predict with classifier (if available)
    if face_classifier is not None:
        try:
            prediction, conf = face_classifier.predict(face_resized)
            detection_info["pred_id"] = int(prediction)
            detection_info["conf"] = float(conf)
            # Map numeric prediction to name using face_names (face_names was inverted so keys are ints or strings)
            # face_names likely maps numeric id -> human name (loaded from pickle and inverted earlier)
            name = face_names.get(prediction) if isinstance(face_names.keys(), (list, tuple)) else face_names.get(prediction)
            # face_names may have string keys depending on how it was created; handle both:
            if name is None:
                # try string key
                name = face_names.get(str(prediction))
            detection_info["pred_name"] = name
        except Exception:
            detection_info["pred_id"] = None
            detection_info["conf"] = None
            detection_info["pred_name"] = None

    # draw rectangle (colored later by caller depending on recognition)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2)

    return frame, detection_info


# ---------------- GUI App ----------------
class FaceRecognizerDesktop(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VirtuaPet - Reconhecimento (SSD + LBPH)")
        self.resize(960, 720)

        # UI layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # camera preview
        self.camera_label = QLabel("Aguardando câmera...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.camera_label)

        # log box
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(160)
        layout.addWidget(self.log_box)

        # close button
        btn_close = QPushButton("Fechar Aplicação")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

        # load resources
        self.log("Aplicação iniciada.")
        self.log(f"Recognizer directory: {REC_DIR}")

        # load SSD detector
        self.detector = None
        try:
            self.log(f"Carregando detector SSD: {SSD_PROTO}, {SSD_MODEL}")
            self.detector = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
            self.log("Detector SSD carregado.")
        except Exception as e:
            self.log(f"ERRO: não foi possível carregar SSD detector: {e}")

        # load face_names.pickle (original labels inverted as in your snippet)
        self.face_names = {}
        if os.path.exists(PICKLE_PATH):
            try:
                orig = safe_load_pickle(PICKLE_PATH)
                # original_labels presumably maps name->id, the snippet inverted to id->name
                # We will invert if needed to ensure mapping numeric id -> name
                # If orig keys are names and values are ids, invert
                keys_are_str_ids = all(isinstance(k, (str,)) and str(k).isdigit() for k in orig.keys())
                # The uploaded snippet did: original_labels = pickle.load; face_names = {v: k for k, v in original_labels.items()}
                # So orig likely is original_labels (name->id). Let's invert to id->name:
                inv = {v: k for k, v in orig.items()}
                # Keep numeric keys as int for direct lookup
                self.face_names = {}
                for k, v in inv.items():
                    try:
                        key_int = int(k)
                    except Exception:
                        key_int = k
                    self.face_names[key_int] = v
                self.log(f"face_names carregado: {len(self.face_names)} entradas.")
            except Exception as e:
                self.log(f"ERRO ao carregar face_names.pickle: {e}")
        else:
            self.log("Aviso: face_names.pickle não encontrado.")

        # load LBPH recognizer
        self.recognizer = None
        if os.path.exists(LBPH_PATH):
            try:
                self.recognizer = load_recognizer(LBPH_PATH)
                self.log("LBPH recognizer carregado.")
            except Exception as e:
                self.log(f"ERRO ao carregar LBPH recognizer: {e}")
        else:
            self.log("Aviso: arquivo LBPH não encontrado.")

        # load GLB model (optional — used for display; safe fallback)
        self.renderer = None
        try:
            if os.path.exists(GLB_PATH):
                scene = trimesh.load(GLB_PATH, force="scene")
                mesh = next(iter(scene.geometry.values())) if isinstance(scene, trimesh.Scene) else scene
                node = pyrender.Mesh.from_trimesh(mesh)
                self.pyr_scene = pyrender.Scene()
                self.pyr_scene.add(node)
                cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                self.pyr_scene.add(cam, pose=np.eye(4))
                self.renderer = pyrender.OffscreenRenderer(320, 320)
                self.log("Modelo GLB carregado e renderer inicializado.")
            else:
                self.log("GLB não encontrado (pular render 3D).")
        except Exception as e:
            self.log(f"Erro carregando GLB/renderer: {e}")
            self.renderer = None

        # open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log("ERRO: não foi possível abrir a webcam.")
        else:
            self.log("Webcam aberta.")

        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(30)

        # for rate-limiting logs (avoid spam)
        self._last_log_time = 0.0

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {message}")

    def on_timer(self):
        if self.detector is None or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Optionally resize frame to speed up processing (kept original size here)
        processed_frame, info = recognize_faces(
            self.detector,
            self.recognizer,
            frame,
            self.face_names,
            threshold_val=LBPH_CONF_THRESHOLD,
            conf_min=SSD_CONF_MIN
        )

        # Draw results based on info
        if info is None:
            # no detection
            pass
        else:
            pred_id = info.get("pred_id")
            pred_name = info.get("pred_name")
            conf = info.get("conf")

            # Normalize the pred_name (if None try mapping via pred_id)
            if pred_name is None and pred_id is not None:
                pred_name = self.face_names.get(pred_id) or self.face_names.get(str(pred_id))

            # Visual and log
            bbox = info.get("bbox")
            (sx, sy, ex, ey) = bbox
            if pred_id is not None and pred_name is not None and conf is not None and conf >= LBPH_CONF_THRESHOLD:
                color = (0, 255, 0)
                text = f"{pred_name} -> {conf:.2f}"
                # log once per change or every 1.5s
                now = time.time()
                if now - self._last_log_time > 1.5:
                    self.log(f"Reconhecido: {pred_name} (id {pred_id}, conf {conf:.2f})")
                    self._last_log_time = now
            else:
                color = (0, 0, 255)
                text = f"Desconhecido -> {conf:.2f}" if conf is not None else "Desconhecido"
                now = time.time()
                if now - self._last_log_time > 1.5:
                    self.log(f"Desconhecido (id {pred_id}, conf {conf})")
                    self._last_log_time = now

            # Draw rectangle and text (note: recognize_faces already drew rectangle in yellow)
            cv2.rectangle(processed_frame, (sx, sy), (ex, ey), color, 2)
            cv2.putText(processed_frame, text, (sx, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # show GLB render (if available) — we won't embed into GUI canvas, only keep it ready
        if self.renderer is not None:
            try:
                color_img, _ = self.renderer.render(self.pyr_scene)
                # we won't display the 3D render in this version; it's kept as optional feature
            except Exception as e:
                # log OpenGL errors occasionally
                now = time.time()
                if now - self._last_log_time > 3:
                    self.log(f"Erro OpenGL render: {e}")
                    self._last_log_time = now

        # Convert processed_frame to QImage and display
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qt_img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.log("Aplicação encerrada.")
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = FaceRecognizerDesktop()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
