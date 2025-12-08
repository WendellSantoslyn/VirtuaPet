import os
import sys
import time
import pickle

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QPlainTextEdit
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

# ---------------------- CONFIG ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(BASE_DIR, "recognizer")

SSD_PROTO = os.path.join(REC_DIR, "deploy.prototxt.txt")
SSD_MODEL = os.path.join(REC_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
PICKLE_PATH = os.path.join(REC_DIR, "face_names.pickle")
LBPH_PATH = os.path.join(REC_DIR, "lbph_classifier.yml")

SSD_CONF_MIN = 0.7
LBPH_CONF_THRESHOLD = 80.0  # LBPH: lower = better (we accept <= threshold)


# ---------------------- UTILS ----------------------
def safe_load_pickle(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def load_recognizer(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recognizer not found: {path}")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(path)
    return recognizer


def detect_and_recognize(detector, recognizer, frame, face_names):
    """
    Detect with SSD -> recognize with LBPH.
    Returns (processed_frame, info) where info is None or dict with keys:
      - id, name, conf, box (x1,y1,x2,y2)
    """
    frame_out = frame.copy()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return frame_out, None

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                 (104.0, 117.0, 123.0))
    try:
        detector.setInput(blob)
        detections = detector.forward()
    except Exception:
        return frame_out, None

    best_conf = 0.0
    best_box = None
    for i in range(0, detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > SSD_CONF_MIN and conf > best_conf:
            bb = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_conf = conf
            best_box = bb.astype(int)

    if best_box is None:
        return frame_out, None

    x1, y1, x2, y2 = best_box
    # clamp
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_out, None

    roi_gray = gray[y1:y2, x1:x2]
    try:
        face_resized = cv2.resize(roi_gray, (200, 200))
    except Exception:
        return frame_out, None

    # Predict with recognizer
    try:
        pred_id, conf = recognizer.predict(face_resized)
    except Exception:
        return frame_out, None

    # Map id -> name (handle int or str keys)
    name = face_names.get(pred_id) or face_names.get(str(pred_id))

    # Draw a yellow SSD box first (will be overwritten by caller with color)
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return frame_out, {
        "id": int(pred_id),
        "name": name,
        "conf": float(conf),
        "box": (x1, y1, x2, y2)
    }


# ---------------------- APP ----------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Controle de Acesso - Reconhecimento Facial")
        self.resize(1000, 720)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Camera display
        self.camera_label = QLabel("Inicializando câmera...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumHeight(420)
        main_layout.addWidget(self.camera_label)

        # Logs - side by side
        logs_layout = QHBoxLayout()

        # Operations log (left)
        self.operations_log = QPlainTextEdit()
        self.operations_log.setReadOnly(True)
        self.operations_log.setMaximumHeight(200)
        self.operations_log.setPlaceholderText("Log de Operações (atualiza a cada 3s)...")
        logs_layout.addWidget(self._make_labeled_widget("Operações", self.operations_log))

        # Access log (right)
        self.access_log = QPlainTextEdit()
        self.access_log.setReadOnly(True)
        self.access_log.setMaximumHeight(200)
        self.access_log.setPlaceholderText("Log de Acessos (registra apenas novos acessos)...")
        logs_layout.addWidget(self._make_labeled_widget("Acessos", self.access_log))

        main_layout.addLayout(logs_layout)

        # Close button
        btn_close = QPushButton("Fechar Sistema")
        btn_close.clicked.connect(self.close)
        main_layout.addWidget(btn_close)

        # internal state for rate limiting / single registration
        self.last_access_time = 0.0
        self.last_op_time = 0.0
        self.last_pid = None  # last successfully logged PID

        # load models with safe logging
        self._safe_load_models()

        # start capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log_operation("ERRO: não foi possível abrir a webcam.")
        else:
            self.log_operation("Webcam iniciada.")

        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    def _make_labeled_widget(self, title: str, widget: QPlainTextEdit):
        """Return a QWidget-like (QVBoxLayout in a QWidget) with title and widget."""
        container = QWidget()
        v = QVBoxLayout()
        container.setLayout(v)
        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(label)
        v.addWidget(widget)
        return container

    def _safe_load_models(self):
        """Load detector, recognizer and names, but never crash the UI — log errors instead."""
        # face_names
        try:
            orig = safe_load_pickle(PICKLE_PATH)
            # invert if mapping is name->id; we want id->name
            try:
                inv = {v: k for k, v in orig.items()}
                # coerce keys to int when possible
                self.face_names = {}
                for k, v in inv.items():
                    try:
                        k_int = int(k)
                    except Exception:
                        k_int = k
                    self.face_names[k_int] = v
            except Exception:
                # fallback: assume orig already maps id->name
                self.face_names = orig
            self.log_operation(f"face_names carregado: {len(self.face_names)} entradas.")
        except Exception as e:
            self.face_names = {}
            self.log_operation(f"Erro ao carregar face_names.pickle: {e}")

        # detector
        try:
            self.detector = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
            self.log_operation("Detector SSD carregado.")
        except Exception as e:
            self.detector = None
            self.log_operation(f"Erro ao carregar detector SSD: {e}")

        # recognizer
        try:
            self.recognizer = load_recognizer(LBPH_PATH)
            self.log_operation("LBPH carregado.")
        except Exception as e:
            self.recognizer = None
            self.log_operation(f"Erro ao carregar LBPH recognizer: {e}")

    # ---- Logging helpers ----
    def log_operation(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.operations_log.appendPlainText(f"[{ts}] {msg}")

    def log_access(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.access_log.appendPlainText(f"[{ts}] {msg}")

    # ---- Frame loop ----
    def update_frame(self):
        if getattr(self, "detector", None) is None or getattr(self, "recognizer", None) is None:
            # still show camera preview if possible
            ret, frame = (False, None)
            if hasattr(self, "cap") and self.cap is not None:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(qimg))
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        processed_frame, info = detect_and_recognize(
            self.detector, self.recognizer, frame, self.face_names
        )

        now = time.time()

        if info:
            pid = info.get("id")
            name = info.get("name")
            conf = info.get("conf")
            x1, y1, x2, y2 = info.get("box")

            # ACCESS GRANTED: register only once per user (or after timeout)
            if conf is not None and conf >= LBPH_CONF_THRESHOLD and name is not None:
                color = (0, 255, 0)
                text = f"{name} ({conf:.2f})"

                # register access only if different PID or after timeout (3s)
                if (self.last_pid is None) or (self.last_pid != pid) or (now - self.last_access_time > 3.0):
                    self.log_access(f"Acesso liberado para {name} (ID {pid}, conf {conf:.2f})")
                    self.last_access_time = now
                    self.last_pid = pid

                # operations log limited to once every 3s
                if (self.last_op_time is None) or (now - self.last_op_time > 3.0):
                    self.log_operation(f"Reconhecimento positivo: {name} (ID {pid}, conf {conf:.2f})")
                    self.last_op_time = now

            # ACCESS DENIED: limited ops log only
            else:
                color = (0, 0, 255)
                text = f"Acesso negado ({None if conf is None else f'{conf:.2f}'})"
                if (self.last_op_time is None) or (now - self.last_op_time > 3.0):
                    self.log_operation(f"Tentativa de acesso negada (id {pid}, conf {conf})")
                    self.last_op_time = now

                # reset last_pid so a later success for same pid is treated as a new access
                self.last_pid = None

            # draw final colored rectangle and label
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show frame
        try:
            rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.log_operation("Aplicação encerrada.")
        event.accept()


# ---------------------- MAIN ----------------------
def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
