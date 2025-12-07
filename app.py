import sys
import json
import cv2
import numpy as np
import trimesh
import pyrender
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap


class FaceRecognizerDesktop(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Reconhecimento Facial - Desktop")
        self.resize(900, 600)

        # Interface
        self.video_label = QLabel("Iniciando câmera…")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.model_label = QLabel("Status do Modelo GLB: aguardando detecção")
        self.btn_exit = QPushButton("Fechar")
        self.btn_exit.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.model_label)
        layout.addWidget(self.btn_exit)
        self.setLayout(layout)

        # Iniciar webcam
        self.cap = cv2.VideoCapture(0)

        # Carregar o modelo LBPH treinado
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("recognizer/trainer.yml")

        # Carregar labels.json
        with open("recognizer/labels.json", "r") as f:
            self.labels = json.load(f)
        self.labels_inv = {v: k for k, v in self.labels.items()}

        # Carregar modelo GLB
        self.mesh = trimesh.load("models/pet_red.glb")
        self.scene = pyrender.Scene()
        self.node = pyrender.Mesh.from_trimesh(self.mesh)
        self.scene.add(self.node)

        self.viewer = pyrender.Viewer(
            self.scene,
            use_raymond_lighting=True,
            run_in_thread=True,
            title="Modelo GLB"
        )

        # Timer para capturar frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # -----------------------
    # Loop Principal de Vídeo
    # -----------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
            .detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        user_status = "desconhecido"

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]

            id_, confidence = self.recognizer.predict(roi)

            if confidence < 50:  # Quanto menor, mais certeza
                name = self.labels_inv[id_]
                user_status = "conhecido"
                cv2.putText(frame, f"{name} ({confidence:.1f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                # Se conhecido → animação conhecida
                self.model_label.setText("Modelo GLB: Animação → CONHECIDO")
                self.apply_known_animation()
            else:
                user_status = "desconhecido"
                cv2.putText(frame, "DESCONHECIDO", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # Se desconhecido → animação desconhecida
                self.model_label.setText("Modelo GLB: Animação → DESCONHECIDO")
                self.apply_unknown_animation()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Exibir frame na janela PyQt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    # -----------------------
    # Animação para CONHECIDO
    # -----------------------
    def apply_known_animation(self):
        """
        Aqui você pode trocar animações do GLB se seu arquivo tiver animações,
        ou alterar transformações no mesh.
        """
        self.node.mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(10), [0, 1, 0]
        ))

    # ---------------------------
    # Animação para DESCONHECIDO
    # ---------------------------
    def apply_unknown_animation(self):
        self.node.mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(-10), [0, 1, 0]
        ))

    # -----------------------
    # Encerrar Programa
    # -----------------------
    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


# -----------------------
# Inicialização da Aplicação
# -----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognizerDesktop()
    window.show()
    sys.exit(app.exec())
