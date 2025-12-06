import os
import json
import time
from threading import Thread

from flask import Flask, render_template
from flask_socketio import SocketIO

import cv2
import numpy as np

# Config
TRAINER_PATH = 'trainer/trainer.yml'
LABELS_PATH = 'trainer/labels.json'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 60.0  # LBPH: quanto menor, melhor (ajuste conforme seu modelo)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Carrega recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(TRAINER_PATH):
    recognizer.read(TRAINER_PATH)
    print(f"Modelo carregado: {TRAINER_PATH}")
else:
    print(f"Modelo não encontrado em {TRAINER_PATH}. Execute o script de treino primeiro.")

# labels
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
else:
    labels = {}

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

camera = None
running = False


def recognition_loop():
    global camera, running
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print('Não foi possível abrir a câmera.')
        return

    running = True
    while running:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        # Se detectar múltiplas faces, itera
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            try:
                id_, confidence = recognizer.predict(face_roi)
            except Exception as e:
                # Em caso de erro na predição
                print('Erro predict:', e)
                continue

            # LBPH retorna 'confidence' onde menor é melhor
            if confidence <= CONFIDENCE_THRESHOLD and str(id_) in labels:
                name = labels[str(id_)]
                payload = {'status': 'known', 'label': name, 'confidence': float(confidence)}
                socketio.emit('recognition', payload)
                print('Conhecido:', payload)
            else:
                payload = {'status': 'unknown'}
                socketio.emit('recognition', payload)
                print('Desconhecido')

            # Só processa a primeira face detectada por frame para simplificar
            break

        # pequena pausa para reduzir uso de CPU
        socketio.sleep(0.05)

    camera.release()


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def on_connect():
    print('Cliente conectado')


@socketio.on('start_rec')
def on_start(data):
    # Inicia thread de reconhecimento se não estiver rodando
    global running
    if not running:
        thread = Thread(target=recognition_loop, daemon=True)
        thread.start()
        print('Loop de reconhecimento iniciado')


@socketio.on('stop_rec')
def on_stop(data):
    global running
    running = False


if __name__ == '__main__':
    # Executar com eventlet para suporte websocket
    socketio.run(app, host='0.0.0.0', port=5000)