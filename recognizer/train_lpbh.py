import os
import json
import cv2
import numpy as np

DATASET_DIR = "dataset"
RECOGNIZER_DIR = "../recognizer"

os.makedirs(RECOGNIZER_DIR, exist_ok=True)

# Criar detector HaarCascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
labels = {}
faces = []
ids = []

print("🔍 Lendo dataset…")

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(("jpg", "jpeg", "png", "bmp")):

            path = os.path.join(root, file)
            label = os.path.basename(root)  # nome da pasta = nome da pessoa

            if label not in labels:
                labels[label] = current_id
                current_id += 1

            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detectar rosto
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces_rect:
                roi = gray[y:y + h, x:x + w]
                faces.append(roi)
                ids.append(labels[label])

print(f"📊 Total de faces detectadas: {len(faces)}")
print(f"👤 Labels gerados: {labels}")

# Treinar modelo LBPH
if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.save(os.path.join(RECOGNIZER_DIR, "trainer.yml"))
    print("✔️ Modelo LBPH salvo em recognizer/trainer.yml")
else:
    print("❌ Nenhuma face encontrada. Verifique o dataset.")

# Salvar labels.json
with open(os.path.join(RECOGNIZER_DIR, "labels.json"), "w") as f:
    json.dump(labels, f, indent=4)

print("✔️ labels.json salvo em recognizer/labels.json")
print("🏁 Finalizado!")