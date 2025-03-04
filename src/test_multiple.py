import os
import random
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Etichette delle emozioni (devono corrispondere all'ordine del training)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Carica il modello
model_path = '../models/emotion_model_improved.h5'  # Aggiorna se necessario
model = load_model(model_path)

# Directory di test
test_dir = '../data/test'  # Aggiorna se necessario

# Quante immagini vuoi analizzare al massimo
max_images = 20

# Lista per salvare i path delle immagini e la label corrispondente
test_images = []

print(f"[INFO] Leggo un campione dalla cartella di test: {test_dir}")

# Scorri le sottocartelle (una per ogni emozione)
for label in emotion_labels:
    label_dir = os.path.join(test_dir, label)
    if not os.path.isdir(label_dir):
        print(f"[ATTENZIONE] La cartella {label_dir} non esiste, salto...")
        continue

    # Prendi tutti i file immagine in questa cartella
    image_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    for img_file in image_files:
        img_path = os.path.join(label_dir, img_file)
        # Salviamo la tupla (img_path, label) per predire dopo
        test_images.append((img_path, label))

# Se non ci sono immagini, usciamo
if not test_images:
    print("\n[Nessuna immagine trovata nelle sottocartelle di test!]")
    exit()

# Mescola le immagini e prendi un campione di max_images
random.shuffle(test_images)
sample = test_images[:max_images]

# Ora eseguiamo la predizione su questo campione
total_images = 0
correct_predictions = 0

print(f"\n[INFO] Analizzo un campione di {len(sample)} immagini...\n")

for i, (img_path, true_label) in enumerate(sample, 1):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"{i}) Impossibile caricare {img_path}, salto...")
        continue

    # Se le immagini non sono già 48x48, ridimensioniamo (decommenta se necessario)
    # img = cv2.resize(img, (48, 48))

    roi = img.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)   # (1, 48, 48)
    roi = np.expand_dims(roi, axis=-1)  # (1, 48, 48, 1)

    preds = model.predict(roi)
    emotion_index = int(np.argmax(preds))
    predicted_label = emotion_labels[emotion_index]

    total_images += 1
    if predicted_label == true_label:
        correct_predictions += 1

    # Stampa i risultati
    print(f"{i}) File: {img_path}")
    print(f"   - Etichetta reale: {true_label}")
    print(f"   - Predizione: {predicted_label}")
    print(f"   - Probabilità: {preds[0]}")
    print("")

# Calcolo dell'accuratezza su questo campione
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"[RISULTATO] Immagini analizzate: {total_images}")
print(f"Predizioni corrette: {correct_predictions}")
print(f"Accuratezza su questo campione: {accuracy:.2f}%")

print("\nScript completato. Ora puoi scrivere nuovamente sul terminale.")
