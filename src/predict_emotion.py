import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Definisci le etichette delle emozioni (deve corrispondere all'ordine usato durante l'addestramento)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Carica il modello (modifica il path se necessario)
model_path = '../models/emotion_model_improved.h5'
model = load_model(model_path)

# Inserisci il percorso dell'immagine che vuoi analizzare
image_path = '../test.jpeg'
img = cv2.imread(image_path)

if img is None:
    print("Immagine non trovata!")
    exit()

# Converte l'immagine in scala di grigi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (Opzionale) Rilevamento del volto con Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Funzione per mostrare la finestra OpenCV in modo non bloccante:
def show_image_nonblocking(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    # Attendi fino a che l'utente preme 'q' oppure la finestra viene chiusa
    while True:
        key = cv2.waitKey(1) & 0xFF
        # Se l'utente preme 'q', esci
        if key == ord('q'):
            break
        # Se la finestra viene chiusa, getWindowProperty ritorna un valore < 1
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(window_name)

# Funzione per plottare le probabilità di predizione
def plot_prediction(probs, labels):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, probs, color='skyblue')
    plt.title("Probabilità di Predizione")
    plt.xlabel("Emozioni")
    plt.ylabel("Probabilità")
    plt.ylim(0, 1)
    plt.show()

# Se nessun volto è rilevato, usa l'intera immagine
if len(faces) == 0:
    print("Nessun volto rilevato, uso l'intera immagine per la predizione.")
    roi = cv2.resize(gray, (48, 48))
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)  # dimensione batch
    roi = np.expand_dims(roi, axis=-1) # canale (grigio)
    
    preds = model.predict(roi)
    emotion_index = np.argmax(preds)
    emotion_text = emotion_labels[emotion_index]
    
    # Stampa l'emozione trovata
    print("Emozione predetta:", emotion_text)
    
    cv2.putText(img, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)
    
    # Mostra la finestra con il risultato in modalità non bloccante (premi 'q' per chiudere)
    show_image_nonblocking("Emotion Recognition", img)
    
    # Plotta il grafico delle probabilità
    plot_prediction(preds[0], emotion_labels)

# Se vengono rilevati uno o più volti
else:
    all_preds = []  # per salvare le predizioni (ad es. solo per il primo volto)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        preds = model.predict(roi)
        all_preds.append(preds[0])
        emotion_index = np.argmax(preds)
        emotion_text = emotion_labels[emotion_index]
        
        # Stampa l'emozione trovata per il volto corrente
        print("Emozione predetta per volto in posizione ({}, {}): {}".format(x, y, emotion_text))
        
        # Disegna un rettangolo attorno al volto e annota l'emozione
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)
    
    show_image_nonblocking("Emotion Recognition", img)
    
    # Plotta il grafico per il primo volto rilevato
    if all_preds:
        plot_prediction(all_preds[0], emotion_labels)

print("Script completato. Ora puoi scrivere nuovamente sul terminale.")
