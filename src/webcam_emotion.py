import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Definisci le etichette delle emozioni (deve corrispondere all'ordine usato durante l'addestramento)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Carica il modello (aggiorna il path se necessario)
model_path = '../models/emotion_model_improved.h5'
model = load_model(model_path)

# Crea il classificatore Haar Cascade per la rilevazione del volto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Avvia la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Impossibile aprire la webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Impossibile leggere un frame dalla webcam!")
        break

    # Converti il frame in scala di grigi per la rilevazione del volto
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Rileva i volti nel frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Per ogni volto rilevato, esegui la predizione
    for (x, y, w, h) in faces:
        # Estrai la regione d'interesse (ROI) contenente il volto
        roi_gray = gray[y:y+h, x:x+w]
        # Ridimensiona la ROI a 48x48 (dimensione attesa dal modello)
        roi = cv2.resize(roi_gray, (48, 48))
        # Normalizza l'immagine
        roi = roi.astype('float32') / 255.0
        # Aggiungi le dimensioni necessarie: (1, 48, 48, 1)
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        # Effettua la predizione
        preds = model.predict(roi)
        emotion_index = int(np.argmax(preds))
        emotion_text = emotion_labels[emotion_index]
        
        # Disegna un rettangolo attorno al volto e annota l'emozione
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)
    
    # Mostra il frame con le predizioni
    cv2.imshow("Riconoscimento Emozioni", frame)
    
    # Premi "q" per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la webcam e chiudi le finestre
cap.release()
cv2.destroyAllWindows()
