from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carica il modello e definisci le etichette delle emozioni
model_path = "models/emotion_model_improved.h5"
model = load_model(model_path)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Crea l'app FastAPI
app = FastAPI()

# Abilita CORS per permettere le richieste dal frontend (modifica allow_origins se necessario)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puoi restringere questo a "http://localhost:3000" se il tuo frontend gira in locale
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "API di Riconoscimento Emozioni"}

@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile = File(...)):
    # Leggi il file caricato
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Immagine non valida"}, status_code=400)

    # Preprocessing: converti in scala di grigi, ridimensiona e normalizza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (48, 48))
    roi = gray_resized.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)  # Aggiungi la dimensione batch
    roi = np.expand_dims(roi, axis=-1) # Aggiungi il canale (immagini in scala di grigi)

    # Effettua la predizione
    preds = model.predict(roi)
    emotion_index = int(np.argmax(preds))
    emotion_text = emotion_labels[emotion_index]
    confidence = float(np.max(preds))

    # Restituisci il risultato come JSON
    return {
        "emotion": emotion_text,
        "confidence": confidence,
        "probabilities": preds[0].tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
