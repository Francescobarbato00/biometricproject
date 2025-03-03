import cv2
import os

image_path = '../test.jpeg'
print("Cartella corrente:", os.getcwd())
print("Percorso immagine:", image_path)
print("Esiste?", os.path.exists(image_path))

img = cv2.imread(image_path)
if img is None:
    print("ERRORE: immagine non trovata o non decodificabile!")
else:
    print("OK: immagine caricata, shape:", img.shape)
