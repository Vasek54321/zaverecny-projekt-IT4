import cv2
import numpy as np

# Načtení předtrénovaného modelu a konfiguračního souboru
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Načtení videa (případně můžete použít 0 pro zachytávání z kamery)
video = cv2.VideoCapture('traffic.mp4')

# Seznam tříd, které dokáže model detekovat (class 7 = auto)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Zpracování videa po snímcích
while True:
    # Načtení snímku
    ret, frame = video.read()
    
    # Pokud se snímek nepodařilo načíst (konec videa), ukončete smyčku
    if not ret:
        break

    # Získání rozměrů snímku
    (h, w) = frame.shape[:2]

    # Převod snímku na blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Předání blobu do sítě
    net.setInput(blob)
    detections = net.forward()

    # Detekce objektů na snímku
    for i in range(detections.shape[2]):
        # Získání skóre (jistota detekce)
        confidence = detections[0, 0, i, 2]

        # Filtrace detekcí s nízkou jistotou
        if confidence > 0.4:  # Můžete nastavit prahovou hodnotu dle potřeby
            idx = int(detections[0, 0, i, 1])

            # Pokud je detekovaný objekt auto (index 7)
            if CLASSES[idx] == "car":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ohraničení auta bílým obdélníkem
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)

    # Zobrazení snímku s detekcí aut
    cv2.imshow("Detected Cars", frame)

    # Zastavení smyčky, pokud uživatel stiskne klávesu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvolnění zdrojů
video.release()
cv2.destroyAllWindows()