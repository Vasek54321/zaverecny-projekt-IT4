import cv2
import numpy as np

# Načtení předtrénovaného modelu a konfiguračního souboru
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Načtení obrázku
image = cv2.imread('auto4.png')
(h, w) = image.shape[:2]

# Převod obrázku na blob (vstup pro neuronovou síť)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# Předání blobu do sítě
net.setInput(blob)
detections = net.forward()

# Seznam tříd, které dokáže model detekovat (class 7 = auto)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Detekce objektů
for i in range(detections.shape[2]):
    # Získání skóre (jistota detekce)
    confidence = detections[0, 0, i, 2]
    
    # Filtrace detekcí s nízkou jistotou
    if confidence > 0.4:  # Můžete nastavit prahovou hodnotu dle potřeby
        idx = int(detections[0, 0, i, 1])

        # Pokud detekovaný objekt je auto (index 7)
        if CLASSES[idx] == "car":
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ohraničení auta bílým obdélníkem
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)

# Zobrazení výsledku
cv2.imshow("Detected Cars", image)
cv2.waitKey(0)
cv2.destroyAllWindows()