import cv2
import easyocr
import numpy as np
import time

# Načtení videa
cap = cv2.VideoCapture('video.mp4')

# Detekce pohybu pomocí modelu pozadí MOG2.
fgbg = cv2.createBackgroundSubtractorMOG2()

# Inicializace proměnných
prev_time = 0
prev_position = None
speed = 0
max_speed = 50
center_x = 0
center_y = 0
start_time = None
elapsed_time = None
license_plate = None

# Nastavení pozic čar v pixelech a vzdálenosti mezi nimi v metrech
line1_y = 460
line2_y = 860
lines_distance = 25

# Inicializace souboru (vymaže obsah, pokud soubor již existuje)
with open("rychlosti.txt", "w") as file:
    file.write("Rychlosti:\n")

# Funkce pro přidání řádku do souboru
def write_line(text):
    with open("rychlosti.txt", "a") as file:
        file.write(text)
        file.write("\n")

# Funkce pro detekci SPZ
def detect_license(frame):
    reader = easyocr.Reader(['en'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 2 < aspect_ratio < 6:
                license_plate_region = frame[y:y+h, x:x+w]
                result = reader.readtext(license_plate_region)
                if result:
                    return result[0][1] 
    return None
    

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Funkce na najdení největší kontury
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:  # Odfiltrování šumu
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            current_position = (center_x, center_y)
            current_time = time.time()

            # Zaznačení středu detekovaného auta
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Zjištění jestli se auto dotýká první čáry
            if center_y >= line1_y and start_time is None:
                start_time = current_time
                
            # Zjištění jestli se auto dotýká druhé čáry
            if center_y >= line2_y and start_time is not None:
                elapsed_time = current_time - start_time
                start_time = None  # Resetování času pro další měření

            if elapsed_time is not None and elapsed_time > 0:
                # Výpočet rychlosti
                speed = (lines_distance / elapsed_time) * 3.6 
                license_plate = detect_license(frame)
                if speed < max_speed:
                    write_line(f'SPZ: {license_plate}, Rychlost: {int(speed)} km/h, Cas: {elapsed_time:.2f}')
                else:
                    write_line(f'SPZ: {license_plate}, Rychlost: {int(speed)} km/h, Cas: {elapsed_time:.2f} >>> Prekrocena povolena rychlost o {int(speed)-max_speed} km/h')

 
    # Vykreslení čar k detekci rychlosti
    cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), (0, 0, 255), 2)
    cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), (0, 0, 255), 2)

    cv2.imshow('Car Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()