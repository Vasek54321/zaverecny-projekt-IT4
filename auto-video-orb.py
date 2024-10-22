import cv2

# Načtení videa
cap = cv2.VideoCapture('video-dalnice.mp4')  # Nahraď 'video.mp4' vlastním videem

# Inicializace detektoru pohybu (odčítání pozadí)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

# Načtení kaskádového klasifikátoru pro detekci aut
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Převod rámce na šedotónový obraz
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplikace detekce pohybu (odčítání pozadí)
    fgmask = fgbg.apply(gray)

    # Zlepšení masky: odstranění šumu pomocí dilatace a erozi
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, None)

    # Najdi kontury (obrysy pohybujících se objektů)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Procházení přes všechny detekované kontury
    for contour in contours:
        # Ignoruj malé pohyby (šum)
        if cv2.contourArea(contour) < 500:
            continue

        # Najdi ohraničující obdélník pohybujícího se objektu
        x, y, w, h = cv2.boundingRect(contour)

        # Vyber region zájmu (ROI) v oblasti pohybu
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detekce aut uvnitř pohybujícího se objektu pomocí Haar Cascade
        cars = car_cascade.detectMultiScale(roi_gray, 1.1, 1)

        # Pro každé detekované auto vykresli obdélník
        for (cx, cy, cw, ch) in cars:
            cv2.rectangle(frame, (x+cx, y+cy), (x+cx+cw, y+cy+ch), (255, 255, 255), 2)

    # Zobrazení aktuálního rámce
    cv2.imshow('Detekce pohybujících se aut', frame)

    # Konec při stisknutí klávesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvolnění videa a zavření všech oken
cap.release()
cv2.destroyAllWindows()