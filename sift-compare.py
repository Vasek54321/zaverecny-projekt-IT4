import cv2

# Inicializace SIFT detektoru
sift = cv2.SIFT_create()

# Otevření webové kamery
cap = cv2.VideoCapture('Untitled.mp4')  # Pokud chcete použít video soubor, nahraďte '0' cestou k souboru

if not cap.isOpened():
    print("Chyba: Nelze otevřít kameru.")
    exit()

while True:
    # Čtení snímku z kamery
    ret, frame = cap.read()
    if not ret:
        print("Chyba: Nelze načíst snímek z kamery.")
        break

    # Převod snímku na odstíny šedi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekce klíčových bodů
    kp = sift.detect(gray, None)

    # Vykreslení klíčových bodů na původní barevný snímek
    frame_with_keypoints = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Zobrazení snímku s klíčovými body
    cv2.imshow('SIFT Keypoints', frame_with_keypoints)

    # Ukončení programu při stisku klávesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvolnění zdrojů
cap.release()
cv2.destroyAllWindows()