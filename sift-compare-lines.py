import cv2
import numpy as np

# Inicializace SIFT detektoru
sift = cv2.SIFT_create()

# Otevření webové kamery
cap = cv2.VideoCapture('0001-1037.mkv')  # Pokud chcete použít video soubor, nahraďte '0' cestou k souboru

if not cap.isOpened():
    print("Chyba: Nelze otevřít kameru.")
    exit()

# Proměnné pro uchování předchozího snímku a klíčových bodů
prev_kp = None
prev_des = None
prev_frame = None

while True:
    # Čtení snímku z kamery
    ret, frame = cap.read()
    if not ret:
        print("Chyba: Nelze načíst snímek z kamery.")
        break

    # Převod snímku na odstíny šedi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekce klíčových bodů a výpočet jejich deskriptorů
    kp, des = sift.detectAndCompute(gray, None)

    if prev_frame is not None and prev_kp is not None and prev_des is not None:
        # Použití BFMatcher pro nalezení odpovídajících bodů mezi snímky
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(prev_des, des, k=2)

        # Filtrace dobrých odpovídajících bodů pomocí poměrového testu
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Vytvoření kopie aktuálního snímku pro vykreslení
        frame_with_lines = frame.copy()

        # Vykreslení propojení mezi odpovídajícími body přímo do aktuálního snímku
        for match in good_matches:
            pt1 = tuple(map(int, prev_kp[match.queryIdx].pt))
            pt2 = tuple(map(int, kp[match.trainIdx].pt))
            cv2.line(frame_with_lines, pt1, pt2, (0, 255, 0), 2)

        # Zobrazení snímku s propojenými body
        cv2.imshow('SIFT Matches in One Frame', frame_with_lines)

    # Uchování aktuálního snímku a jeho klíčových bodů pro další iteraci
    prev_frame = frame
    prev_kp = kp
    prev_des = des

    # Zobrazení aktuálního snímku s klíčovými body
    #frame_with_keypoints = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('SIFT Keypoints', frame_with_keypoints)

    # Ukončení programu při stisku klávesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvolnění zdrojů
cap.release()
cv2.destroyAllWindows()
