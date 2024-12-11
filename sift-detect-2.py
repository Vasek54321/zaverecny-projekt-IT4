import cv2
import numpy as np

# Konstanty
PIXEL_TO_METER = 0.05  # Předpokládaná velikost jednoho pixelu v metrech (kalibrace nutná)
FPS = 30  # Frekvence snímků videa (musí odpovídat vstupnímu videu)

# Načtení videa
video_path = "0001-1037.MKV"  # Nahraďte cestu k videu
cap = cv2.VideoCapture(video_path)

# Inicializace SIFT
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

prev_frame = None
prev_kp = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ukončeno.")
        break

    # Převod na odstíny šedi
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekce klíčových bodů a deskriptorů
    kp, des = sift.detectAndCompute(gray_frame, None)

    if prev_frame is not None and prev_kp is not None:
        # Spárování klíčových bodů mezi aktuálním a předchozím snímkem
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Výpočet posunů klíčových bodů
        distances = []
        for match in matches:
            pt1 = np.array(prev_kp[match.queryIdx].pt)
            pt2 = np.array(kp[match.trainIdx].pt)
            dist = np.linalg.norm(pt2 - pt1)
            distances.append(dist)

        if distances:
            avg_distance = np.mean(distances)
            # Výpočet rychlosti (posun v pixelech přepočtený na metry a čas)
            speed_m_per_s = (avg_distance * PIXEL_TO_METER) * FPS
            speed_kmh = speed_m_per_s * 3.6

            # Výpis rychlosti
            print(f"Aktuální rychlost: {speed_kmh:.2f} km/h")

            # Zobrazení na videu
            cv2.putText(frame, f"Speed: {speed_kmh:.2f} km/h", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Zobrazení videa
    cv2.imshow("Frame", frame)

    # Posun ke dalšímu snímku
    prev_frame = frame
    prev_kp, prev_des = kp, des

    # Ukončení klávesou 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()