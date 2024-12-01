import cv2
import numpy as np

def calculate_speed_with_visualization(video_path, real_world_scale, fps):
    # Načtení videa
    cap = cv2.VideoCapture(video_path)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Inicializace
    prev_frame = None
    prev_keypoints = None
    prev_descriptors = None
    frame_idx = 0
    speeds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Převod na šedotónový obrázek
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detekce a popis rysů
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if prev_frame is not None:
            # Porovnání rysů mezi snímky
            matches = bf.match(prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Vypočítání posunu
            displacements = []
            for match in matches:
                pt1 = prev_keypoints[match.queryIdx].pt
                pt2 = keypoints[match.trainIdx].pt
                displacements.append(np.linalg.norm(np.array(pt1) - np.array(pt2)))

                # Vizualizace spárovaných bodů
                pt1 = tuple(map(int, pt1))
                pt2 = tuple(map(int, pt2))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                cv2.circle(frame, pt1, 3, (255, 0, 0), -1)
                cv2.circle(frame, pt2, 3, (0, 0, 255), -1)

            if displacements:
                avg_displacement = np.mean(displacements)
                # Převod na reálnou vzdálenost
                real_displacement = avg_displacement * real_world_scale
                # Výpočet rychlosti v km/h
                speed_kmh = real_displacement * fps * 3.6
                speeds.append(speed_kmh)

                # Zobrazení rychlosti na snímku
                cv2.putText(
                    frame, f"Speed: {speed_kmh:.2f} km/h", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
                )

        # Aktualizace pro další snímek
        prev_frame = frame
        prev_keypoints = keypoints
        prev_descriptors = descriptors
        frame_idx += 1

        # Zobrazení aktuálního snímku
        cv2.imshow("Radar Visualization", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Stiskem 'q' ukončíte vizualizaci
            break

    cap.release()
    cv2.destroyAllWindows()
    return speeds

# Parametry
video_path = "Untitled.mp4"  # Cesta k videu
real_world_scale = 0.01  # Konverzní měřítko (např. 1 pixel = 0.01 metrů)
fps = 30  # Snímky za sekundu ve videu

speeds = calculate_speed_with_visualization(video_path, real_world_scale, fps)
print(f"Průměrná rychlost: {np.mean(speeds):.2f} km/h")