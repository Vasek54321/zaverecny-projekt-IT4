import cv2
import numpy as np

def detect_objects(frame, net, output_layers, confidence_threshold=0.5):
    # Převod rámu na blob pro YOLO
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Inicializace výstupu
    boxes, confidences, class_ids = [], [], []
    
    # Zpracování výstupů YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:  # Filtrování podle prahu
                center_x, center_y, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS - odstranění redundantních boxů
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    result = []
    for i in indexes:
        i = i[0]
        result.append((boxes[i], class_ids[i], confidences[i]))
    return result


def calculate_speed_for_car(video_path, real_world_scale, fps):
    # Načtení YOLO modelu
    net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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

        # Detekce auta v aktuálním snímku
        detections = detect_objects(frame, net, output_layers)
        car_boxes = [box for box, class_id, conf in detections if classes[class_id] == "car"]

        if len(car_boxes) == 0:
            continue  # Pokud nebylo auto detekováno, přeskočíme snímek

        # Vybereme první detekované auto
        x, y, w, h = car_boxes[0]

        # Ořízneme oblast, kde se auto nachází
        car_roi = frame[y:y+h, x:x+w]
        gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)

        # Detekce a popis rysů
        keypoints, descriptors = sift.detectAndCompute(gray_car, None)

        if prev_frame is not None:
            matches = bf.match(prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Výpočet posunů
            displacements = []
            for match in matches:
                pt1 = prev_keypoints[match.queryIdx].pt
                pt2 = keypoints[match.trainIdx].pt
                displacements.append(np.linalg.norm(np.array(pt1) - np.array(pt2)))

            if displacements:
                avg_displacement = np.mean(displacements)
                real_displacement = avg_displacement * real_world_scale
                speed_kmh = real_displacement * fps * 3.6
                speeds.append(speed_kmh)

                # Zobrazení rychlosti na snímku
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Speed: {speed_kmh:.2f} km/h", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
                )

        # Aktualizace pro další snímek
        prev_frame = car_roi
        prev_keypoints = keypoints
        prev_descriptors = descriptors
        frame_idx += 1

        # Zobrazení aktuálního snímku
        cv2.imshow("Car Speed Measurement", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Stiskem 'q' ukončíte vizualizaci
            break

    cap.release()
    cv2.destroyAllWindows()
    return speeds


# Parametry
video_path = "Untitled.mp4"  # Cesta k videu
real_world_scale = 0.01  # Konverzní měřítko (např. 1 pixel = 0.01 metrů)
fps = 25  # Snímky za sekundu ve videu

speeds = calculate_speed_for_car(video_path, real_world_scale, fps)
print(f"Průměrná rychlost: {np.mean(speeds):.2f} km/h")