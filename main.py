from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import re
import winsound  # Only works on Windows for sound alert

# -----------------------------
# Plate validation function
# -----------------------------
def validate_indian_plate(plate_text):
    pattern = r'^[A-Z]{2}\s\d{2}\s[A-Z]{1,2}\s\d{4}$'
    return bool(re.match(pattern, plate_text))

# -----------------------------
# Initialize tracker and results
# -----------------------------
mot_tracker = Sort()
results = {}

# -----------------------------
# Load YOLO models
# -----------------------------
coco_model = YOLO('yolov8n.pt')  # Vehicle detection
license_plate_detector = YOLO('license_plate_detector.pt')  # License plate detection

# -----------------------------
# Load video
# -----------------------------
cap = cv2.VideoCapture('./videos/sample.mp4')
vehicles = [2, 3, 5, 7]  # car, motorbike, bus, truck

cv2.namedWindow("ANPR Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ANPR Live", 960, 540)  # Resize window

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # Vehicle detection
    detections = coco_model(frame)[0]
    detections_ = []
    for det in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # SORT tracking
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Draw vehicle bounding boxes + IDs
    for i, det in enumerate(detections_):
        x1, y1, x2, y2, score = det
        if i < len(track_ids):
            car_id = int(track_ids[i][4])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{car_id}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # License plate detection
    lps = license_plate_detector(frame)[0]
    for lp in lps.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)

        if car_id != -1:
            lp_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)
            text, text_score = read_license_plate(lp_thresh)

            if text is not None:
                # Save results
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': text,
                        'bbox_score': score,
                        'text_score': text_score
                    }
                }

                # Validate plate
                if validate_indian_plate(text):
                    color = (0, 255, 0)  # Green for valid
                    label = "VALID"
                else:
                    color = (0, 0, 255)  # Red for invalid
                    # Optional sound alert
                    label = "INVALID"
                    winsound.Beep(1000, 200)

                # Draw license plate bbox + text
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'{text}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Draw validation label below the plate
                cv2.putText(frame, f'{label}', (int(x1), int(y2)+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display real-time frame
    cv2.imshow('ANPR Live', frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save CSV
write_csv(results, './test.csv')
print("Done! Results saved to test.csv")
