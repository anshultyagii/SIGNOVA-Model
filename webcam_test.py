import cv2

print("Starting webcam test...", flush=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.", flush=True)
else:
    print("Webcam started! Press Q to quit.", flush=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.", flush=True)
            break

        cv2.imshow("SignSpeak AI - Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
