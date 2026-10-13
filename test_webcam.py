
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("Webcam opened. Press Q to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break

        cv2.imshow("SignSpeakAI - Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
