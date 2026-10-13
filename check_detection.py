
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam open nahi hui")
    raise SystemExit

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        # Draw detected landmarks
        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        if result.left_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, result.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if result.right_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        left = result.left_hand_landmarks is not None
        right = result.right_hand_landmarks is not None
        pose = result.pose_landmarks is not None

        cv2.putText(
            frame,
            f"Left hand: {left} | Right hand: {right} | Pose: {pose}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

        cv2.imshow("MediaPipe Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
