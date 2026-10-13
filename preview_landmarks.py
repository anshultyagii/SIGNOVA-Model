
import cv2
import mediapipe as mp

video_path = r"F:\SignSpeakAI\dataset\raw\1_One.mp4"

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

cap = cv2.VideoCapture(video_path)

cv2.namedWindow("SignSpeak AI - Landmark Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SignSpeak AI - Landmark Preview", 1000, 650)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (960, 540))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )

        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        cv2.putText(
            frame,
            f"Left: {'YES' if results.left_hand_landmarks else 'NO'} | "
            f"Right: {'YES' if results.right_hand_landmarks else 'NO'}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("SignSpeak AI - Landmark Preview", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
