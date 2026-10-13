
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
cv2.namedWindow("SignSpeak AI - Holistic Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "SignSpeak AI - Holistic Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        # Draw body / pose landmarks
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Draw facial landmarks
        if results.face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None
            )

        # Draw left hand
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # Draw right hand
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        face = "Yes" if results.face_landmarks else "No"
        body = "Yes" if results.pose_landmarks else "No"
        left = "Yes" if results.left_hand_landmarks else "No"
        right = "Yes" if results.right_hand_landmarks else "No"

        status = f"Face: {face} | Body: {body}"
        hands_status = f"Left hand: {left} | Right hand: {right}"

        cv2.putText(
            frame, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )
        cv2.putText(
            frame, hands_status, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        cv2.imshow("SignSpeak AI - Holistic Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
