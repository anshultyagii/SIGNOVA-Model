
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

MODEL_PATH = r"F:\SignSpeakAI\sign_model.pkl"
model = joblib.load(MODEL_PATH)

mp_holistic = mp.solutions.holistic
SEQUENCE_LENGTH = 30
BUFFER_SIZE = 30

sequence = deque(maxlen=BUFFER_SIZE)
prediction_text = "Show a sign"
confidence_text = ""

def get_coords(obj, count):
    if obj:
        return np.array(
            [[p.x, p.y, p.z] for p in obj.landmark],
            dtype=np.float32
        ).flatten()
    return np.zeros(count * 3, dtype=np.float32)

def extract_features(result):
    pose = get_coords(result.pose_landmarks, 33)
    left_hand = get_coords(result.left_hand_landmarks, 21)
    right_hand = get_coords(result.right_hand_landmarks, 21)
    return np.concatenate([pose, left_hand, right_hand])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam could not be opened.")
    raise SystemExit

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    refine_face_landmarks=False
) as holistic:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        
        # Draw face, pose, and hand landmarks on webcam
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        if result.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.face_landmarks,
                mp.solutions.holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )


        features = extract_features(result)
        sequence.append(features)

        if len(sequence) == SEQUENCE_LENGTH:
            # Same 30-frame input format used during training
            sample = np.asarray(sequence, dtype=np.float32).flatten()
            probabilities = model.predict_proba(sample.reshape(1, -1))[0]
            best_index = np.argmax(probabilities)
            print({
    word: round(float(prob), 3)
    for word, prob in zip(model.classes_, probabilities)
})
            confidence = probabilities[best_index]
            predicted_word = model.classes_[best_index]

            if confidence >= 0.0:
                prediction_text = str(predicted_word)
                confidence_text = f"Confidence: {confidence:.2f}"
            else:
                prediction_text = "Uncertain"
                confidence_text = f"Confidence: {confidence:.2f}"

        cv2.putText(
            frame, prediction_text, (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame, confidence_text, (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.putText(
            frame, "Press Q to quit", (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.imshow("SignSpeak AI - Live Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
