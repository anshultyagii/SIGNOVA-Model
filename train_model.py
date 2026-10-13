
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_DIR = Path(r"F:\SignSpeakAI\dataset\recorded_processed")
WORDS = ["Drink", "Eat", "Help", "Water", "I_Me"]
SEQUENCE_LENGTH = 30

X, y = [], []


def make_sequence(features):
    # Keep pose (99 values) + left and right hands (126 values)
    pose = features[:, :99]
    hands = features[:, 99 + 468 * 3:]
    features = np.concatenate([pose, hands], axis=1)

    # Resample each video to 30 frames
    indices = np.linspace(
        0, len(features) - 1, SEQUENCE_LENGTH
    ).astype(int)

    return features[indices].flatten()


for word in WORDS:
    files = sorted(DATA_DIR.glob(f"{word}_sample_*_landmarks.npz"))

    for file in files:
        data = np.load(file)
        features = data["features"]

        if len(features) == 0:
            continue

        X.append(make_sequence(features))
        y.append(word)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

print("Total samples:", len(X))
print("Feature shape:", X.shape)
print("Class counts:", {word: int(np.sum(y == word)) for word in WORDS})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced", probability=True)
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Train final model on all 50 samples
model.fit(X, y)

MODEL_PATH = Path(r"F:\SignSpeakAI\sign_model.pkl")
joblib.dump(model, MODEL_PATH)
print("\nSaved model:", MODEL_PATH)

