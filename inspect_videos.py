
from pathlib import Path
import cv2

video_folder = Path(r"F:\SignSpeakAI\dataset\raw")

for video_path in sorted(video_folder.glob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: {video_path.name} could not open")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0

    print(
        f"{video_path.name:30} | "
        f"{width}x{height} | "
        f"{fps:.2f} FPS | "
        f"{frames} frames | "
        f"{duration:.2f} sec"
    )

    cap.release()
