import cv2

def test_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera 0 failed to open. Trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("No camera found.")
            return

    # Check default resolution
    dw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    dh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Default resolution: {int(dw)}x{int(dh)} (Aspect ratio: {dw/dh:.2f})")

    # Try setting HD 720p (1280x720 16:9 widescreen)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    hw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    hh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"After requesting 1280x720: {int(hw)}x{int(hh)} (Aspect ratio: {hw/hh:.2f})")

    ret, frame = cap.read()
    if ret:
        print(f"Captured frame shape: {frame.shape}")
    cap.release()

if __name__ == "__main__":
    test_camera()
