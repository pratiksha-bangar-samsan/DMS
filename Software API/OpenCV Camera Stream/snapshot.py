import cv2
import os
import pathlib
import datetime

# -------------------- Configuration -------------------- #
CAM_ID = 0
FRAME_SIZE = (1280, 720)
SNAPSHOT_SAVE_DIR = pathlib.Path("/home/jetson/DMS/Recording/snapshots")
# -------------------------------------------------------- #

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)

# Set camera properties
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH , FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

if not cap.isOpened():
    print("Error: Cannot open USB camera.")
    exit()

print("Controls:  S = snapshot   q / ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to grab frame")
        break

    cv2.imshow("Live USB Camera (S=snapshot, q=quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):  # Quit
        break

    if key == ord('S'):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = SNAPSHOT_SAVE_DIR / now
        folder.mkdir(parents=True, exist_ok=True)
        filepath = folder / f"snapshot_{now}.jpg"
        cv2.imwrite(str(filepath), frame)
        print(f"Snapshot saved to {filepath}")

cap.release()
cv2.destroyAllWindows()
print("Snapshot program finished.")
