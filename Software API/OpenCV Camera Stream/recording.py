
'''
Live preview + key-controlled video recording for Jetson Nano with USB camera.

Controls:
- Press R (uppercase only) to start recording (saves to a new timestamped folder)
- Press S to stop recording
- Press q or ESC to quit
'''


import cv2          #camera access and video writing.
import os           #Used for path operations
import pathlib      #file path
import datetime     #Current timestamp
import time

# -------------------- Configuration -------------------- #
CAM_ID          = 0                # Camera index Usually /dev/video0
FRAME_SIZE      = (1280, 720)     # Width x Height
DEFAULT_FPS     = 20              # Default fallback FPS (Frame Per Second)
ROOT_SAVE_DIR   = pathlib.Path("/home/jetson/DMS/Recording/videos")
FOURCC          = cv2.VideoWriter_fourcc(*"mp4v")  # Output code
# -------------------------------------------------------- #

# Use V4L2 backend to avoid GStreamer issues
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)

# Set camera properties
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))   #Sets the camera to MJPG format 
cap.set(cv2.CAP_PROP_FRAME_WIDTH , FRAME_SIZE[0])               #Sets width, height, and FPS.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
cap.set(cv2.CAP_PROP_FPS         , DEFAULT_FPS)

# Get actual FPS from camera
actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps <= 1.0:
    print("Warning: Unable to detect camera FPS, falling back to default:", DEFAULT_FPS)
    actual_fps = DEFAULT_FPS
else:
    print("Detected camera FPS:", actual_fps)

# Verify camera is open, check camera access
if not cap.isOpened():
    print("Error: Cannot open USB camera.")
    exit()

recording = False
writer = None
last_frame_time = time.time()

print("Controls:  R=start recording  S=stop recording  q/ESC=quit")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Warning: Failed to grab frame")
        break

    cv2.imshow("Live USB Camera (R=start, S=stop, q=quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):  # Quit on 'q' or ESC
        break

    # Start recording
    if key == ord('R') and not recording:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = ROOT_SAVE_DIR / now
        folder.mkdir(parents=True, exist_ok=True)
        filepath = folder / f"capture_{now}.mp4"

        writer = cv2.VideoWriter(str(filepath), FOURCC, actual_fps, FRAME_SIZE)
        if not writer.isOpened():
            print("Error: Failed to open video writer.")
            writer = None
        else:
            recording = True
            last_frame_time = time.time()
            print(f"Recording started -> {filepath}")

    # Stop recording
    if key == ord('S') and recording:
        recording = False
        writer.release()
        writer = None
        print("Recording stopped.")

    # Save frame if recording with correct FPS pacing
    if recording and writer is not None:
        current_time = time.time()
        if current_time - last_frame_time >= (1.0 / actual_fps):
            writer.write(frame)
            last_frame_time = current_time

# Cleanup
if writer is not None:
    writer.release()
cap.release()
cv2.destroyAllWindows()
print("Program finished.")


