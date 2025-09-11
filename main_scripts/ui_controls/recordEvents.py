# ==================== Main Script ======================
import cv2
import time
import numpy as np
import os
import argparse
import mediapipe as mp
from scipy.spatial import distance as dist
from playsound import playsound
import threading
import configparser
import logging
from datetime import datetime
import csv
import subprocess
import signal
import sys
import shutil

# ==================== Configuration and Paths ======================
# Ensure these directories are accessible and writable.
# VIDEO_LOG_DIR is for the continuous recording on the local system.
# USB_MOUNT_POINT and LOG_DIR_USB are for event logs and cropped videos on a USB drive.
VIDEO_LOG_DIR = "/media/jetson/CCCOMA_X64FRE_EN-US_DV9/loop_recording"
USB_MOUNT_POINT = "/media/jetson/CCCOMA_X64FRE_EN-US_DV9" # Change this to your USB drive's mount point
LOG_DIR_USB = os.path.join(USB_MOUNT_POINT, "/media/jetson/CCCOMA_X64FRE_EN-US_DV9/DMS/cropped_events")
CSV_FILE_PATH = os.path.join(USB_MOUNT_POINT, "event_log.csv")

# Ensure required directories exist
try:
    os.makedirs(VIDEO_LOG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR_USB, exist_ok=True)
except OSError as e:
    print(f"Error creating directories: {e}. Check permissions and USB drive.")
    exit()

# Logging configuration
logging.basicConfig(filename='dmm_monitor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for video recording
video_writer = None
recording_active = False
video_filename = None
event_data_queue = []
event_lock = threading.Lock()

# ==================== Functions to be used in the Class ======================
def eye_aspect_ratio(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) as a measure of eye openness.
    """
    p2_p6 = dist.euclidean(np.array([eye_landmarks[1].x, eye_landmarks[1].y]), np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    p3_p5 = dist.euclidean(np.array([eye_landmarks[2].x, eye_landmarks[2].y]), np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    p1_p4 = dist.euclidean(np.array([eye_landmarks[0].x, eye_landmarks[0].y]), np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def mouth_aspect_ratio(landmarks, MOUTH_INNER_VERTICAL, MOUTH_OUTER_HORIZONTAL):
    """
    Calculates the Mouth Aspect Ratio (MAR) to detect yawning.
    """
    A = dist.euclidean(np.array([landmarks[MOUTH_INNER_VERTICAL[0]].x, landmarks[MOUTH_INNER_VERTICAL[0]].y]),
                       np.array([landmarks[MOUTH_INNER_VERTICAL[1]].x, landmarks[MOUTH_INNER_VERTICAL[1]].y]))
    B = dist.euclidean(np.array([landmarks[MOUTH_OUTER_HORIZONTAL[0]].x, landmarks[MOUTH_OUTER_HORIZONTAL[0]].y]),
                       np.array([landmarks[MOUTH_OUTER_HORIZONTAL[1]].x, landmarks[MOUTH_OUTER_HORIZONTAL[1]].y]))
    if B == 0:
        return 0
    mar = A / B
    return mar

# ==================== EdgetensorDMMMonitor Class ======================
class EdgetensorDMMMonitor:
    def __init__(self, roi_position, config_file='config.ini'):
        self.config = self.load_config(config_file)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Retrieve thresholds and timers from config file
        self.EAR_THRESHOLD = self.config.getfloat('THRESHOLDS', 'ear_threshold')
        self.YAWN_THRESHOLD = self.config.getfloat('THRESHOLDS', 'yawn_threshold')
        self.BRIGHTNESS_THRESHOLD = self.config.getfloat('THRESHOLDS', 'brightness_threshold')
        self.COLOR_VAR_THRESHOLD = self.config.getfloat('THRESHOLDS', 'color_var_threshold')
        self.LEFT_THRESHOLD = self.config.getfloat('THRESHOLDS', 'left_threshold')
        self.RIGHT_THRESHOLD = self.config.getfloat('THRESHOLDS', 'right_threshold')
        self.UP_THRESHOLD = self.config.getfloat('THRESHOLDS', 'up_threshold')
        self.DOWN_THRESHOLD = self.config.getfloat('THRESHOLDS', 'down_threshold')

        self.YAWN_TIME_THRESHOLD = self.config.getfloat('TIMERS', 'yawn_time')
        self.FATIGUE_TIME_THRESHOLD = self.config.getfloat('TIMERS', 'fatigue_time')
        self.DISTRACTION_TIME_THRESHOLD = self.config.getfloat('TIMERS', 'distraction_time')
        self.FACE_OBSTRUCTION_TIME = self.config.getfloat('TIMERS', 'face_obstruction_time')
        self.CAMERA_OBSTRUCTION_TIME = self.config.getfloat('TIMERS', 'camera_obstruction_time')
        self.SMOKING_TIME = self.config.getfloat('TIMERS', 'smoking_time')
        self.PHONE_USE_TIME = self.config.getfloat('TIMERS', 'phone_use_time')
        self.EATING_DRINKING_TIME = self.config.getfloat('TIMERS', 'eating_drinking_time')
        self.ALERT_DELAY_SECONDS = self.config.getfloat('TIMERS', 'alert_delay_seconds')

        # Define landmark indices for face parts
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH_INNER_VERTICAL = [13, 14]
        self.MOUTH_OUTER_HORIZONTAL = [61, 291]
        self.NOSE_LANDMARKS = [4, 6, 195]

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.roi_position = roi_position
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = self.get_roi_coordinates(roi_position)

        self.audio_path = "/home/jetson/obstruction_gpu/audio_alert" # Change to the correct path for your audio files
        
        self.voice_alerts = {
            'yawning': 'yawning.mp3',
            'distraction': 'distraction.mp3',
            'fatigue': 'fatigue.mp3',
            'camera_obstruction': 'obstruction.mp3',
            'face_obstruction': 'face_obstruction.mp3',
            'smoking': 'smoking.mp3',
            'phone_uses': 'phone_uses.mp3',
            'eating_drinking': 'eating_drinking.mp3',
            'seatbelt_off': 'seatbelt_off.mp3',
        }

        # Event state tracking dictionary
        self.events = {
            'smoking': {'active': False, 'timer': 0.0, 'threshold': self.SMOKING_TIME, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'yawning': {'active': False, 'timer': 0.0, 'threshold': self.YAWN_TIME_THRESHOLD, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'eating_drinking': {'active': False, 'timer': 0.0, 'threshold': self.EATING_DRINKING_TIME, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'phone_uses': {'active': False, 'timer': 0.0, 'threshold': self.PHONE_USE_TIME, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'wearing_mask': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'seatbelt_off': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'wearing_seatbelt': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'fatigue': {'active': False, 'timer': 0.0, 'threshold': self.FATIGUE_TIME_THRESHOLD, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'distraction': {'active': False, 'timer': 0.0, 'threshold': self.DISTRACTION_TIME_THRESHOLD, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'camera_obstruction': {'active': False, 'timer': 0.0, 'threshold': self.CAMERA_OBSTRUCTION_TIME, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'face_obstruction': {'active': False, 'timer': 0.0, 'threshold': self.FACE_OBSTRUCTION_TIME, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'left_eye_closed': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
            'right_eye_closed': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0, 'start_time': None, 'end_time': None},
        }

        # Duration counters for events
        self.yawning_duration = 0.0
        self.fatigue_duration = 0.0
        self.distraction_duration = 0.0
        self.camera_obstruction_duration = 0.0
        self.face_obstruction_duration = 0.0
        self.left_eye_closed_duration = 0.0
        self.right_eye_closed_duration = 0.0
        
        self.gaze_text = "No Face"
        self.metrics = {'fps': 0, 'frame_processing_time': 0, 'faces_detected': 0}
        self.head_pose_yaw = 0.0
        self.head_pose_pitch = 0.0
        self.head_pose_roll = 0.0
        self.last_time = time.time()
        self.face_not_detected = False
        
        # CSV logging setup
        self.csv_file = None
        self.csv_writer = None
        self.setup_csv_logging()

    def load_config(self, config_file):
        """Loads configuration from a .ini file."""
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            logging.error(f"Config file '{config_file}' not found.")
            exit(1)
        config.read(config_file)
        return config

    def get_roi_coordinates(self, roi_position):
        """Gets Region of Interest (ROI) coordinates from the config file."""
        try:
            section_name = f'ROI_{roi_position}'
            if not self.config.has_section(section_name):
                section_name = f"ROI_{self.config.get('DEFAULT', 'active_roi')}"
            x1 = self.config.getint(section_name, 'x1')
            y1 = self.config.getint(section_name, 'y1')
            x2 = self.config.getint(section_name, 'x2')
            y2 = self.config.getint(section_name, 'y2')
            return x1, y1, x2, y2
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logging.error(f"Error reading ROI configuration: {e}. Using hardcoded default.")
            return 200, 150, 600, 600

    def setup_csv_logging(self):
        """Initializes the CSV file for event logging."""
        try:
            is_new_file = not os.path.exists(CSV_FILE_PATH)
            self.csv_file = open(CSV_FILE_PATH, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            if is_new_file:
                self.csv_writer.writerow(['Timestamp', 'Event', 'Event_Start_Time', 'Event_End_Time'])
            logging.info(f"CSV logging to {CSV_FILE_PATH} initialized.")
        except Exception as e:
            logging.error(f"Failed to set up CSV logging to USB drive: {e}")
            self.csv_file = None
            self.csv_writer = None

    def play_alert(self, event_name):
        """Plays an audio alert asynchronously."""
        def play_sound_async():
            if event_name in self.voice_alerts:
                audio_file = os.path.join(self.audio_path, self.voice_alerts[event_name])
                if os.path.exists(audio_file):
                    try:
                        playsound(audio_file)
                    except Exception as e:
                        logging.error(f"Error playing sound for {event_name}: {e}")
                else:
                    logging.warning(f"Audio file not found for event '{event_name}' at {audio_file}")
            else:
                logging.warning(f"No voice alert defined for event '{event_name}'.")
        thread = threading.Thread(target=play_sound_async)
        thread.daemon = True
        thread.start()

    def log_event_to_csv(self, event_name, start_time, end_time, video_file):
        """Logs event data to the CSV file."""
        if self.csv_writer:
            try:
                self.csv_writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    event_name,
                    start_time,
                    end_time,
                    video_file
                ])
                self.csv_file.flush() # Ensure data is written immediately
                logging.info(f"Logged event '{event_name}' to CSV.")
            except Exception as e:
                logging.error(f"Failed to write to CSV file: {e}")

    def detect_eye_closure(self, landmarks):
        """Detects if the eyes are closed based on EAR."""
        left_eye_landmarks = [landmarks[i] for i in self.LEFT_EYE]
        right_eye_landmarks = [landmarks[i] for i in self.RIGHT_EYE]
        left_ear = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)
        left_eye_closed = left_ear < self.EAR_THRESHOLD
        right_eye_closed = right_ear < self.EAR_THRESHOLD
        return left_eye_closed, right_eye_closed

    def detect_yawn(self, landmarks):
        """Detects a yawn based on MAR."""
        mar = mouth_aspect_ratio(landmarks, self.MOUTH_INNER_VERTICAL, self.MOUTH_OUTER_HORIZONTAL)
        yawn_detected = mar > self.YAWN_THRESHOLD
        yawn_confidence = mar / (self.YAWN_THRESHOLD * 1.5) if yawn_detected else 0
        yawn_confidence = min(1.0, yawn_confidence)
        return yawn_detected, yawn_confidence

    def detect_head_pose(self, landmarks):
        """
        Estimates head pose (yaw, pitch, roll) to detect distraction.
        """
        image_points = np.array([
            (landmarks[33].x * 1280, landmarks[33].y * 720),
            (landmarks[263].x * 1280, landmarks[263].y * 720),
            (landmarks[1].x * 1280, landmarks[1].y * 720),
            (landmarks[61].x * 1280, landmarks[61].y * 720),
            (landmarks[291].x * 1280, landmarks[291].y * 720),
            (landmarks[199].x * 1280, landmarks[199].y * 720)
        ], dtype="double")
        model_points = np.array([
            (0.0, 0.0, 0.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0), (0.0, 0.0, -330.0)
        ])
        focal_length = 1280
        center = (1280 / 2, 720 / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            else:
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                roll = 0.0
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            roll_deg = np.degrees(roll)

            distraction_detected = False
            direction = None
            if yaw_deg > self.RIGHT_THRESHOLD:
                distraction_detected = True
                direction = 'right'
            elif yaw_deg < -self.LEFT_THRESHOLD:
                distraction_detected = True
                direction = 'left'
            elif pitch_deg > self.UP_THRESHOLD:
                distraction_detected = True
                direction = 'up'
            elif pitch_deg < -self.DOWN_THRESHOLD:
                distraction_detected = True
                direction = 'down'
            return distraction_detected, direction, yaw_deg, pitch_deg, roll_deg
        return False, None, 0.0, 0.0, 0.0

    def check_camera_obstruction(self, frame):
        """Checks for camera obstruction by analyzing frame brightness and color variance."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_frame)
        color_variance = np.std(frame)
        is_dark_obstructed = mean_intensity < self.BRIGHTNESS_THRESHOLD
        is_uniform_obstructed = color_variance < self.COLOR_VAR_THRESHOLD
        return is_dark_obstructed or is_uniform_obstructed

    def update_event_states(self, yawn_detected, distraction_detected, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected):
        """Updates the timers and states for all monitored events."""
        delta_time = 1 / self.metrics['fps'] if self.metrics['fps'] > 0 else 0
        
        # Update event durations
        if faces_detected == 0 and not is_camera_obstructed:
            self.face_obstruction_duration += delta_time
        else:
            self.face_obstruction_duration = 0
            
        if is_camera_obstructed:
            self.camera_obstruction_duration += delta_time
        else:
            self.camera_obstruction_duration = 0
            
        if yawn_detected:
            self.yawning_duration += delta_time
        else:
            self.yawning_duration = 0
            
        if left_eye_closed and right_eye_closed:
            self.fatigue_duration += delta_time
        else:
            self.fatigue_duration = 0

        if distraction_detected:
            self.distraction_duration += delta_time
        else:
            self.distraction_duration = 0
            
        # Check for event activation and logging
        for event_name, event_data in self.events.items():
            timer_duration = getattr(self, f"{event_name}_duration", 0)
            threshold = event_data['threshold']

            # Event is currently happening and exceeds threshold
            if threshold is not None and timer_duration >= threshold:
                if not event_data['active']:
                    event_data['active'] = True
                    event_data['start_time'] = time.time() - timer_duration # Get the precise start time
                    self.play_alert(event_name)
                    logging.info(f"Event '{event_name}' started at time {event_data['start_time']}.")
            else:
                # Event was active and now has stopped
                if event_data['active']:
                    event_data['active'] = False
                    event_data['end_time'] = time.time()
                    # Log event to CSV with video filename
                    if video_filename:
                        self.log_event_to_csv(
                            event_name=event_name,
                            start_time=event_data['start_time'],
                            end_time=event_data['end_time'],
                            video_file=video_filename
                        )
                    logging.info(f"Event '{event_name}' ended at time {event_data['end_time']}.")
                    event_data['start_time'] = None
                    event_data['end_time'] = None
            
    def run(self):
        """The main loop for video capture and monitoring."""
        global video_writer, recording_active, video_filename

        print("Starting Edgetensor DMM Monitor...")
        logging.info("Starting Edgetensor DMM Monitor.")

        # Start the video recording loop
        if not recording_active:
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec for .mp4 files
            video_filename = f"dmm_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            video_path = os.path.join(VIDEO_LOG_DIR, video_filename)
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1280, 720))
            recording_active = True
            logging.info(f"Started video recording to {video_path}.")
        
        while self.cap.isOpened():
            frame_start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Write frame to video file
            if recording_active and video_writer:
                video_writer.write(frame)

            video_resized = cv2.resize(frame, (640, 480))
            is_camera_obstructed = self.check_camera_obstruction(video_resized)
            
            yawn_detected = False
            distraction_detected_from_head_pose = False
            head_direction = None
            left_eye_closed = False
            right_eye_closed = False
            faces_detected = 0
            self.head_pose_yaw = 0.0
            self.head_pose_pitch = 0.0
            self.head_pose_roll = 0.0
            
            if not is_camera_obstructed:
                rgb_frame = cv2.cvtColor(video_resized, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    faces_detected = len(results.multi_face_landmarks)
                    for face_landmarks in results.multi_face_landmarks:
                        left_eye_closed, right_eye_closed = self.detect_eye_closure(face_landmarks.landmark)
                        yawn_detected, _ = self.detect_yawn(face_landmarks.landmark)
                        distraction_detected_from_head_pose, head_direction, self.head_pose_yaw, self.head_pose_pitch, self.head_pose_roll = self.detect_head_pose(face_landmarks.landmark)

            self.update_event_states(yawn_detected, distraction_detected_from_head_pose, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected)

            current_time = time.time()
            self.metrics['frame_processing_time'] = current_time - frame_start_time
            if self.metrics['frame_processing_time'] > 0:
                self.metrics['fps'] = 1.0 / self.metrics['frame_processing_time']

        self.release_resources()

    def release_resources(self):
        """Releases camera and video writer resources."""
        global video_writer, recording_active
        self.cap.release()
        if video_writer:
            video_writer.release()
        if self.csv_file:
            self.csv_file.close()
        recording_active = False
        logging.info("Resources released. Script terminated.")

# ==================== Video Cropping Script ======================
def crop_event_videos():
    """Reads the CSV, crops video segments using FFmpeg, and stores them on the USB."""
    if not os.path.exists(CSV_FILE_PATH):
        logging.warning("No event log CSV found to process.")
        return

    try:
        with open(CSV_FILE_PATH, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # Skip header
            for row in reader:
                try:
                    timestamp, event, start_time_str, end_time_str, video_file = row
                    
                    video_path = os.path.join(VIDEO_LOG_DIR, video_file)
                    if not os.path.exists(video_path):
                        logging.warning(f"Original video file not found: {video_path}. Skipping.")
                        continue
                    
                    start_time = float(start_time_str)
                    end_time = float(end_time_str)
                    
                    duration = end_time - start_time
                    
                    output_file_name = f"{event}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                    output_path = os.path.join(LOG_DIR_USB, output_file_name)
                    
                    # Use FFmpeg to crop the video segment
                    command = [
                        'ffmpeg', '-y', '-i', video_path, '-ss', str(start_time),
                        '-t', str(duration), '-c', 'copy', output_path
                    ]
                    
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    logging.info(f"Successfully cropped and stored {output_file_name} on USB.")
                except Exception as e:
                    logging.error(f"Error processing row {row}: {e}")
    except Exception as e:
        logging.error(f"Failed to read event log CSV: {e}")

# ==================== Main Execution Logic ======================
def signal_handler(sig, frame):
    """Handles graceful shutdown."""
    print("Interrupt signal received. Shutting down...")
    logging.info("Interrupt signal received. Initiating graceful shutdown.")
    if 'monitor' in globals() and monitor is not None:
        monitor.release_resources()
    crop_event_videos()
    sys.exit(0)

def setup_service():
    """
    Creates and installs a systemd service file to run the script on boot.
    This part is for demonstration and requires root privileges.
    """
    service_content = f"""
[Unit]
Description=DMM Monitor
After=network.target

[Service]
ExecStart=/usr/bin/python3 {os.path.abspath(__file__)}
WorkingDirectory={os.getcwd()}
StandardOutput=inherit
StandardError=inherit
Restart=always
User={os.getlogin()}

[Install]
WantedBy=multi-user.target
    """
    
    service_file = "/etc/systemd/system/dmm_monitor.service"
    print(f"Service file content:\n{service_content}")
    print("To install, save this content to /etc/systemd/system/dmm_monitor.service and run:")
    print("sudo systemctl daemon-reload && sudo systemctl enable dmm_monitor.service && sudo systemctl start dmm_monitor.service")

if __name__ == '__main__':
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Run the Edgetensor DMM Monitor.")
    parser.add_argument("--roi", type=str, default="DEFAULT", help="Specify the ROI position from config file.")
    args = parser.parse_args()
    
    # Check for the config.ini file
    if not os.path.exists("config.ini"):
        print("config.ini not found. Please create it with the required thresholds and timers.")
        print("Example content for config.ini:")
        print("[THRESHOLDS]")
        print("ear_threshold = 0.25")
        print("yawn_threshold = 0.5")
        print("brightness_threshold = 50.0")
        print("color_var_threshold = 15.0")
        print("left_threshold = 15.0")
        print("right_threshold = 15.0")
        print("up_threshold = 15.0")
        print("down_threshold = 15.0")
        print("[TIMERS]")
        print("yawn_time = 1.5")
        print("fatigue_time = 3.0")
        print("distraction_time = 2.0")
        print("face_obstruction_time = 5.0")
        print("camera_obstruction_time = 5.0")
        print("smoking_time = 2.0")
        print("phone_use_time = 2.0")
        print("eating_drinking_time = 2.0")
        print("alert_delay_seconds = 5.0")
        print("[DEFAULT]")
        print("active_roi = DEFAULT")
        print("[ROI_DEFAULT]")
        print("x1 = 200")
        print("y1 = 150")
        print("x2 = 600")
        print("y2 = 600")
        sys.exit(1)

    monitor = EdgetensorDMMMonitor(roi_position=args.roi)
    monitor.run()
