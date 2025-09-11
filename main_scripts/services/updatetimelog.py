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
import subprocess
from datetime import datetime

# ==================== Functions to be used in the Class ======================
def eye_aspect_ratio(eye_landmarks):
    """Calculates the Eye Aspect Ratio (EAR)."""
    p2_p6 = dist.euclidean(np.array([eye_landmarks[1].x, eye_landmarks[1].y]), np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    p3_p5 = dist.euclidean(np.array([eye_landmarks[2].x, eye_landmarks[2].y]), np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    p1_p4 = dist.euclidean(np.array([eye_landmarks[0].x, eye_landmarks[0].y]), np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def mouth_aspect_ratio(landmarks, MOUTH_INNER_VERTICAL, MOUTH_OUTER_HORIZONTAL):
    """Calculates the Mouth Aspect Ratio (MAR)."""
    A = dist.euclidean(np.array([landmarks[MOUTH_INNER_VERTICAL[0]].x, landmarks[MOUTH_INNER_VERTICAL[0]].y]),
                       np.array([landmarks[MOUTH_INNER_VERTICAL[1]].x, landmarks[MOUTH_INNER_VERTICAL[1]].y]))
    B = dist.euclidean(np.array([landmarks[MOUTH_OUTER_HORIZONTAL[0]].x, landmarks[MOUTH_OUTER_HORIZONTAL[0]].y]),
                       np.array([landmarks[MOUTH_OUTER_HORIZONTAL[1]].x, landmarks[MOUTH_OUTER_HORIZONTAL[1]].y]))
    if B == 0:
        return 0
    mar = A / B
    return mar

# ==================== EventRecorder Class ======================
class EventRecorder:
    def __init__(self, base_dir, fps, logger):
        self.base_dir = base_dir
        self.fps = fps
        self.logger = logger
        self.video_writer = None
        self.audio_process = None
        self.is_recording = False
        self.event_dir = None
        self.video_path = None
        self.audio_path = None

    def start_recording(self, event_name, frame):
        if self.is_recording:
            return

        self.is_recording = True
        self.logger.info(f"Starting recording for event: {event_name}")

        date_dir = datetime.now().strftime('%Y-%m-%d')
        self.event_dir = os.path.join(self.base_dir, date_dir, event_name)
        
        if not os.path.exists(self.event_dir):
            os.makedirs(self.event_dir)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.video_path = os.path.join(self.event_dir, f"{timestamp}.avi")
        self.audio_path = os.path.join(self.event_dir, f"{timestamp}.wav")

        frame_height, frame_width, _ = frame.shape
        codec = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.video_path, codec, self.fps, (frame_width, frame_height))

        # This subprocess command is for audio recording using arecord
        self.audio_process = subprocess.Popen(
            ['arecord', '-D', 'plughw:1', '-f', 'S16_LE', '-c1', '-r16000', self.audio_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.logger.info(f"Recording video to: {self.video_path}")
        self.logger.info(f"Recording audio to: {self.audio_path}")
        
    def record_frame(self, frame):
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.logger.info("Stopping recording...")
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
            if self.audio_process:
                self.audio_process.terminate()
                self.audio_process.wait()
                self.audio_process = None
                
            self.logger.info("Recording stopped and files saved.")

# ==================== EdgetensorDMMMonitor Class ======================
class EdgetensorDMMMonitor:
    def __init__(self, roi_position, config_file='config.ini'):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.config = self.load_config(config_file)

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

        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH_INNER_VERTICAL = [13, 14]
        self.MOUTH_OUTER_HORIZONTAL = [61, 291]
        self.NOSE_LANDMARKS = [4, 6, 195]

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.roi_position = roi_position
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = self.get_roi_coordinates(roi_position)

        self.audio_path = "/home/jetson/obstruction_gpu/audio_alert"

        self.voice_alerts = {
            'yawning': 'yawning.mp3',
            'fatigue': 'fatigue.mp3',
            'camera_obstruction': 'obstruction.mp3',
            'face_obstruction': 'face_obstruction.mp3',
            'smoking': 'smoking.mp3',
            'phone_uses': 'phone_uses.mp3',
            'eating_drinking': 'eating_drinking.mp3',
            'seatbelt_off': 'seatbelt_off.mp3',
        }

        self.events = {
            'smoking': {'active': False, 'timer': 0.0, 'threshold': self.SMOKING_TIME, 'alert_time': 0.0},
            'yawning': {'active': False, 'timer': 0.0, 'threshold': self.YAWN_TIME_THRESHOLD, 'alert_time': 0.0},
            'eating_drinking': {'active': False, 'timer': 0.0, 'threshold': self.EATING_DRINKING_TIME, 'alert_time': 0.0},
            'phone_uses': {'active': False, 'timer': 0.0, 'threshold': self.PHONE_USE_TIME, 'alert_time': 0.0},
            'wearing_mask': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
            'seatbelt_off': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
            'wearing_seatbelt': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
            'fatigue': {'active': False, 'timer': 0.0, 'threshold': self.FATIGUE_TIME_THRESHOLD, 'alert_time': 0.0},
            'distraction': {'active': False, 'timer': 0.0, 'threshold': self.DISTRACTION_TIME_THRESHOLD, 'alert_time': 0.0},
            'camera_obstruction': {'active': False, 'timer': 0.0, 'threshold': self.CAMERA_OBSTRUCTION_TIME, 'alert_time': 0.0},
            'face_obstruction': {'active': False, 'timer': 0.0, 'threshold': self.FACE_OBSTRUCTION_TIME, 'alert_time': 0.0},
            'left_eye_closed': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
            'right_eye_closed': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
        }

        self.direction_timers = {'left': 0.0, 'right': 0.0, 'up': 0.0, 'down': 0.0}

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

        self.logger = logging.getLogger(__name__)
        self.current_log_hour = None
        self.setup_logging()

        self.recordings_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'event_recordings')
        self.event_recorder = EventRecorder(self.recordings_base_dir, 10, self.logger)
        self.last_processed_frame = None
        self.last_primary_event = None

    def setup_logging(self):
        current_date_str = datetime.now().strftime('%Y-%m-%d')
        current_hour = datetime.now().hour
        
        if current_hour == self.current_log_hour:
            return

        self.current_log_hour = current_hour
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dms_logs', current_date_str)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_path = os.path.join(log_dir, f"dms_log_{current_date_str}_{current_hour:02d}.log")
        
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.info(f"New log file started: {log_file_path}")

    def load_config(self, config_file):
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            self.logger.error(f"Config file '{config_file}' not found.")
            exit(1)
        config.read(config_file)
        return config
    
    def get_roi_coordinates(self, roi_position):
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
            self.logger.error(f"Error reading ROI configuration: {e}. Using hardcoded default.")
            return 200, 150, 600, 600

    def play_alert(self, event_name):
        def play_sound_async():
            if event_name in self.voice_alerts:
                audio_file = os.path.join(self.audio_path, self.voice_alerts[event_name])
                if os.path.exists(audio_file):
                    try:
                        playsound(audio_file)
                    except Exception as e:
                        self.logger.error(f"Error playing sound for {event_name}: {e}")
                else:
                    self.logger.warning(f"Audio file not found for event '{event_name}' at {audio_file}")
            else:
                self.logger.warning(f"No voice alert defined for event '{event_name}'.")
        thread = threading.Thread(target=play_sound_async)
        thread.daemon = True
        thread.start()

    def detect_eye_closure(self, landmarks):
        left_eye_landmarks = [landmarks[i] for i in self.LEFT_EYE]
        right_eye_landmarks = [landmarks[i] for i in self.RIGHT_EYE]
        left_ear = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)
        left_eye_closed = left_ear < self.EAR_THRESHOLD
        right_eye_closed = right_ear < self.EAR_THRESHOLD
        return left_eye_closed, right_eye_closed

    def detect_yawn(self, landmarks):
        mar = mouth_aspect_ratio(landmarks, self.MOUTH_INNER_VERTICAL, self.MOUTH_OUTER_HORIZONTAL)
        yawn_detected = mar > self.YAWN_THRESHOLD
        yawn_confidence = mar / (self.YAWN_THRESHOLD * 1.5) if yawn_detected else 0
        yawn_confidence = min(1.0, yawn_confidence)
        return yawn_detected, yawn_confidence

    def detect_head_pose(self, landmarks):
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
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_frame)
        color_variance = np.std(frame)
        is_dark_obstructed = mean_intensity < self.BRIGHTNESS_THRESHOLD
        is_uniform_obstructed = color_variance < self.COLOR_VAR_THRESHOLD
        return is_dark_obstructed or is_uniform_obstructed

    def update_event_states(self, yawn_detected, distraction_detected, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected):
        delta_time = 1 / self.metrics['fps'] if self.metrics['fps'] > 0 else 0
        current_time = time.time()
        
        primary_event = None
        if is_camera_obstructed:
            primary_event = 'camera_obstruction'
        elif faces_detected == 0 and not is_camera_obstructed:
            primary_event = 'face_obstruction'
        elif yawn_detected:
            primary_event = 'yawning'
        elif distraction_detected:
            primary_event = 'distraction'
        elif left_eye_closed and right_eye_closed:
            primary_event = 'fatigue'

        if primary_event != self.last_primary_event:
            if self.event_recorder.is_recording:
                self.event_recorder.stop_recording()
            
            if primary_event is not None and self.last_processed_frame is not None:
                self.event_recorder.start_recording(primary_event, self.last_processed_frame)

        self.last_primary_event = primary_event

        # Corrected logic for updating event durations
        if yawn_detected:
            self.yawning_duration += delta_time
        else:
            self.yawning_duration = 0.0

        if left_eye_closed and right_eye_closed:
            self.fatigue_duration += delta_time
        else:
            self.fatigue_duration = 0.0

        if distraction_detected:
            self.distraction_duration += delta_time
        else:
            self.distraction_duration = 0.0
            
        if is_camera_obstructed:
            self.camera_obstruction_duration += delta_time
        else:
            self.camera_obstruction_duration = 0.0
            
        if faces_detected == 0 and not is_camera_obstructed:
            self.face_obstruction_duration += delta_time
        else:
            self.face_obstruction_duration = 0.0

        # Check and trigger alerts based on updated durations
        if self.yawning_duration >= self.YAWN_TIME_THRESHOLD:
            self.events['yawning']['active'] = True
            if (current_time - self.events['yawning']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('yawning')
                self.logger.warning(f"Yawning Detected! Duration: {self.yawning_duration:.1f} secs")
                self.events['yawning']['alert_time'] = current_time

        if self.fatigue_duration >= self.FATIGUE_TIME_THRESHOLD:
            self.events['fatigue']['active'] = True
            if (current_time - self.events['fatigue']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('fatigue')
                self.logger.critical(f"FATIGUE DETECTED! Duration: {self.fatigue_duration:.1f} secs")
                self.events['fatigue']['alert_time'] = current_time

        if self.distraction_duration >= self.DISTRACTION_TIME_THRESHOLD:
            self.events['distraction']['active'] = True
            if (current_time - self.events['distraction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('distraction')
                self.logger.warning(f"Distraction Detected! Duration: {self.distraction_duration:.1f} secs")
                self.events['distraction']['alert_time'] = current_time

        if self.camera_obstruction_duration >= self.CAMERA_OBSTRUCTION_TIME:
            self.events['camera_obstruction']['active'] = True
            if (current_time - self.events['camera_obstruction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('camera_obstruction')
                self.logger.error(f"Camera Obstructed! Duration: {self.camera_obstruction_duration:.1f} secs")
                self.events['camera_obstruction']['alert_time'] = current_time

        if self.face_obstruction_duration >= self.FACE_OBSTRUCTION_TIME:
            self.events['face_obstruction']['active'] = True
            if (current_time - self.events['face_obstruction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('face_obstruction')
                self.logger.error(f"Face Obstructed! Duration: {self.face_obstruction_duration:.1f} secs")
                self.events['face_obstruction']['alert_time'] = current_time
        
    def run(self):
        self.logger.info("DMS Monitor service started.")

        try:
            while True:
                if datetime.now().hour != self.current_log_hour:
                    self.setup_logging()

                frame_start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to read a frame from the camera.")
                    break
                
                self.last_processed_frame = cv2.resize(frame, (640, 480))

                is_camera_obstructed = self.check_camera_obstruction(self.last_processed_frame)

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
                    rgb_frame = cv2.cvtColor(self.last_processed_frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:
                        faces_detected = len(results.multi_face_landmarks)
                        for face_landmarks in results.multi_face_landmarks:
                            left_eye_closed, right_eye_closed = self.detect_eye_closure(face_landmarks.landmark)
                            yawn_detected, yawn_confidence = self.detect_yawn(face_landmarks.landmark)
                            distraction_detected_from_head_pose, head_direction, self.head_pose_yaw, self.head_pose_pitch, self.head_pose_roll = self.detect_head_pose(face_landmarks.landmark)

                self.update_event_states(yawn_detected, distraction_detected_from_head_pose, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected)
                self.event_recorder.record_frame(self.last_processed_frame)

                current_time = time.time()
                self.metrics['frame_processing_time'] = current_time - frame_start_time
                if self.metrics['frame_processing_time'] > 0:
                    self.metrics['fps'] = 1.0 / self.metrics['frame_processing_time']
        
        except KeyboardInterrupt:
            self.logger.info("Service stopped by user (KeyboardInterrupt).")
        finally:
            self.cap.release()
            self.event_recorder.stop_recording()
            self.logger.info("DMS Monitor service stopped gracefully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Edgetensor DMM Monitor.")
    parser.add_argument("--roi", type=str, default="DEFAULT", help="Specify the ROI position from config file (e.g., 'TOP', 'BOTTOM').")
    args = parser.parse_args()
    monitor = EdgetensorDMMMonitor(roi_position=args.roi)
    monitor.run()
