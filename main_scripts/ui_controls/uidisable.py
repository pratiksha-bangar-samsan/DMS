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

# ==================== Functions to be used in the Class ======================
def eye_aspect_ratio(eye_landmarks):
    p2_p6 = dist.euclidean(np.array([eye_landmarks[1].x, eye_landmarks[1].y]), np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    p3_p5 = dist.euclidean(np.array([eye_landmarks[2].x, eye_landmarks[2].y]), np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    p1_p4 = dist.euclidean(np.array([eye_landmarks[0].x, eye_landmarks[0].y]), np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def mouth_aspect_ratio(landmarks, MOUTH_INNER_VERTICAL, MOUTH_OUTER_HORIZONTAL):
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
        self.mp_drawing = mp.solutions.drawing_utils
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

        self.events = {'smoking': {'active': False, 'timer': 0.0, 'threshold': self.SMOKING_TIME, 'alert_time': 0.0},
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

        # === New Logging Setup ===
        log_file_path = '/home/jetson/obstruction_gpu/dms_logs/dms.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='a'),  # 'a' for append mode
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

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

        for event in self.events:
            self.events[event]['active'] = False
        
        self.yawning_duration += delta_time if yawn_detected else -self.yawning_duration
        
        if left_eye_closed and right_eye_closed:
            self.fatigue_duration += delta_time
        else:
            self.fatigue_duration = max(0, self.fatigue_duration - delta_time * 2) 

        self.distraction_duration += delta_time if distraction_detected else -self.distraction_duration
        self.camera_obstruction_duration += delta_time if is_camera_obstructed else -self.camera_obstruction_duration
        self.face_obstruction_duration += delta_time if faces_detected == 0 and not is_camera_obstructed else -self.face_obstruction_duration

        if left_eye_closed:
            self.left_eye_closed_duration += delta_time
        else:
            self.left_eye_closed_duration = max(0, self.left_eye_closed_duration - delta_time * 2)

        if right_eye_closed:
            self.right_eye_closed_duration += delta_time
        else:
            self.right_eye_closed_duration = max(0, self.right_eye_closed_duration - delta_time * 2)

        self.yawning_duration = max(0, self.yawning_duration)
        self.fatigue_duration = max(0, self.fatigue_duration)
        self.distraction_duration = max(0, self.distraction_duration)
        self.camera_obstruction_duration = max(0, self.camera_obstruction_duration)
        self.face_obstruction_duration = max(0, self.face_obstruction_duration)

        self.events['yawning']['timer'] = self.yawning_duration
        if self.events['yawning']['timer'] >= self.YAWN_TIME_THRESHOLD:
            self.events['yawning']['active'] = True
            if (current_time - self.events['yawning']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('yawning')
                self.logger.warning(f"Yawning Detected! Duration: {self.yawning_duration:.1f} secs")
                self.events['yawning']['alert_time'] = current_time

        self.events['fatigue']['timer'] = self.fatigue_duration
        if self.events['fatigue']['timer'] >= self.FATIGUE_TIME_THRESHOLD:
            self.events['fatigue']['active'] = True
            if (current_time - self.events['fatigue']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('fatigue')
                self.logger.critical(f"FATIGUE DETECTED! Duration: {self.fatigue_duration:.1f} secs")
                self.events['fatigue']['alert_time'] = current_time

        self.events['distraction']['timer'] = self.distraction_duration
        if self.events['distraction']['timer'] >= self.DISTRACTION_TIME_THRESHOLD:
            self.events['distraction']['active'] = True
            if (current_time - self.events['distraction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('distraction')
                self.logger.warning(f"Distraction Detected! Duration: {self.distraction_duration:.1f} secs")
                self.events['distraction']['alert_time'] = current_time

        self.events['camera_obstruction']['timer'] = self.camera_obstruction_duration
        if self.events['camera_obstruction']['timer'] >= self.CAMERA_OBSTRUCTION_TIME:
            self.events['camera_obstruction']['active'] = True
            if (current_time - self.events['camera_obstruction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('camera_obstruction')
                self.logger.error(f"Camera Obstructed! Duration: {self.camera_obstruction_duration:.1f} secs")
                self.events['camera_obstruction']['alert_time'] = current_time

        self.events['face_obstruction']['timer'] = self.face_obstruction_duration
        if self.events['face_obstruction']['timer'] >= self.FACE_OBSTRUCTION_TIME:
            self.events['face_obstruction']['active'] = True
            if (current_time - self.events['face_obstruction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                self.play_alert('face_obstruction')
                self.logger.error(f"Face Obstructed! Duration: {self.face_obstruction_duration:.1f} secs")
                self.events['face_obstruction']['alert_time'] = current_time

        self.events['left_eye_closed']['active'] = left_eye_closed
        self.events['right_eye_closed']['active'] = right_eye_closed

        if left_eye_closed:
            self.logger.info(f"Left Eye Closed Duration: {self.left_eye_closed_duration:.1f} secs")
        if right_eye_closed:
            self.logger.info(f"Right Eye Closed Duration: {self.right_eye_closed_duration:.1f} secs")
        
    def run(self):
        self.logger.info("DMS Monitor service started.")

        while True:
            frame_start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to read a frame from the camera.")
                break
            
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
                        yawn_detected, yawn_confidence = self.detect_yawn(face_landmarks.landmark)
                        distraction_detected_from_head_pose, head_direction, self.head_pose_yaw, self.head_pose_pitch, self.head_pose_roll = self.detect_head_pose(face_landmarks.landmark)

            self.update_event_states(yawn_detected, distraction_detected_from_head_pose, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected)

            current_time = time.time()
            self.metrics['frame_processing_time'] = current_time - frame_start_time
            if self.metrics['frame_processing_time'] > 0:
                self.metrics['fps'] = 1.0 / self.metrics['frame_processing_time']

        self.cap.release()
        self.logger.info("DMS Monitor service stopped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Edgetensor DMM Monitor.")
    parser.add_argument("--roi", type=str, default="DEFAULT", help="Specify the ROI position from config file (e.g., 'TOP', 'BOTTOM').")
    args = parser.parse_args()
    monitor = EdgetensorDMMMonitor(roi_position=args.roi)
    monitor.run()
