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

def mouth_aspect_ratio(landmarks, MOUTH_INNER_TOP, MOUTH_INNER_BOTTOM, MOUTH_OUTER_HORIZONTAL):
    """Calculates the Mouth Aspect Ratio (MAR)."""
    A = dist.euclidean(np.array([landmarks[MOUTH_INNER_TOP[0]].x, landmarks[MOUTH_INNER_TOP[0]].y]),
                       np.array([landmarks[MOUTH_INNER_BOTTOM[0]].x, landmarks[MOUTH_INNER_BOTTOM[0]].y]))
    B = dist.euclidean(np.array([landmarks[MOUTH_INNER_TOP[1]].x, landmarks[MOUTH_INNER_TOP[1]].y]),
                       np.array([landmarks[MOUTH_INNER_BOTTOM[1]].x, landmarks[MOUTH_INNER_BOTTOM[1]].y]))
    C = dist.euclidean(np.array([landmarks[MOUTH_INNER_TOP[2]].x, landmarks[MOUTH_INNER_TOP[2]].y]),
                       np.array([landmarks[MOUTH_INNER_BOTTOM[2]].x, landmarks[MOUTH_INNER_BOTTOM[2]].y]))
    D = dist.euclidean(np.array([landmarks[MOUTH_OUTER_HORIZONTAL[0]].x, landmarks[MOUTH_OUTER_HORIZONTAL[0]].y]),
                       np.array([landmarks[MOUTH_OUTER_HORIZONTAL[1]].x, landmarks[MOUTH_OUTER_HORIZONTAL[1]].y]))
    mar = (A + B + C) / (3.0 * D)
    return mar

# ==================== EdgetensorDMMMonitor Class ======================
class EdgetensorDMMMonitor:
    def __init__(self, roi_position, config_file='config.ini'):
        self.config = self.load_config(config_file)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Screen layout and thresholds from config
        self.video_width = 800
        self.panel_width = 490
        self.display_height = 750
        self.video_height = self.display_height
        self.display_width = self.video_width + self.panel_width
 # Load thresholds and timers from config
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

        # Facial landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH_INNER_TOP = [13, 308, 324]
        self.MOUTH_INNER_BOTTOM = [14, 84, 314]
        self.MOUTH_OUTER_HORIZONTAL = [61, 291]
        self.NOSE_LANDMARKS = [4, 6, 195]

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.roi_position = roi_position
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = self.get_roi_coordinates(roi_position)

        self.icons_path = "/home/jetson/DMS_new/Detection/icons"
        self.audio_path = "/home/jetson/obstruction_gpu/audio_alert" # New Audio Path
        self.icons = {}
        self.load_icons()

        # Voice alerts mapping
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
        self.distraction_start_time = None
        self.fatigue_start_time = None
        self.yawning_start_time = None
        self.smoking_start_time = None
        self.phone_use_start_time = None
        self.eating_drinking_start_time = None
        self.camera_obstructed_start_time = None
        self.face_obstructed_start_time = None

        self.gaze_text = "No Face"
        self.metrics = {'fps': 0, 'frame_processing_time': 0, 'faces_detected': 0}
        self.head_pose_yaw = 0.0
        self.head_pose_pitch = 0.0
        self.head_pose_roll = 0.0
        self.last_time = time.time()
        self.face_not_detected = False

    def load_config(self, config_file):
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            print(f"Error: Config file '{config_file}' not found.")
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
            print(f"Error reading ROI configuration: {e}. Using hardcoded default.")
            return 200, 150, 600, 600

    def play_alert(self, event_name):
        """Plays the audio alert for a given event in a separate thread."""
        def play_sound_async():
            if event_name in self.voice_alerts:
                audio_file = os.path.join(self.audio_path, self.voice_alerts[event_name])
                if os.path.exists(audio_file):
                    try:
                        playsound(audio_file)
                    except Exception as e:
                        print(f"Error playing sound for {event_name}: {e}")
                else:
                    print(f"Warning: Audio file not found for event '{event_name}' at {audio_file}")
            else:
                print(f"Warning: No voice alert defined for event '{event_name}'.")

        thread = threading.Thread(target=play_sound_async)
        thread.daemon = True
        thread.start()

    def load_icons(self):
        icon_mappings = {
            'smoking': ['Smoking/smoke.jpg'], 'yawning': ['yawning/yawn.png'],
            'eating_drinking': ['eating_drinking/food.jpg'], 'phone_uses': ['phone use/phone.png'],
            'wearing_mask': ['mask/mask.png'], 'seatbelt_off': ['noseatbelt/nobelt.jpg'],
            'wearing_seatbelt': ['seatbelt/seatbelt.jpg'], 'distraction': ['distraction/distraction.png'],
            'camera_obstruction': ['camera_obstraction/obstraction.jpeg'],
            'face_obstruction': ['face_obstruction/face.png'],
            'fatigue': ['fatigue/drowsy.png'],
        }
        icon_size = (60, 60)
        for icon_name, possible_paths in icon_mappings.items():
            icon_loaded = False
            for path in possible_paths:
                full_path = os.path.join(self.icons_path, path)
                if os.path.exists(full_path):
                    try:
                        icon = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                        if icon is not None:
                            if len(icon.shape) == 2:
                                icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)
                            icon = cv2.resize(icon, icon_size)
                            self.icons[icon_name] = icon
                            icon_loaded = True
                            break
                    except Exception as e:
                        print(f"Error loading icon {path}: {e}")
            if not icon_loaded:
                print(f"Warning: Could not load icon for {icon_name}")
                placeholder = np.zeros((*icon_size, 3), dtype=np.uint8)
                placeholder[:] = (128, 128, 128)
                cv2.putText(placeholder, icon_name[:3].upper(), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.icons[icon_name] = placeholder

    def draw_icon(self, frame, icon_name, x, y, bg_color):
        if icon_name in self.icons:
            icon_original = self.icons[icon_name]
            h, w = icon_original.shape[:2]
            icon_to_draw = icon_original.copy()
            if bg_color == (0, 255, 0):
                if icon_to_draw.shape[2] == 4:
                    mask = icon_to_draw[:, :, 3] > 0
                    green_overlay = np.zeros_like(icon_to_draw, dtype=np.uint8)
                    green_overlay[mask] = (0, 255, 0, 255)
                    icon_to_draw = cv2.addWeighted(icon_to_draw, 0.5, green_overlay, 0.5, 0)
                else:
                    gray = cv2.cvtColor(icon_to_draw, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                    icon_to_draw[mask > 0] = (0, 255, 0)
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), bg_color, -1)
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 0), 1)
            frame_h, frame_w = frame.shape[:2]
            if y + h > frame_h or x + w > frame_w:
                pass
            if len(icon_to_draw.shape) == 3 and icon_to_draw.shape[2] == 4:
                alpha_channel = icon_to_draw[:, :, 3] / 255.0
                for c in range(3):
                    frame[y:y + h, x:x + w, c] = frame[y:y + h, x:x + w, c] * (1 - alpha_channel) + icon_to_draw[:, :, c] * alpha_channel
            else:
                if len(icon_to_draw.shape) == 2:
                    icon_to_draw = cv2.cvtColor(icon_to_draw, cv2.COLOR_GRAY2BGR)
                frame[y:y + h, x:x + w] = icon_to_draw

    def detect_eye_closure(self, landmarks):
        left_eye_landmarks = [landmarks[i] for i in self.LEFT_EYE]
        right_eye_landmarks = [landmarks[i] for i in self.RIGHT_EYE]
        left_ear = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)
        left_eye_closed = left_ear < self.EAR_THRESHOLD
        right_eye_closed = right_ear < self.EAR_THRESHOLD
        return left_eye_closed, right_eye_closed

    def detect_yawn(self, landmarks):
        mar = mouth_aspect_ratio(landmarks, self.MOUTH_INNER_TOP, self.MOUTH_INNER_BOTTOM, self.MOUTH_OUTER_HORIZONTAL)
        yawn_detected = mar > self.YAWN_THRESHOLD
        yawn_confidence = mar / (self.YAWN_THRESHOLD * 1.5) if yawn_detected else 0
        yawn_confidence = min(1.0, yawn_confidence)
        return yawn_detected, yawn_confidence

    def detect_head_pose(self, landmarks):
        image_points = np.array([
            (landmarks[33].x * self.video_width, landmarks[33].y * self.video_height),
            (landmarks[263].x * self.video_width, landmarks[263].y * self.video_height),
            (landmarks[1].x * self.video_width, landmarks[1].y * self.video_height),
            (landmarks[61].x * self.video_width, landmarks[61].y * self.video_height),
            (landmarks[291].x * self.video_width, landmarks[291].y * self.video_height),
            (landmarks[199].x * self.video_width, landmarks[199].y * self.video_height)
        ], dtype="double")
        model_points = np.array([
            (0.0, 0.0, 0.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0), (0.0, 0.0, -330.0)
        ])
        focal_length = self.video_width
        center = (self.video_width / 2, self.video_height / 2)
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

    def update_event_states(self, yawn_detected, distraction_detected_from_head_pose, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected):
        delta_time = 1 / self.metrics['fps'] if self.metrics['fps'] > 0 else 0
        current_time = time.time()

        # Handle Camera Obstruction
        if is_camera_obstructed:
            if self.camera_obstructed_start_time is None:
                self.camera_obstructed_start_time = current_time
            self.events['camera_obstruction']['timer'] = current_time - self.camera_obstructed_start_time
            if self.events['camera_obstruction']['timer'] >= self.CAMERA_OBSTRUCTION_TIME:
                self.events['camera_obstruction']['active'] = True
                if (current_time - self.events['camera_obstruction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                    self.play_alert('camera_obstruction')
                    self.events['camera_obstruction']['alert_time'] = current_time
            for event_name in self.events:
                if event_name not in ['camera_obstruction', 'face_obstruction']:
                    self.events[event_name]['active'] = False
                    self.events[event_name]['timer'] = 0.0
            return
        else:
            self.camera_obstructed_start_time = None
            self.events['camera_obstruction']['active'] = False
            self.events['camera_obstruction']['timer'] = 0.0

        # Handle Face Obstruction
        if faces_detected == 0 and not is_camera_obstructed:
            if self.face_obstructed_start_time is None:
                self.face_obstructed_start_time = current_time
            self.events['face_obstruction']['timer'] = current_time - self.face_obstructed_start_time
            if self.events['face_obstruction']['timer'] >= self.FACE_OBSTRUCTION_TIME:
                self.events['face_obstruction']['active'] = True
                if (current_time - self.events['face_obstruction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                    self.play_alert('face_obstruction')
                    self.events['face_obstruction']['alert_time'] = current_time
            for event_name in self.events:
                if event_name not in ['camera_obstruction', 'face_obstruction']:
                    self.events[event_name]['active'] = False
                    self.events[event_name]['timer'] = 0.0
            return
        else:
            self.face_obstructed_start_time = None
            self.events['face_obstruction']['active'] = False
            self.events['face_obstruction']['timer'] = 0.0

        # Handle Distraction (Head Pose)
        is_distracted = distraction_detected_from_head_pose
        if is_distracted:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            self.events['distraction']['timer'] = current_time - self.distraction_start_time
            if self.events['distraction']['timer'] >= self.DISTRACTION_TIME_THRESHOLD:
                self.events['distraction']['active'] = True
                if (current_time - self.events['distraction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                    self.play_alert('distraction')
                    self.events['distraction']['alert_time'] = current_time
            for direction in self.direction_timers:
                if direction == head_direction:
                    self.direction_timers[direction] += delta_time
                else:
                    self.direction_timers[direction] = 0.0
        else:
            self.distraction_start_time = None
            self.events['distraction']['active'] = False
            self.events['distraction']['timer'] = 0.0
            for direction in self.direction_timers:
                self.direction_timers[direction] = 0.0

        # Handle Yawning
        if yawn_detected:
            if self.yawning_start_time is None:
                self.yawning_start_time = current_time
            self.events['yawning']['timer'] = current_time - self.yawning_start_time
            if self.events['yawning']['timer'] >= self.YAWN_TIME_THRESHOLD:
                self.events['yawning']['active'] = True
                if (current_time - self.events['yawning']['alert_time']) > self.ALERT_DELAY_SECONDS:
                     self.play_alert('yawning')
                    self.events['yawning']['alert_time'] = current_time
        else:
            self.yawning_start_time = None
            self.events['yawning']['active'] = False
            self.events['yawning']['timer'] = 0.0

        # Handle Fatigue
        if left_eye_closed and right_eye_closed:
            if self.fatigue_start_time is None:
                self.fatigue_start_time = current_time
            closed_time = current_time - self.fatigue_start_time
            self.events['fatigue']['timer'] = closed_time
            if closed_time >= self.FATIGUE_TIME_THRESHOLD:
                self.events['fatigue']['active'] = True
                if (current_time - self.events['fatigue']['alert_time']) > self.ALERT_DELAY_SECONDS:
                    self.play_alert('fatigue')
                    self.events['fatigue']['alert_time'] = current_time
        else:
            self.fatigue_start_time = None
            self.events['fatigue']['active'] = False
            self.events['fatigue']['timer'] = 0.0

        self.events['left_eye_closed']['active'] = left_eye_closed
        self.events['left_eye_closed']['timer'] = (self.events['left_eye_closed']['timer'] + delta_time) if left_eye_closed else 0.0
        self.events['right_eye_closed']['active'] = right_eye_closed
        self.events['right_eye_closed']['timer'] = (self.events['right_eye_closed']['timer'] + delta_time) if right_eye_closed else 0.0

        # Reset other timers if a major event is active
        if self.events['fatigue']['active'] or self.events['yawning']['active'] or self.events['distraction']['active']:
            self.events['smoking']['active'] = False
            self.events['smoking']['timer'] = 0.0
            self.smoking_start_time = None

            self.events['phone_uses']['active'] = False
            self.events['phone_uses']['timer'] = 0.0
            self.phone_use_start_time = None

            self.events['eating_drinking']['active'] = False
            self.events['eating_drinking']['timer'] = 0.0
            self.eating_drinking_start_time = None

            self.events['wearing_mask']['active'] = False
            self.events['wearing_mask']['timer'] = 0.0

    def draw_dms_events_panel(self, frame):
        panel_x, panel_y, events_panel_height = self.video_width, 0, 500
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + events_panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + events_panel_height), (0, 255, 0), 2)

        label_text = "DMS Events"
        (text_w, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_x = panel_x + (self.panel_width - text_w) // 2
        cv2.putText(frame, label_text, (label_x, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        events_list = [
            ('smoking', 'Smoking'), ('yawning', 'Yawning'), ('camera_obstruction', 'Cam Obsc'),
            ('face_obstruction', 'Face Obsc'),('fatigue', 'Fatigue'),('distraction', 'Distraction'), ('phone_uses', 'Phone'),
            ('wearing_mask', 'Mask'),('seatbelt_off', 'S_OFF'), ('wearing_seatbelt', 'S_ON'),
            ('eating_drinking', 'Eating/Drinking'),
        ]
        icon_start_y, icon_size, row_spacing, icon_spacing_x = panel_y + 80, 60, 92, 90
        for i, (icon_name, label) in enumerate(events_list):
            row, col = i // 5, i % 5
            icon_x, icon_y = panel_x + 20 + col * icon_spacing_x, icon_start_y + row * row_spacing
            is_active = self.events.get(icon_name, {'active': False})['active']
            bg_color = (0, 255, 0) if is_active else (220, 220, 220)
            label_offset_y = icon_size + 15
            (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_x = icon_x + (icon_size - text_w) // 2
            cv2.putText(frame, label, (label_x, icon_y + label_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            self.draw_icon(frame, icon_name, icon_x, icon_y, bg_color)

    def draw_dms_output_panel(self, frame):
        panel_x, panel_y = self.video_width, 350
        panel_height = self.display_height - panel_y
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + panel_height), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + panel_height), (0, 255, 0), 2)
        cv2.putText(frame, "DMS Output Parameters:", (panel_x + 10, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset = 60

        obstruction_text = ""
        if self.events['camera_obstruction']['active']:
            obstruction_text = "Camera is Obstructed"
        elif self.events['face_obstruction']['active']:
            obstruction_text = "Face is Obstructed"

        params = [
            f"Camera FPS: {self.metrics['fps']:.1f}",
            f"Distraction_Looking_left: {self.direction_timers['left']:.1f} secs",
            f"Distraction_Looking_right: {self.direction_timers['right']:.1f} secs",
            f"Distraction_Looking_up: {self.direction_timers['up']:.1f} secs",
            f"Distraction_Looking_down: {self.direction_timers['down']:.1f} secs",
            f"Head Pose (Yaw): {self.head_pose_yaw:.1f} deg",
            f"Head Pose (Pitch): {self.head_pose_pitch:.1f} deg",
            f"Head Pose (Roll): {self.head_pose_roll:.1f} deg",
            f"Fatigue: {self.events['fatigue']['timer']:.1f} secs",
            f"Yawning: {self.events['yawning']['timer']:.1f} secs",
            f"Phone Use: {self.events['phone_uses']['timer']:.1f} secs",
            f"Eating/Drinking: {self.events['eating_drinking']['timer']:.1f} secs",
            f"Smoking: {self.events['smoking']['timer']:.1f} secs",
            f"Mask: {self.events['wearing_mask']['timer']:.1f} secs",
            f"Left Eye Closed: {self.events['left_eye_closed']['timer']:.1f} secs",
            f"Right Eye Closed: {self.events['right_eye_closed']['timer']:.1f} secs",
        ]

        if obstruction_text:
            params.append(f"Obstruction: {obstruction_text}")

        for param in params:
            cv2.putText(frame, param, (panel_x + 10, panel_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_offset += 20

    def run(self):
        print("Starting Samsan DMS Monitor...")
        print("Press 'q' to quit")

        while True:
            frame_start_time = time.time()
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            display_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
            display_frame[0:self.video_height, 0:self.video_width] = cv2.resize(frame, (self.video_width, self.video_height))
            video_resized = display_frame[0:self.video_height, 0:self.video_width]

            # Draw the ROI bounding box
            cv2.rectangle(video_resized, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 255, 0), 2)
            cv2.putText(video_resized, f"ROI: {self.roi_position}", (self.roi_x1, self.roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
                # Crop the frame to the ROI before processing with MediaPipe
                roi_frame = video_resized[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
                rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                 results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    faces_detected = len(results.multi_face_landmarks)
                    for face_landmarks in results.multi_face_landmarks:
                        # Adjust landmark coordinates to be relative to the full frame for drawing and calculations
                        h_roi, w_roi, _ = roi_frame.shape
                        full_frame_landmarks = face_landmarks
                        for landmark in full_frame_landmarks.landmark:
                            landmark.x = (landmark.x * w_roi + self.roi_x1) / self.video_width
                            landmark.y = (landmark.y * h_roi + self.roi_y1) / self.video_height

                        # Draw face contour
                        self.mp_drawing.draw_landmarks(
                            image=video_resized,
                            landmark_list=full_frame_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        # Draw eyes
                        self.mp_drawing.draw_landmarks(
                            image=video_resized,
                            landmark_list=full_frame_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        self.mp_drawing.draw_landmarks(
                            image=video_resized,
                            landmark_list=full_frame_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        # Draw lips (outer and inner)
                        self.mp_drawing.draw_landmarks(
                            image=video_resized,
                            landmark_list=full_frame_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_LIPS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        landmarks_list = list(full_frame_landmarks.landmark)

                        yawn_detected, _ = self.detect_yawn(landmarks_list)
                        distraction_detected_from_head_pose, head_direction, self.head_pose_yaw, self.head_pose_pitch, self.head_pose_roll = self.detect_head_pose(landmarks_list)
                        left_eye_closed, right_eye_closed = self.detect_eye_closure(landmarks_list)

            self.update_event_states(yawn_detected, distraction_detected_from_head_pose, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected)

            # Draw the UI panels
            self.draw_dms_events_panel(display_frame)
            self.draw_dms_output_panel(display_frame)

            # Calculate FPS
            frame_end_time = time.time()
            self.metrics['frame_processing_time'] = (frame_end_time - frame_start_time) * 1000
            if self.metrics['frame_processing_time'] > 0:
                self.metrics['fps'] = 1000 / self.metrics['frame_processing_time']
            else:
                self.metrics['fps'] = 0

            # Display the result
            cv2.imshow("Samsan DMS Monitor", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Samsan DMS Monitor.')
    parser.add_argument('--roi', type=str, default='left_hand', help='Specify the ROI to use from config.ini.')
    args = parser.parse_args()
    
    monitor = EdgetensorDMMMonitor(roi_position=args.roi)
    monitor.run()


    


