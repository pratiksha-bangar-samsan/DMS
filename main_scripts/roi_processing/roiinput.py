import cv2
import time
import numpy as np
import os
import argparse
import mediapipe as mp
from scipy.spatial import distance as dist

# ===== MediaPipe FaceMesh =====
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== Screen Layout Parameters and Thresholds =====
BRIGHTNESS_THRESHOLD = 100
COLOR_VAR_THRESHOLD = 30
EAR_THRESHOLD = 0.25
YAWN_THRESHOLD = 0.6  # New Yawn threshold from your logic
YAWN_TIME_THRESHOLD = 0.7
FATIGUE_TIME = 3.0
DISTRACTION_TIME_THRESHOLD = 2.0
DISTRACTION_ANGLE_THRESHOLD = 10

LEFT_THRESHOLD = 15
RIGHT_THRESHOLD =15
UP_THRESHOLD =15
DOWN_THRESHOLD =15

# Facial landmark indices for eyes and mouth (MediaPipe specific)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Adjusted mouth landmarks for MediaPipe based on provided logic's principles
# Approximating the Dlib points with MediaPipe's 468 points
MOUTH_INNER_TOP = [13, 308, 324]  # Top inner lip points
MOUTH_INNER_BOTTOM = [14, 84, 314] # Bottom inner lip points
MOUTH_OUTER_HORIZONTAL = [61, 291] # Left and right mouth corners
# Example nose landmarks for drawing
NOSE_LANDMARKS = [4, 6, 195]

class EdgetensorDMMMonitor:
    def __init__(self, roi_position):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.video_width = 800
        self.panel_width = 480
        self.display_height = 750
        self.video_height = self.display_height
        self.display_width = self.video_width + self.panel_width
        self.roi_position = roi_position

        self.icons_path = "/home/jetson/DMS_new/Detection/icons"
        self.icons = {}
        self.load_icons()

        self.events = {
            'smoking': {'active': False, 'timer': 0.0},
            'yawning': {'active': False, 'timer': 0.0},
            'eating_drinking': {'active': False, 'timer': 0.0},
            'phone_uses': {'active': False, 'timer': 0.0},
            'wearing_mask': {'active': False, 'timer': 0.0},
            'seatbelt_off': {'active': False, 'timer': 0.0},
            'wearing_seatbelt': {'active': False, 'timer': 0.0},
            'fatigue': {'active': False, 'timer': 0.0},
            'distraction': {'active': False, 'timer': 0.0},
            'camera_obstruction': {'active': False, 'timer': 0.0},
            'left_eye_closed': {'active': False, 'timer': 0.0},
            'right_eye_closed': {'active': False, 'timer': 0.0},
        }

        self.direction_timers = {'left': 0.0, 'right': 0.0, 'up': 0.0, 'down': 0.0}
        self.distraction_start_time = None
        self.fatigue_start_time = None
        self.yawning_start_time = None
        self.gaze_text = "No Face"
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = 0, 0, 0, 0
        self.metrics = {'fps': 0, 'frame_processing_time': 0, 'faces_detected': 0}
        self.head_pose_yaw = 0.0
        self.head_pose_pitch = 0.0
        self.head_pose_roll = 0.0
        self.last_time = time.time()
        self.face_not_detected = False

    def load_icons(self):
        icon_mappings = {
            'smoking': ['Smoking/smoke.jpg'], 'yawning': ['yawning/yawn.png'],
            'eating_drinking': ['eating_drinking/food.jpg'], 'phone_uses': ['phone use/phone.png'],
            'wearing_mask': ['mask/mask.png'], 'seatbelt_off': ['noseatbelt/nobelt.jpg'],
            'wearing_seatbelt': ['seatbelt/seatbelt.jpg'], 'distraction': ['distraction/distraction.png'],
            'camera_obstruction': ['camera_obstraction/obstraction.jpeg'], 'fatigue': ['fatigue/drowsy.png'],
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

    # ===== Missing methods that caused the error =====
    def eye_aspect_ratio(self, eye_landmarks):
        p2_p6 = dist.euclidean(np.array([eye_landmarks[1].x, eye_landmarks[1].y]), np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
        p3_p5 = dist.euclidean(np.array([eye_landmarks[2].x, eye_landmarks[2].y]), np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
        p1_p4 = dist.euclidean(np.array([eye_landmarks[0].x, eye_landmarks[0].y]), np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def detect_eye_closure(self, landmarks):
        left_eye_landmarks = [landmarks[i] for i in LEFT_EYE]
        right_eye_landmarks = [landmarks[i] for i in RIGHT_EYE]
        left_ear = self.eye_aspect_ratio(left_eye_landmarks)
        right_ear = self.eye_aspect_ratio(right_eye_landmarks)
        left_eye_closed = left_ear < EAR_THRESHOLD
        right_eye_closed = right_ear < EAR_THRESHOLD
        return left_eye_closed, right_eye_closed
    # ===================================================

    def mouth_aspect_ratio(self, landmarks):
        """Calculate mouth aspect ratio for yawn detection using MediaPipe landmarks."""
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

    def detect_yawn(self, landmarks):
        """
        Detect yawning based on mouth aspect ratio.
        landmarks: MediaPipe facial landmarks
        """
        mar = self.mouth_aspect_ratio(landmarks)
        yawn_detected = mar > YAWN_THRESHOLD
        
        yawn_confidence = mar / (YAWN_THRESHOLD * 1.5) if yawn_detected else 0
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
            (0.0, 0.0, 0.0),             
            (-225.0, 170.0, -135.0),    
            (225.0, 170.0, -135.0),     
            (-150.0, -150.0, -125.0),   
            (150.0, -150.0, -125.0),    
            (0.0, 0.0, -330.0)          
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

            distraction_detected = abs(yaw_deg) > DISTRACTION_ANGLE_THRESHOLD or abs(pitch_deg) > DISTRACTION_ANGLE_THRESHOLD

            direction = None
            if distraction_detected:
                if abs(yaw_deg) > abs(pitch_deg):
                    direction = 'right' if yaw_deg > 0 else 'left'
                else:
                    direction = 'up' if pitch_deg > 0 else 'down'
            return distraction_detected, direction, yaw_deg, pitch_deg, roll_deg
        return False, None, 0.0, 0.0, 0.0

    def check_camera_obstruction(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_frame)
        color_variance = np.std(frame)
        is_dark_obstructed = mean_intensity < BRIGHTNESS_THRESHOLD
        is_uniform_obstructed = color_variance < COLOR_VAR_THRESHOLD
        return is_dark_obstructed or is_uniform_obstructed

    def update_event_states(self, yawn_detected, distraction_detected_from_head_pose, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected):
        delta_time = 1 / self.metrics['fps'] if self.metrics['fps'] > 0 else 0

        if is_camera_obstructed:
            self.events['camera_obstruction']['active'] = True
            self.events['camera_obstruction']['timer'] += delta_time
            for event_name in self.events:
                if event_name != 'camera_obstruction':
                    self.events[event_name]['active'] = False
                    self.events[event_name]['timer'] = 0.0
            self.face_not_detected = False
            for direction in self.direction_timers:
                self.direction_timers[direction] = 0.0
            self.distraction_start_time = None
            self.fatigue_start_time = None
            self.yawning_start_time = None
            return
        else:
            self.events['camera_obstruction']['active'] = False
            self.events['camera_obstruction']['timer'] = 0.0
            
        self.face_not_detected = (faces_detected == 0)

        is_distracted = distraction_detected_from_head_pose or self.face_not_detected
        if is_distracted:
            if self.distraction_start_time is None:
                self.distraction_start_time = time.time()
            if (time.time() - self.distraction_start_time) >= DISTRACTION_TIME_THRESHOLD:
                self.events['distraction']['active'] = True
                self.events['distraction']['timer'] = time.time() - self.distraction_start_time

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

        if yawn_detected:
            if self.yawning_start_time is None:
                self.yawning_start_time = time.time()
            elif (time.time() - self.yawning_start_time) >= YAWN_TIME_THRESHOLD:
                self.events['yawning']['active'] = True
                self.events['yawning']['timer'] = time.time() - self.yawning_start_time
        else:
            self.yawning_start_time = None
            self.events['yawning']['active'] = False
            self.events['yawning']['timer'] = 0.0

        if left_eye_closed and right_eye_closed:
            if self.fatigue_start_time is None:
                self.fatigue_start_time = time.time()

            closed_time = time.time() - self.fatigue_start_time
            self.events['fatigue']['timer'] = closed_time

            if closed_time >= FATIGUE_TIME:
                self.events['fatigue']['active'] = True
            else:
                self.events['fatigue']['active'] = False

        else:
            self.fatigue_start_time = None
            self.events['fatigue']['active'] = False
            self.events['fatigue']['timer'] = 0.0

        self.events['left_eye_closed']['active'] = left_eye_closed
        self.events['left_eye_closed']['timer'] = (self.events['left_eye_closed']['timer'] + delta_time) if left_eye_closed else 0.0

        self.events['right_eye_closed']['active'] = right_eye_closed
        self.events['right_eye_closed']['timer'] = (self.events['right_eye_closed']['timer'] + delta_time) if right_eye_closed else 0.0

    def draw_dms_events_panel(self, frame):
        panel_x, panel_y, events_panel_height = self.video_width, 0, 350
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + events_panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + events_panel_height), (0, 255, 0), 2)
        label_text = "DMS Events"
        (text_w, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_x = panel_x + (self.panel_width - text_w) // 2
        cv2.putText(frame, label_text, (label_x, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        events_list = [
            ('smoking', 'Smoking'), ('yawning', 'Yawning'), ('camera_obstruction', 'Obstruction'),
            ('seatbelt_off', 'S_OFF'), ('phone_uses', 'Phone'), ('wearing_mask', 'Mask'),
            ('fatigue', 'Fatigue'), ('wearing_seatbelt', 'S_ON'), ('distraction', 'Distraction'),
            ('eating_drinking', 'Eating/Drinking'),
        ]
        icon_start_y, icon_size, row_spacing, icon_spacing_x = panel_y + 80, 60, 100, 90
        # Corrected loop to properly unpack tuples
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

        face_obstruction_text = ""
        if self.events['camera_obstruction']['active']:
            face_obstruction_text = "Camera is Obstructed"
        elif self.face_not_detected:
            face_obstruction_text = "Face is not Detected"

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

        if face_obstruction_text:
            params.append(f"Face Obstruction: {face_obstruction_text}")

        for param in params:
            cv2.putText(frame, param, (panel_x + 10, panel_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_offset += 20

    def run(self):
        print("Starting Edgetensor DMM Monitor...")
        print("Press 'q' to quit")

        # Define ROI coordinates based on roi_position
        if self.roi_position == "center":
            self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = 200, 150, 600, 600
        elif self.roi_position == "left":
            self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = 50, 150, 450, 600
        elif self.roi_position == "right":
            self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = 500, 150, 900, 600
        else:
            self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = 200, 150, 600, 600

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
            self.face_not_detected = False

            if not is_camera_obstructed:
                # Crop the frame to the ROI before processing with MediaPipe
                roi_frame = video_resized[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
                rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

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
                        mp_drawing.draw_landmarks(
                            image=video_resized,
                            landmark_list=full_frame_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        # Draw eyes
                        mp_drawing.draw_landmarks(
                            image=video_resized,
                            landmark_list=full_frame_landmarks,
                            connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        mp_drawing.draw_landmarks(
                            image=video_resized,
                            landmark_list=full_frame_landmarks,
                            connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        # Draw lips (outer and inner)
                        mp_drawing.draw_landmarks(
                            image=video_resized,
                            landmark_list=full_frame_landmarks,
                            connections=mp_face_mesh.FACEMESH_LIPS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        # Draw nose (approximated by key points)
                        for index in NOSE_LANDMARKS:
                           if len(full_frame_landmarks.landmark) > index:
                               landmark = full_frame_landmarks.landmark[index]
                               x, y = int(landmark.x * self.video_width), int(landmark.y * self.video_height)
                               cv2.circle(video_resized, (x, y), 2, (0, 255, 0), -1)

                        landmarks = full_frame_landmarks.landmark
                        yawn_detected, _ = self.detect_yawn(landmarks)
                        distraction_detected_from_head_pose, head_direction, self.head_pose_yaw, self.head_pose_pitch, self.head_pose_roll = self.detect_head_pose(landmarks)
                        left_eye_closed, right_eye_closed = self.detect_eye_closure(landmarks)
                else:
                    self.face_not_detected = True

            self.update_event_states(yawn_detected, distraction_detected_from_head_pose, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected)

            self.draw_dms_events_panel(display_frame)
            self.draw_dms_output_panel(display_frame)

            now = time.time()
            self.metrics['fps'] = 1.0 / (now - self.last_time)
            self.metrics['frame_processing_time'] = (now - frame_start_time) * 1000
            self.last_time = now

            cv2.imshow("Edgetensor DMM Monitor", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edgetensor Driver Monitoring System")
    parser.add_argument("--roi", type=str, default="center", choices=["left", "right", "center"],
                        help="Specify the ROI position (e.g., 'left', 'right', 'center'). Default is 'center'.")
    args = parser.parse_args()
    try:
        monitor = EdgetensorDMMMonitor(roi_position=args.roi)
        monitor.run()
    except Exception as e:
        print(f"An error occurred: {e}")
