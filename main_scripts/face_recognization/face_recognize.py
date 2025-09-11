import cv2
import numpy as np
import time
import pyttsx3
from deepface import DeepFace

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Set speech rate

def speak(text):
    """Function to generate a voice alert."""
    engine.say(text)
    engine.runAndWait()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_person_on_frame(face_db, similarity_threshold=0.75):
    """
    Recognize a person from live webcam with voice alerts.
    Returns the recognized name or "Unknown" / "No face detected".
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    prev_time = 0  # For FPS calculation
    last_spoken_name = None  # To prevent repeated voice alerts

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        recognized_name = "Unknown"
        is_known_face = False

        try:
            result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)
            emb = np.array(result[0]["embedding"])

            max_sim = -1
            best_match_name = "Unknown"
            
            for pid, data in face_db.items():
                db_emb = np.array(data["embedding"])
                sim = cosine_similarity(emb, db_emb)
                
                if sim > max_sim:
                    max_sim = sim
                    best_match_name = data["name"]

            if max_sim >= similarity_threshold:
                recognized_name = best_match_name
                is_known_face = True
            else:
                recognized_name = "Unknown"

        except Exception as e:
            recognized_name = "No face detected"
            
        # Voice Alert Logic
        if recognized_name != last_spoken_name:
            if recognized_name != "Unknown" and recognized_name != "No face detected":
                speak(f"Welcome, {recognized_name}")
            elif recognized_name == "Unknown":
                speak("Face is not recognized.")
            
            last_spoken_name = recognized_name

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Display recognized name (top-right) and FPS (top-left)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        name_text = f"{recognized_name}"
        text_size = cv2.getTextSize(name_text, font, font_scale, thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = 30
        cv2.putText(frame, name_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), font, font_scale, (0, 255, 255), thickness)

        cv2.imshow("Face Recognition", frame)
        
        # Break loop if 'q' is pressed or a known face is detected
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recognized_name = "Quit"
            break
        if is_known_face:
            time.sleep(1) # Small delay to show the name
            break

    cap.release()
    cv2.destroyAllWindows()
    return recognized_name
