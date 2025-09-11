# face_capture.py
import cv2
from deepface import DeepFace
from sklearn.cluster import KMeans
import numpy as np
import pyttsx3 # Import the text-to-speech library

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Set speech rate

def speak(text):
    """Function to generate a voice alert."""
    engine.say(text)
    engine.runAndWait()

def capture_faces(angles=["front", "left", "right", "up", "down"]):
    """
    Capture multiple face images for different angles and return a clustered embedding.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    embeddings = []
    for angle in angles:
        # Voice alert to instruct the user
        speak(f"Please turn your face to the {angle}")
        print(f"Turn your face: {angle}")
        
        captured = False
        while not captured:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Display instructions on the video frame
            instruction_text = f"Turn face {angle}, press 'c'"
            cv2.putText(frame, instruction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Capture - Press 'c' to capture", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                try:
                    emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]
                    embeddings.append(emb)
                    captured = True
                    print(f"{angle} captured successfully")
                except Exception as e:
                    print(f"Face not detected, try again: {e}")

    cap.release()
    cv2.destroyAllWindows()

    # If no embeddings were captured, return an empty list or handle the error
    if not embeddings:
        print("No faces were captured. Registration failed.")
        return None

    # Cluster embeddings into 1 representative vector
    kmeans = KMeans(n_clusters=1, random_state=0)
    kmeans.fit(embeddings)
    clustered_embedding = kmeans.cluster_centers_[0]

    return clustered_embedding.tolist()
