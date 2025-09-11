import tkinter as tk
from tkinter import simpledialog, messagebox
import json
import os
from datetime import datetime
from face_capture import capture_faces
from face_recognize import recognize_person_on_frame
from updatetimelog import start_event_detection # Import the new function

DATA_FILE = "faces.json"

# Load existing database
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

def register_person():
    """
    Register a new person: ask name, capture multiple face angles,
    cluster embeddings, and save to a database.
    """
    name = simpledialog.askstring("Name", "Enter person's name:")
    if not name:
        return

    person_id = str(int(datetime.now().timestamp()))  # Unique ID

    try:
        embedding = capture_faces()  # Capture and cluster embeddings
        face_db[person_id] = {"name": name, "embedding": embedding}

        with open(DATA_FILE, "w") as f:
            json.dump(face_db, f, indent=2)

        messagebox.showinfo("Success", f"{name} registered successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to register face: {str(e)}")

def recognize_ui():
    """
    Starts face recognition. If a known face is found, it proceeds to event detection.
    Otherwise, it stops.
    """
    if not face_db:
        messagebox.showwarning("Warning", "No registered faces! Please register a face first.")
        return
        
    # Start recognition and get the result (known name or "Unknown")
    recognized_name = recognize_person_on_frame(face_db)

    # If a known face is recognized, proceed to the next step
    if recognized_name != "Unknown" and recognized_name != "No face detected" and recognized_name != "Quit":
        messagebox.showinfo("Success", f"Welcome, {recognized_name}! Starting event detection.")
        start_event_detection() # Call the event detection function
    else:
        messagebox.showwarning("Recognition Failed", "Face not recognized. The program will not proceed to event detection.")

def run_ui():
    """
    Launch the Tkinter UI for face registration and recognition.
    """
    root = tk.Tk()
    root.title("Face Registration & Recognition")
    root.geometry("350x200")

    # Buttons for registration and recognition
    tk.Button(root, text="Register New Face", command=register_person).pack(pady=10)
    tk.Button(root, text="Recognize Face", command=recognize_ui).pack(pady=10)
    tk.Button(root, text="Exit", command=root.destroy).pack(pady=10)

    root.mainloop()
