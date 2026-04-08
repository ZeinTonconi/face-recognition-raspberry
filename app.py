"""
tk_face_ui.py — Tkinter UI for live face recognition + registration

Features:
  - Always live face recognition
  - Left-side control panel with:
      - Add Person (modal + 5 poses)
      - Data Augmentation
      - Generate Embeddings
      - Train Model
  - Registration mode shows thumbnails of captured poses
  - After Accept, returns to main recognition
"""

import os
import sys
import cv2
import threading
import time
import tkinter as tk
from tkinter import Toplevel, Entry, Label, Button, Frame, Canvas, PhotoImage

from PIL import Image, ImageTk
from recognize_live import main as recognize_main  # Or import functions
from capture_faces import POSES, POSE_INSTRUCTIONS, crop_face, detect_faces, select_tracked_box, RAW_DIR

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class FaceApp:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition System")

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.recognizing = True  # Recognition loop flag

        # UI Layout
        self.left_panel = Frame(root, width=200, bg="#d3d3d3")
        self.left_panel.pack(side="left", fill="y")

        self.camera_canvas = Canvas(root, width=640, height=480)
        self.camera_canvas.pack(side="right")

        # Buttons
        Button(self.left_panel, text="Agregar Persona", command=self.add_person_modal).pack(pady=10)
        Button(self.left_panel, text="Data Augmentation", command=self.data_augmentation).pack(pady=10)
        Button(self.left_panel, text="Generar Embeddings", command=self.generate_embeddings).pack(pady=10)
        Button(self.left_panel, text="Entrenar modelo", command=self.train_model).pack(pady=10)

        # Thumbnails for registration
        self.thumb_frames = []
        self.thumbs_canvas = Frame(self.left_panel, bg="#f0f0f0")
        self.thumbs_canvas.pack(pady=20)
        for i in range(5):
            f = Frame(self.thumbs_canvas, width=80, height=80, bg="white", relief="ridge", bd=2)
            f.grid(row=i//2, column=i%2, padx=5, pady=5)
            self.thumb_frames.append(f)

        # Recognition thread
        self.update_loop()

    # -----------------------------------------------------------------------
    # Camera Loop
    # -----------------------------------------------------------------------
    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame.copy()
            # Only recognize if not in registration
            if self.recognizing:
                boxes = detect_faces(frame)
                for box in boxes:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Convert to Tkinter image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_canvas.imgtk = imgtk
            self.camera_canvas.create_image(0, 0, anchor="nw", image=imgtk)

        self.root.after(30, self.update_loop)

    # -----------------------------------------------------------------------
    # Add Person Modal
    # -----------------------------------------------------------------------
    def add_person_modal(self):
        modal = Toplevel(self.root)
        modal.title("Agregar Persona")
        Label(modal, text="Nombre:").pack(pady=5)
        name_entry = Entry(modal)
        name_entry.pack(pady=5)

        def start_registration():
            name = name_entry.get().strip()
            if not name:
                return
            modal.destroy()
            self.recognizing = False
            self.register_person(name)

        Button(modal, text="Aceptar", command=start_registration).pack(side="left", padx=10, pady=10)
        Button(modal, text="Cancelar", command=modal.destroy).pack(side="right", padx=10, pady=10)

    # -----------------------------------------------------------------------
    # Registration Loop
    # -----------------------------------------------------------------------
    def register_person(self, name: str):
        person_dir = os.path.join(RAW_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        for idx, pose in enumerate(POSES):
            captured = False
            while not captured:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                boxes = detect_faces(frame)
                tracked_box = select_tracked_box(boxes, None)
                if tracked_box:
                    x, y, w, h = tracked_box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, pose, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255,0,0), 2)
                    cv2.imshow("Pose Capture", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (32, ord("c")) and tracked_box:
                    crop = crop_face(frame, tracked_box)
                    save_path = os.path.join(person_dir, f"{name}_{pose.lower()}.jpg")
                    cv2.imwrite(save_path, crop)
                    self.update_thumbnail(idx, crop)
                    captured = True
                if key == 27:
                    cv2.destroyWindow("Pose Capture")
                    self.recognizing = True
                    return

        cv2.destroyWindow("Pose Capture")
        self.recognizing = True

    # -----------------------------------------------------------------------
    # Update thumbnail panel
    # -----------------------------------------------------------------------
    def update_thumbnail(self, idx, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((80, 80))
        imgtk = ImageTk.PhotoImage(img_pil)
        label = tk.Label(self.thumb_frames[idx], image=imgtk)
        label.imgtk = imgtk
        label.pack()

    # -----------------------------------------------------------------------
    # Placeholder functions
    # -----------------------------------------------------------------------
    def data_augmentation(self):
        print("Data augmentation triggered")

    def generate_embeddings(self):
        print("Generate embeddings triggered")

    def train_model(self):
        print("Train model triggered")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()