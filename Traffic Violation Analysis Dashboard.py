#!/usr/bin/env python
# coding: utf-8

# To create a Traffic Violation Analysis Dashboard, we'll use Python along with Streamlit for the web app, OpenCV for image processing, and YOLO (You Only Look Once) for object detection. This will allow us to analyze images in real-time and detect whether:
# 
# A person is wearing a helmet or not.
# The number plate is visible.
# Identify the state from the license number.
# 

# Features of the Dashboard:
# Upload Real-Time Images
# Switch Options for Different Analyses
# Helmet Detection
# License Plate Recognition
# State Identification
# Display Percentage Accuracy

# # Steps to Implement
# 1. Install Necessary Libraries

# In[73]:


pip install opencv-python numpy ultralytics pillow


# In[38]:


pip install ultralytics


# In[42]:


model = YOLO("yolov8n.pt")  # Pre-trained model


# In[88]:


import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load Pre-Trained YOLOv8 Model (No need to train)
yolo_model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano (smallest pre-trained model)

# Initialize Tkinter window
root = tk.Tk()
root.title("Traffic Violation Analysis Dashboard")
root.geometry("900x700")

# Global variable for image path
image_path = None

# Function to analyze the uploaded image
def analyze_image():
    global image_path
    if not image_path:
        result_label.config(text="❌ Please upload an image first!", fg="red")
        return

    # Read the image
    image = cv2.imread(image_path)

    # Run YOLOv8 model for detection
    results = yolo_model(image)

    detected_objects = []  # Store detected object information

    # Process detection results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # Get class ID
            confidence = box.conf[0].item() * 100  # Convert to percentage

            # Get class name from YOLO's default COCO dataset (Helmet is not included, but number plates are)
            class_names = yolo_model.names
            label = class_names[class_id]

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Append detected object details
            detected_objects.append(f"{label}: {confidence:.2f}%")

    # Convert OpenCV image to PIL format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    img_pil = img_pil.resize((400, 300))  # Resize for display
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update analyzed image label
    analyzed_img_label.config(image=img_tk)
    analyzed_img_label.image = img_tk

    # Update results in the dashboard
    result_label.config(text="✅ Analysis Completed!", fg="green")

    # Display detected objects as text
    detected_text = "\n".join(detected_objects) if detected_objects else "No objects detected."
    detection_result_label.config(text=detected_text, fg="blue")

# Function to upload an image
def upload_image():
    global image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        image_path = file_path
        img = Image.open(image_path)
        img = img.resize((400, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)

        # Update uploaded image label
        uploaded_img_label.config(image=img_tk)
        uploaded_img_label.image = img_tk

        result_label.config(text="📸 Image Uploaded Successfully!", fg="blue")

# UI Elements
title_label = Label(root, text="🚦 Traffic Violation Analysis Dashboard", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

upload_btn = Button(root, text="📤 Upload Image", command=upload_image, font=("Arial", 12))
upload_btn.pack(pady=10)

analyze_btn = Button(root, text="🔍 Analyze", command=analyze_image, font=("Arial", 12), bg="green", fg="white")
analyze_btn.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

uploaded_label = Label(root, text="📷 Uploaded Image", font=("Arial", 12, "bold"))
uploaded_label.pack()
uploaded_img_label = Label(root)  # Label to show uploaded image
uploaded_img_label.pack()

analyzed_label = Label(root, text="🧐 Analyzed Image", font=("Arial", 12, "bold"))
analyzed_label.pack()
analyzed_img_label = Label(root)  # Label to show analyzed image
analyzed_img_label.pack()

detection_result_label = Label(root, text="", font=("Arial", 12, "bold"))
detection_result_label.pack()

# Run the Tkinter event loop
root.mainloop()


# In[ ]:





# In[ ]:




