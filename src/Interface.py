import pickle
import cv2
import tkinter as tk
from tkinter import filedialog
import joblib

# Load the model from the .pkl file
with open('knn_model.pkl', 'rb') as file:
    model = joblib.load(file)

# Preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (2, 7))
    img_normalized = img_resized / 255.0
    img_flattened = img_normalized.flatten()  # Flatten the image array
    return img_flattened.reshape(1, -1)  # Reshape to 2-dimensional array

# Predict the eye state
def predict_eye_state(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    eye_state = 'Open' if prediction[0] > 0.5 else 'Closed'
    return eye_state

# Browse for an image and predict the eye state
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if file_path:
        eye_state = predict_eye_state(file_path)
        result_label.config(text=f"Eye state: {eye_state}")

# Create the main window
root = tk.Tk()
root.title("Eye State Predictor")

# Create a button to browse for an image
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(root, text="Eye state: N/A")
result_label.pack(pady=10)

# Run the main loop
root.mainloop()
