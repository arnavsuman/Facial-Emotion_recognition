import cv2
import dlib
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from fer import FER
from PIL import Image
import io

# Initialize the FER detector
detector = FER(mtcnn=True)

# Function to create a polar chart image from emotion scores
def create_polar_chart(emotions):
    # Define the emotions and their corresponding angles
    labels = list(emotions.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    # The values of the emotions and repeat the first value for a complete loop
    values = list(emotions.values())
    values += values[:1]  # Repeat the first value to close the loop
    angles += angles[:1]  # Repeat the first angle to close the loop

    # Create a polar chart
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)

    # Set the labels
    ax.set_yticklabels([])  # Hide radial labels
    ax.set_xticks(angles[:-1])  # Set emotion labels
    ax.set_xticklabels(labels)

    # Save the polar chart to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close()
    buf.seek(0)

    # Open the image with PIL and convert to OpenCV format
    polar_image = Image.open(buf)
    polar_image = polar_image.convert("RGBA")
    return polar_image

# Function to overlay an image with alpha channel onto another image
def overlay_image_alpha(img, img_overlay, pos=(0, 0), alpha_mask=None):
    x, y = pos

    # Image ranges
    y1, y2 = max(y, 0), min(y + img_overlay.shape[0], img.shape[0])
    x1, x2 = max(x, 0), min(x + img_overlay.shape[1], img.shape[1])

    # Overlay ranges
    y1o, y2o = max(-y, 0), min(img.shape[0] - y, img_overlay.shape[0])
    x1o, x2o = max(-x, 0), min(img.shape[1] - x, img_overlay.shape[1])

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    if alpha_mask is None:
        alpha = img_overlay[y1o:y2o, x1o:x2o, 3] / 255.0
    else:
        alpha = alpha_mask[y1o:y2o, x1o:x2o] / 255.0

    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                  alpha_inv * img[y1:y2, x1:x2, c])

# Load Dlib's face detector
detector_dlib = dlib.get_frontal_face_detector()

# Specify the path to the shape predictor model
model_path = 'shape_predictor_68_face_landmarks.dat'  # Update this path as needed

# Check if the model file exists
if not os.path.isfile(model_path):
    print(f"Model file not found at {model_path}. Please check the path.")
    exit()

# Load the shape predictor
predictor = dlib.shape_predictor(model_path)

# Load the image file
image_path = '4.jpg'  # Update this to your image file path
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not read image from {image_path}.")
    exit()

# Detect emotions in the image
emotions_detected = detector.detect_emotions(frame)

# If faces are detected
if emotions_detected:
    for face in emotions_detected:
        emotions = face["emotions"]
        # Create polar chart
        polar_img = create_polar_chart(emotions)

        # Convert PIL image to OpenCV format
        polar_cv = cv2.cvtColor(np.array(polar_img), cv2.COLOR_RGBA2BGRA)

        # Define position to place the polar chart (e.g., top-left corner of the face)
        (x, y, w, h) = face["box"]
        pos_x = x + w + 10  # 10 pixels to the right of the face
        pos_y = y

        # Overlay the polar chart on the frame
        overlay_image_alpha(frame, polar_cv, pos=(pos_x, pos_y))

        # Optionally, draw the bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get landmarks for the face
        landmarks = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dlib.rectangle(x, y, x + w, y + h))

        # Create a list to store random points
        points = []

        # Generate 200 random points within the bounding box of the face
        for _ in range(200):
            rand_x = random.randint(x, x + w)
            rand_y = random.randint(y, y + h)
            points.append((rand_x, rand_y))

        # Draw the points on the frame
        for point in points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)  # Draw a green dot

# Display the resulting frame with the emotion chart
cv2.imshow('Facial Expression Recognition with Emotion Polar Chart', frame)
cv2.waitKey(0)  # Wait for a key press to close the window

# Release resources
cv2.destroyAllWindows()
