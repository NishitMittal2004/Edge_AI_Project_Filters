import cv2
import numpy as np
import os
import time

# Load the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Capture video from the webcam  
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Path to the folder containing goggle images
goggle_folder = 'goggles'

# Get the list of goggle images in the folder
goggle_images = [os.path.join(goggle_folder, img) for img in os.listdir(goggle_folder) if img.endswith('.jpg')]

if not goggle_images:
    print("Error: No goggle images found in the folder.")
    exit()

current_goggle_idx = 0  # Index to track the current goggle image
face_out_of_frame = True  # Track whether the face has left the frame

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    centers = []

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if a face has been detected
    if len(faces) > 0:
        # If the face was out of the frame, switch the goggle image
        if face_out_of_frame:
            current_goggle_idx = (current_goggle_idx + 1) % len(goggle_images)
            face_out_of_frame = False  # Mark that the face is now in the frame
    else:
        # If no faces are detected, mark that the face is out of the frame
        face_out_of_frame = True

    for (x, y, w, h) in faces:
        # Create regions of interest (ROI)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Store the coordinates of eyes
        for (ex, ey, ew, eh) in eyes:
            centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))

    if len(centers) > 1:
        # Load the current goggle image
        glass_img = cv2.imread(goggle_images[current_goggle_idx])
        
        # Calculate the width of the glasses based on eye distance
        glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
        overlay_img = np.ones(frame.shape, np.uint8) * 255
        h, w = glass_img.shape[:2]
        scaling_factor = glasses_width / w

        # Resize the glasses image
        overlay_glasses = cv2.resize(glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Adjust x, y positions based on the face size
        x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
        x -= 0.26 * overlay_glasses.shape[1]
        y += 0.85 * overlay_glasses.shape[0]

        # Overlay the glasses on the face
        h, w = overlay_glasses.shape[:2]
        overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses

        # Create a mask and its inverse
        gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_glasses, 110, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Use the mask to create the final image with glasses
        temp = cv2.bitwise_and(frame, frame, mask=mask)
        temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
        final_img = cv2.add(temp, temp2)

        # Display the result
        cv2.imshow('Lets wear Glasses', final_img)
    else:
        # Display the frame without modification if no faces/eyes detected
        cv2.imshow('Lets wear Glasses', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
