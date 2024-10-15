import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm
import mediapipe as mp
import os

# Set up camera and parameters
wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(10, 450)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
current_goggle_idx = 0  # Index to track the current goggle image
face_out_of_frame = True  # Track whether the face has left the frame

# Face detection with Haar Cascades
face_cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Resources/haarcascade_eye.xml')

# Load the fire, rain, and flower videos
fire_effect = cv2.VideoCapture('Resources/fire.mp4')   # Path to fire video
rain_effect = cv2.VideoCapture('Resources/rain.mp4')   # Path to rain video
flower_effect = cv2.VideoCapture('Resources/flower.mp4')  # Path to flower video

# Path to the folder containing goggle images
goggle_folder = 'Resources/goggles'

# Get the list of goggle images in the folder
goggle_images = [os.path.join(goggle_folder, img) for img in os.listdir(goggle_folder) if img.endswith('.jpg')]

# Cartoonify function
prev_edges = None


# Filter Functions
def apply_goggles_filter(img):
    global current_goggle_idx, face_out_of_frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centers = []

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if a face has been detected
    if len(faces) > 0:
        if face_out_of_frame:
            current_goggle_idx = (current_goggle_idx + 1) % len(goggle_images)
            face_out_of_frame = False
    else:
        face_out_of_frame = True

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Store the coordinates of eyes
        for (ex, ey, ew, eh) in eyes:
            centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))

    if len(centers) > 1:
        glass_img = cv2.imread(goggle_images[current_goggle_idx])

        # Calculate the width of the glasses based on eye distance
        glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
        overlay_img = np.ones(img.shape, np.uint8) * 255
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
        temp = cv2.bitwise_and(img, img, mask=mask)
        temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
        final_img = cv2.add(temp, temp2)

        return final_img
    return img

def cartoonify_wholeImage(img):
    global prev_edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges_canny = cv2.Canny(gray_blur, 70, 150)
    edges_laplacian = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize=5)
    combined_edges = cv2.bitwise_or(edges_canny, edges_laplacian)
    kernel = np.ones((3, 3), np.uint8)
    combined_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    combined_edges = cv2.GaussianBlur(combined_edges, (3, 3), 0)

    if prev_edges is None:
        prev_edges = np.zeros_like(combined_edges)  # Initialize with zeros

    if combined_edges.shape != prev_edges.shape:
        prev_edges = cv2.resize(prev_edges, combined_edges.shape[::-1])  # Resize to match

    combined_edges = cv2.addWeighted(combined_edges, 0.8, prev_edges, 0.2, 0)
    prev_edges = combined_edges.copy()

    color = img
    for i in range(0, 2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=100, sigmaSpace=100)

    edges_inv = cv2.bitwise_not(combined_edges)
    edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
    cartoon_image = cv2.bitwise_and(color, edges_colored)

    return cartoon_image

# Filter functions
def blur_face(img, face_coords):
    for (x, y, w, h) in face_coords:
        face_roi = img[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        img[y:y+h, x:x+w] = blurred_face
    return img


def glow_face(img, face_coords):
    # Create a darkened version of the original image
    darkened_img = cv2.convertScaleAbs(img, alpha=0.5, beta=0)  # Darken the entire image

    for (x, y, w, h) in face_coords:
        face_roi = img[y:y + h, x:x + w]
        glow_effect = cv2.convertScaleAbs(face_roi, alpha=1.5, beta=30)

        # Place the glow effect on the darkened image
        darkened_img[y:y + h, x:x + w] = glow_effect

    return darkened_img

def grayscale_face(img):
    gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_face


def pixelate_face(img, face_coords):
    if len(face_coords) > 1:  # Only apply if more than one face detected
        # Find the largest face based on area
        largest_face = max(face_coords, key=lambda face: face[2] * face[3])  # (x, y, w, h)

        for (x, y, w, h) in face_coords:
            # Check if the current face is not the largest face
            if (x, y, w, h) != (largest_face[0], largest_face[1], largest_face[2], largest_face[3]):
                face_roi = img[y:y + h, x:x + w]
                # Pixelate the smaller face
                temp = cv2.resize(face_roi, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR)
                pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                img[y:y + h, x:x + w] = pixelated_face

    return img

def edge_highlight_face(img):
    edges = cv2.Canny(img, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# New Dog Filter
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

tongue_img = cv2.imread("Resources/doggy_tongue.png", cv2.IMREAD_UNCHANGED)
dog_ears_img = cv2.imread("Resources/doggy_ears.png", cv2.IMREAD_UNCHANGED)
dog_nose_img = cv2.imread("Resources/doggy_nose.png", cv2.IMREAD_UNCHANGED)


def dog_filter(img):
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_landmark = face_landmarks.landmark[1]
            mouth_bottom_landmark = face_landmarks.landmark[17]  # Lower lip
            mouth_top_landmark = face_landmarks.landmark[0]  # Upper lip
            nose_coords = (int(nose_landmark.x * img.shape[1]), int(nose_landmark.y * img.shape[0]))
            mouth_bottom_coords = (
            int(mouth_bottom_landmark.x * img.shape[1]), int(mouth_bottom_landmark.y * img.shape[0]))
            mouth_top_coords = (int(mouth_top_landmark.x * img.shape[1]), int(mouth_top_landmark.y * img.shape[0]))
            face_width = int(abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * img.shape[1])

            # Add ears
            ears_height = int(face_width * (dog_ears_img.shape[0] / dog_ears_img.shape[1]))
            resized_ears = cv2.resize(dog_ears_img, (face_width, ears_height))
            face_top_x = int(face_landmarks.landmark[10].x * img.shape[1])
            face_top_y = int(face_landmarks.landmark[10].y * img.shape[0]) - ears_height
            ears_x = face_top_x - face_width // 2
            ears_y = max(0, face_top_y)
            if resized_ears.shape[2] == 4:
                alpha_channel = resized_ears[:, :, 3] / 255.0
                for c in range(0, 3):
                    img[ears_y:ears_y + ears_height, ears_x:ears_x + face_width, c] = (
                                alpha_channel * resized_ears[:, :, c] +
                                (1 - alpha_channel) * img[ears_y:ears_y + ears_height, ears_x:ears_x + face_width, c])

            # Add nose
            nose_width = int(face_width * 0.5)
            nose_height = int(nose_width * (dog_nose_img.shape[0] / dog_nose_img.shape[1]))
            resized_nose = cv2.resize(dog_nose_img, (nose_width, nose_height))
            nose_x = int(nose_coords[0] - nose_width // 2)
            nose_y = int(nose_coords[1] - nose_height // 2)
            if resized_nose.shape[2] == 4:
                alpha_channel = resized_nose[:, :, 3] / 255.0
                for c in range(0, 3):
                    img[nose_y:nose_y + nose_height, nose_x:nose_x + nose_width, c] = (
                                alpha_channel * resized_nose[:, :, c] +
                                (1 - alpha_channel) * img[nose_y:nose_y + nose_height, nose_x:nose_x + nose_width, c])

            # Add tongue
            mouth_top_landmark = np.array([mouth_top_landmark.x, mouth_top_landmark.y])
            mouth_bottom_landmark = np.array([mouth_bottom_landmark.x, mouth_bottom_landmark.y])
            lip_distance = np.linalg.norm(mouth_top_landmark - mouth_bottom_landmark)
            lip_threshold = 0.045

            if lip_distance > lip_threshold:

                tongue_width = int(face_width * 0.4)
                tongue_height = int(tongue_width * (tongue_img.shape[0] / tongue_img.shape[1]))
                resized_tongue = cv2.resize(tongue_img, (tongue_width, tongue_height))
                tongue_x = int(mouth_bottom_coords[0] - tongue_width // 2)
                tongue_y = int(mouth_bottom_coords[1] - 15)
                if resized_tongue.shape[2] == 4:
                    alpha_channel = resized_tongue[:, :, 3] / 255.0
                    for c in range(0, 3):
                        img[tongue_y:tongue_y + tongue_height, tongue_x:tongue_x + tongue_width, c] = (
                                    alpha_channel * resized_tongue[:, :, c] +
                                    (1 - alpha_channel) * img[tongue_y:tongue_y + tongue_height,
                                                          tongue_x:tongue_x + tongue_width, c])
    return img


def emotion_filter(frame):
    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face landmarks
    results = face_mesh.process(rgb_frame)

    # If landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmark points for the mouth and eyebrows
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            left_eyebrow = face_landmarks.landmark[105]
            right_eyebrow = face_landmarks.landmark[282]

            # Calculate distances for smile, mouth openness, and eyebrow distance
            smile_width = math.sqrt((right_mouth.x - left_mouth.x) ** 2 + (right_mouth.y - left_mouth.y) ** 2)
            mouth_open = math.sqrt((upper_lip.x - lower_lip.x) ** 2 + (upper_lip.y - lower_lip.y) ** 2)
            eyebrow_distance = math.sqrt(
                (right_eyebrow.x - left_eyebrow.x) ** 2 + (right_eyebrow.y - left_eyebrow.y) ** 2)
            eyelid_eyebrow_distance = math.sqrt((left_eyebrow.x - face_landmarks.landmark[157].x) ** 2 +
                                                (right_eyebrow.y - face_landmarks.landmark[384].y) ** 2)

            # Set thresholds for different emotions
            smile_threshold = 0.05
            sad_threshold = 0.066
            anger_threshold = 0.069
            eyelid_eyebrow_distance_threshold = 0.035

            # Determine emotion
            if smile_width > smile_threshold and mouth_open > 0.01:
                emotion = 'Smiling'
                color = (0, 255, 0)
            elif smile_width < sad_threshold and mouth_open < 0.01:
                emotion = 'Sad'
                color = (255, 0, 0)
            elif eyelid_eyebrow_distance < eyelid_eyebrow_distance_threshold:
                emotion = 'Angry'
                color = (0, 0, 255)
            else:
                emotion = 'Neutral'
                color = (255, 255, 255)

            # Apply visual effects based on emotion
            if emotion == 'Angry':
                ret_fire, fire_frame = fire_effect.read()
                if ret_fire:
                    fire_frame = cv2.resize(fire_frame, (frame.shape[1], frame.shape[0]))
                    frame = cv2.addWeighted(fire_frame, 0.7, frame, 0.3, 0)

            elif emotion == 'Sad':
                ret_rain, rain_frame = rain_effect.read()
                if ret_rain:
                    rain_frame = cv2.resize(rain_frame, (frame.shape[1], frame.shape[0]))
                    frame = cv2.addWeighted(rain_frame, 0.5, frame, 0.5, 0)

            elif emotion == 'Smiling':
                ret_flower, flower_frame = flower_effect.read()
                if ret_flower:
                    flower_frame = cv2.resize(flower_frame, (frame.shape[1], frame.shape[0]))
                    frame = cv2.addWeighted(flower_frame, 0.7, frame, 0.3, 0)

            # Display the detected emotion on the screen
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return frame

# Initialize the filter list
filters = [cartoonify_wholeImage, blur_face, glow_face, grayscale_face, pixelate_face, edge_highlight_face, dog_filter, apply_goggles_filter, emotion_filter]
filter_index = 0  # Start with the first filter

# State variables to track finger gestures
pinky_was_open = False
index_was_open = False

while True:
    success, img = cap.read()

    # Hand tracking and position detection
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # Face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(lmList) != 0:
        fingers = []

        # Thumb logic
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[9][1], lmList[9][2]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > 50:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers logic
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        # Gesture control for switching filters
        if fingers == [0, 0, 0, 0, 1]:  # Pinky is raised (next filter)
            if not pinky_was_open:  # Only switch if pinky was previously closed
                filter_index = (filter_index + 1) % len(filters)
                pinky_was_open = True  # Mark pinky as open to prevent multiple triggers
        else:
            pinky_was_open = False  # Reset when pinky is closed

        if fingers == [1, 0, 0, 0, 0]:  # Thumb finger is raised (previous filter)
            if not index_was_open:  # Only switch if index was previously closed
                filter_index = (filter_index - 1) % len(filters)
                index_was_open = True  # Mark index as open to prevent multiple triggers
        else:
            index_was_open = False  # Reset when index is closed

        # Display the active filter
        cv2.rectangle(img, (20, 25), (170, 75), (156, 83, 0), cv2.FILLED)
        cv2.putText(img, f'Filter: {filter_index + 1}', (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 164, 238), 2)

    # Apply the filters based on the index
    if len(faces) > 0:
        if filter_index in [1, 2]:  # Blur and Glow filters only on faces
            img = filters[filter_index](img, faces)
        elif filter_index == 4:  # Pixelate on other faces if more than one face detected
            img = filters[filter_index](img, faces)
        else:  # Apply all other filters to the whole frame
            img = filters[filter_index](img)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.rectangle(img, (490, 10), (633, 50), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, f'FPS: {int(fps)}', (500, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 100), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
