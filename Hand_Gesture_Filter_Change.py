import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm

# Set up camera and parameters
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(10, 450)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

# Face detection with Haar Cascades
face_cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

# Cartoonify function
prev_edges = None

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

# Initialize the filter list
filters = [cartoonify_wholeImage, blur_face, glow_face, grayscale_face, pixelate_face, edge_highlight_face]
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
