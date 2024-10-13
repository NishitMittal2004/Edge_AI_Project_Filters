import cv2
import time
import math
import HandTrackingModule as htm

# Set up camera and parameters
wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(10, 450)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

# Face detection with Haar Cascades
face_cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

# Filter functions (same as before)
def blur_face(img, face_coords):
    for (x, y, w, h) in face_coords:
        face_roi = img[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        img[y:y+h, x:x+w] = blurred_face
    return img

def glow_face(img, face_coords):
    for (x, y, w, h) in face_coords:
        face_roi = img[y:y+h, x:x+w]
        glow_effect = cv2.convertScaleAbs(face_roi, alpha=1.5, beta=30)
        img[y:y+h, x:x+w] = glow_effect
    return img

def grayscale_face(img, face_coords):
    for (x, y, w, h) in face_coords:
        face_roi = img[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        img[y:y+h, x:x+w] = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
    return img

def pixelate_face(img, face_coords):
    for (x, y, w, h) in face_coords:
        face_roi = img[y:y+h, x:x+w]
        temp = cv2.resize(face_roi, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR)
        pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y:y+h, x:x+w] = pixelated_face
    return img

def neon_face(img, face_coords):
    for (x, y, w, h) in face_coords:
        face_roi = img[y:y+h, x:x+w]
        edges = cv2.Canny(face_roi, 100, 200)
        neon_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        neon_edges[:, :, 0] = 0
        neon_edges[:, :, 1] = 255
        img[y:y+h, x:x+w] = cv2.addWeighted(face_roi, 0.7, neon_edges, 0.3, 0)
    return img

def edge_highlight_face(img, face_coords):
    for (x, y, w, h) in face_coords:
        face_roi = img[y:y+h, x:x+w]
        edges = cv2.Canny(face_roi, 100, 200)
        img[y:y+h, x:x+w] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return img

# Initialize the filter list
filters = [blur_face, glow_face, grayscale_face, pixelate_face, neon_face, edge_highlight_face]
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

    # Apply the current filter on detected faces
    if len(faces) > 0:
        img = filters[filter_index](img, faces)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.rectangle(img, (490, 10), (633, 50), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, f'FPS: {int(fps)}', (500, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 100), 2)

    # Show the final image with filters
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
