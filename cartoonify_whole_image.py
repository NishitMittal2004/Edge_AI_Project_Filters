import cv2
import numpy as np
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
prev_edges = None

def cartoonify_wholeImage(frame):
    global prev_edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges_canny = cv2.Canny(gray_blur, 70, 150)
    edges_laplacian = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize=5)
    combined_edges = cv2.bitwise_or(edges_canny, edges_laplacian)
    kernel = np.ones((3, 3), np.uint8)
    combined_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    combined_edges = cv2.GaussianBlur(combined_edges, (3, 3), 0)
    if prev_edges is None:
        prev_edges = combined_edges.copy()
    else:
        combined_edges = cv2.addWeighted(combined_edges, 0.8, prev_edges, 0.2, 0)
        prev_edges = combined_edges.copy()
    color = frame
    for i in range(0,2): 
        color = cv2.bilateralFilter(color, d=9, sigmaColor=100, sigmaSpace=100)
    edges_inv = cv2.bitwise_not(combined_edges)
    edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
    res = cv2.bitwise_and(color, edges_colored)
    return res

# def detect_faces(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
#     return faces
    
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    #faces = detect_faces(frame)
    resultt = cartoonify_wholeImage(frame)
    cv2.imshow('res', resultt)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
