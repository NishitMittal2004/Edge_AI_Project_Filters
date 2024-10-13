import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
img = cv2.imread("C:\\Users\\shini\\OneDrive\\Desktop\\doggy_tongue.png", cv2.IMREAD_UNCHANGED)
# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    mouthOpen = False
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame color to RGB (Mediapipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find face landmarks
    results = face_mesh.process(rgb_frame)

    # If landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh landmarks
            #mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

            upper_lip_center = face_landmarks.landmark[0] 
            lower_lip_center = face_landmarks.landmark[17] 


            upper_lip_coords = np.array([upper_lip_center.x, upper_lip_center.y])
            lower_lip_coords = np.array([lower_lip_center.x, lower_lip_center.y])

            # dist b/w centre of lips
            lip_distance = np.linalg.norm(upper_lip_coords - lower_lip_coords)

           
            lip_threshold = 0.060


            if lip_distance > lip_threshold:
                mouthOpen = True
                #cv2.putText(frame, 'Mouth Open', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #else:
                #cv2.putText(frame, 'Mouth Closed', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Optionally, draw circles at the centers for visualization
            #cv2.circle(frame, (int(upper_lip_coords[0] * frame.shape[1]), int(upper_lip_coords[1] * frame.shape[0])), 5, (255, 0, 0), -1)
            #cv2.circle(frame, (int(lower_lip_coords[0] * frame.shape[1]), int(lower_lip_coords[1] * frame.shape[0])), 5, (255, 0, 0), -1)

            if mouthOpen:
                left_mouth = face_landmarks.landmark[61]  
                right_mouth = face_landmarks.landmark[291] 

                # Calculate the position in pixels
                left_mouth_coords = (int(left_mouth.x * frame.shape[1]), int(left_mouth.y * frame.shape[0]))
                right_mouth_coords = (int(right_mouth.x * frame.shape[1]), int(right_mouth.y * frame.shape[0]))

                tongue_width = int(np.linalg.norm(np.array(left_mouth_coords) - np.array(right_mouth_coords)))
                tongue_height = int(tongue_width * (img.shape[0] / img.shape[1]))  
                resized_tongue = cv2.resize(img, (tongue_width, tongue_height))
                tongue_x = left_mouth_coords[0]
                tongue_y = int((left_mouth_coords[1] + right_mouth_coords[1]) / 2) 

                if resized_tongue.shape[2] == 4:  
                    alpha_channel = resized_tongue[:, :, 3] / 255.0  
                    for c in range(0, 3):
                        frame[tongue_y:tongue_y + tongue_height, tongue_x:tongue_x + tongue_width, c] = \
                            alpha_channel * resized_tongue[:, :, c] + \
                            (1 - alpha_channel) * frame[tongue_y:tongue_y + tongue_height, tongue_x:tongue_x + tongue_width, c]

   
    cv2.imshow('Mouth State Detector', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
