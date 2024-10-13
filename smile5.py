import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Load the fire, rain, and flower videos
fire_effect = cv2.VideoCapture('fire.mp4')   # Path to fire video
rain_effect = cv2.VideoCapture('rain.mp4')   # Path to rain video
flower_effect = cv2.VideoCapture('flower.mp4')  # Path to flower video

while cap.isOpened():
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
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

            # Get landmark points for the mouth and eyebrows
            left_upper_eyelid = face_landmarks.landmark[157]
            left_lower_eyelid = face_landmarks.landmark[153]
            right_upper_eyelid = face_landmarks.landmark[384] 
            right_lower_eyelid = face_landmarks.landmark[380]
            left_mouth = face_landmarks.landmark[61]   # Left corner of the mouth
            right_mouth = face_landmarks.landmark[291] # Right corner of the mouth
            upper_lip = face_landmarks.landmark[13]    # Upper lip point
            lower_lip = face_landmarks.landmark[14]    # Lower lip point

            left_eyebrow = face_landmarks.landmark[105]  # Left eyebrow
            right_eyebrow = face_landmarks.landmark[282] # Right eyebrow

            # Calculate distances for smile, mouth openness, and eyebrow distance
            smile_width = math.sqrt((right_mouth.x - left_mouth.x) ** 2 + (right_mouth.y - left_mouth.y) ** 2)
            mouth_open = math.sqrt((upper_lip.x - lower_lip.x) ** 2 + (upper_lip.y - lower_lip.y) ** 2)
            eyebrow_distance = math.sqrt((right_eyebrow.x - left_eyebrow.x) ** 2 + (right_eyebrow.y - left_eyebrow.y) ** 2)

            eyelid_eyebrow_distance = math.sqrt((left_eyebrow.x - face_landmarks.landmark[157].x) ** 2 +
                                                (right_eyebrow.y - face_landmarks.landmark[384].y) ** 2)

            # Set thresholds for different emotions
            smile_threshold = 0.05
            sad_threshold = 0.07
            anger_threshold = 0.08
            eyelid_eyebrow_distance_threshold = 0.1

            # Determine emotion
            if smile_width > smile_threshold and mouth_open > 0.01:
                emotion = 'Smiling'
                color = (0, 255, 0)
            elif smile_width < sad_threshold and mouth_open < 0.01:
                emotion = 'Sad'
                color = (255, 0, 0)
            elif eyebrow_distance < anger_threshold or eyelid_eyebrow_distance < eyelid_eyebrow_distance_threshold and smile_width < 0.08:
                emotion = 'Angry'
                color = (0, 0, 255)
            else:
                emotion = 'Neutral'
                color = (255, 255, 255)

            # Overlay fire effect in "Angry" mode
            if emotion == 'Angry':
                ret_fire, fire_frame = fire_effect.read()
                if ret_fire:
                    fire_frame = cv2.resize(fire_frame, (frame.shape[1], frame.shape[0]))
                    frame = cv2.addWeighted(fire_frame, 0.7, frame, 0.3, 0)

            # Overlay rain effect in "Sad" mode
            if emotion == 'Sad':
                ret_rain, rain_frame = rain_effect.read()
                if ret_rain:
                    rain_frame = cv2.resize(rain_frame, (frame.shape[1], frame.shape[0]))
                    frame = cv2.addWeighted(rain_frame, 0.5, frame, 0.5, 0)

            # Overlay flower effect in "Smiling" mode
            if emotion == 'Smiling':
                ret_flower, flower_frame = flower_effect.read()
                if ret_flower:
                    flower_frame = cv2.resize(flower_frame, (frame.shape[1], frame.shape[0]))
                    frame = cv2.addWeighted(flower_frame, 0.7, frame, 0.3, 0)

            # Display the detected emotion on the screen
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Display the frame with annotations and effects
    cv2.imshow('Emotion Detector', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
fire_effect.release()
rain_effect.release()
flower_effect.release()
cv2.destroyAllWindows()
