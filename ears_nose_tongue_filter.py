import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


tongue_img = cv2.imread("C:\\Users\\shini\\OneDrive\\Desktop\\edgeAI_project\\doggy_tongue.png", cv2.IMREAD_UNCHANGED)
dog_ears_img = cv2.imread("C:\\Users\\shini\\OneDrive\\Desktop\\edgeAI_project\\doggy_ears.png", cv2.IMREAD_UNCHANGED)  # Separate dog ears image
dog_nose_img = cv2.imread("C:\\Users\\shini\\OneDrive\\Desktop\\edgeAI_project\\doggy_nose.png", cv2.IMREAD_UNCHANGED)  # Separate dog nose image

cap = cv2.VideoCapture(0)

while True:
    mouthOpen = False
    _, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_landmark = face_landmarks.landmark[1]
            nose_coords = (int(nose_landmark.x * frame.shape[1]), int(nose_landmark.y * frame.shape[0]))
            face_width = int(abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * frame.shape[1])
            ears_height = int(face_width * (dog_ears_img.shape[0] / dog_ears_img.shape[1])) 
            resized_ears = cv2.resize(dog_ears_img, (face_width, ears_height))
            face_top = face_landmarks.landmark[10] 
            face_top_x = int(face_top.x * frame.shape[1])
            face_top_y = int(face_top.y * frame.shape[0]) - ears_height 
            ears_x = face_top_x - face_width // 2  
            if ears_x < 0:
                ears_x = 0 
            if face_top_y < 0:
                face_top_y = 0  
            if resized_ears.shape[2] == 4: 
                alpha_channel = resized_ears[:, :, 3] / 255.0  
                for c in range(0, 3):
                    overlay_area = frame[face_top_y:face_top_y + ears_height, ears_x:ears_x + face_width]
                    overlay_area_shape = overlay_area.shape[:2] 
                    if resized_ears.shape[:2] == overlay_area_shape: 
                        overlay_area[:, :, c] = (alpha_channel * resized_ears[:, :, c] + (1 - alpha_channel) * overlay_area[:, :, c])
            nose_width = int(face_width * 0.5)
            nose_height = int(nose_width * (dog_nose_img.shape[0] / dog_nose_img.shape[1])) 
            resized_nose = cv2.resize(dog_nose_img, (nose_width, nose_height))
            nose_x = int(nose_coords[0] - nose_width // 2)
            nose_y = int(nose_coords[1] - nose_height // 2) 
            if nose_x + nose_width > frame.shape[1]:
                nose_width = frame.shape[1] - nose_x
            if nose_y + nose_height > frame.shape[0]:
                nose_height = frame.shape[0] - nose_y

            resized_nose = cv2.resize(dog_nose_img, (nose_width, nose_height))
            if resized_nose.shape[2] == 4:  
                alpha_channel = resized_nose[:, :, 3] / 255.0 
                for c in range(0, 3):
                    overlay_area = frame[nose_y:nose_y + nose_height, nose_x:nose_x + nose_width]
                    overlay_area_shape = overlay_area.shape[:2]
                    if resized_nose.shape[:2] == overlay_area_shape:
                        overlay_area[:, :, c] = (alpha_channel * resized_nose[:, :, c] + (1 - alpha_channel) * overlay_area[:, :, c])
            upper_lip_center = face_landmarks.landmark[0] 
            lower_lip_center = face_landmarks.landmark[17] 
            upper_lip_coords = np.array([upper_lip_center.x, upper_lip_center.y])
            lower_lip_coords = np.array([lower_lip_center.x, lower_lip_center.y])
            lip_distance = np.linalg.norm(upper_lip_coords - lower_lip_coords)
            lip_threshold = 0.060 

            if lip_distance > lip_threshold:
                mouthOpen = True

            if mouthOpen:
                left_mouth = face_landmarks.landmark[61]  
                right_mouth = face_landmarks.landmark[291] 
                left_mouth_coords = (int(left_mouth.x * frame.shape[1]), int(left_mouth.y * frame.shape[0]))
                right_mouth_coords = (int(right_mouth.x * frame.shape[1]), int(right_mouth.y * frame.shape[0]))

                tongue_width = int(np.linalg.norm(np.array(left_mouth_coords) - np.array(right_mouth_coords)))
                tongue_height = int(tongue_width * (tongue_img.shape[0] / tongue_img.shape[1])) 
                resized_tongue = cv2.resize(tongue_img, (tongue_width, tongue_height))
                tongue_x = left_mouth_coords[0]
                tongue_y = int((left_mouth_coords[1] + right_mouth_coords[1]) / 2) 
                if tongue_x + tongue_width > frame.shape[1]:
                    tongue_width = frame.shape[1] - tongue_x
                if tongue_y + tongue_height > frame.shape[0]:
                    tongue_height = frame.shape[0] - tongue_y

                resized_tongue = cv2.resize(tongue_img, (tongue_width, tongue_height))
                if resized_tongue.shape[2] == 4:  
                    alpha_channel = resized_tongue[:, :, 3] / 255.0 
                    for c in range(0, 3):
                        overlay_area = frame[tongue_y:tongue_y + tongue_height, tongue_x:tongue_x + tongue_width]
                        overlay_area_shape = overlay_area.shape[:2] 

                        if resized_tongue.shape[:2] == overlay_area_shape:
                            overlay_area[:, :, c] = (alpha_channel * resized_tongue[:, :, c] + (1 - alpha_channel) * overlay_area[:, :, c])

            # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)


    cv2.imshow("Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
