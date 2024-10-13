import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier("C:\\Users\\shini\\OneDrive\\Desktop\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:\\Users\\shini\\OneDrive\\Desktop\\haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier("C:\\Users\\shini\\OneDrive\\Desktop\\Nose.xml")
mouth_cascade = cv2.CascadeClassifier("C:\\Users\\shini\\OneDrive\\Desktop\\Mouth.xml")
profile_face_cascade = cv2.CascadeClassifier("C:\\Users\\shini\\OneDrive\\Desktop\\haarcascade_profileface.xml")
sunglasses = cv2.imread("C:\\Users\\shini\\OneDrive\\Desktop\\glasses.png", cv2.IMREAD_UNCHANGED)
mustache = cv2.imread("C:\\Users\\shini\\OneDrive\\Desktop\\mustache.png", cv2.IMREAD_UNCHANGED)
cap = cv2.VideoCapture(0)
def sunglass(frame, x, y, w, h, sunglasses, a=0.6):
    new_w = int(w * 1.3)  
    new_h = int(h * 1.3) 
    sunglasses_resized = cv2.resize(sunglasses, (new_w, new_h))
    sunglasses_rgb = sunglasses_resized[:, :, :3]
    sunglasses_alpha = sunglasses_resized[:, :, 3]
    y = y + int(0.2 * h)
    #y = y + int(0.1 * h)
    x = x - int(0.08 * new_w)

    roi = frame[y:y+new_h, x:x+new_w]
    mask = (sunglasses_alpha / 255.0) * a
    inverse_mask = 1.0 - mask
    #for c in range(3): 
    roi[:, :, 0] = (mask * sunglasses_rgb[:, :,0] + inverse_mask * roi[:, :, 0])
    roi[:, :, 1] = (mask * sunglasses_rgb[:, :,1] + inverse_mask * roi[:, :, 1])
    roi[:, :, 2] = (mask * sunglasses_rgb[:, :,2] + inverse_mask * roi[:, :, 2])
    frame[y:y+new_h, x:x+new_w] = roi

def filter_sunglass():
    while True:
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        profile_faces = profile_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces_array = np.array(faces) if len(faces) > 0 else np.empty((0, 4), dtype=int)
        profile_faces_array = np.array(profile_faces) if len(profile_faces) > 0 else np.empty((0, 4), dtype=int)
        combined_faces = np.vstack((faces_array, profile_faces_array))
        sunglasses_applied = False
        for (fx, fy, fw, fh) in combined_faces:
            if len(faces) > 0 and not sunglasses_applied:
                #cv2.rectangle(frame, (fx,fy),(fx+fw,fy + fh), (255,0,0), 2)
                roi_gray = gray_frame[fy:fy+fh, fx:fx+fw]
                roi_color = frame[fy:fy+fh, fx:fx+fw]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda eye: eye[0])
                    eye1 = eyes[0]
                    eye2 = eyes[1]
                    x = fx + eye1[0]
                    y = fy + min(eye1[1], eye2[1])
                    w = (fx + eye2[0] + eye2[2]) - (fx + eye1[0])
                    h = max(eye1[3], eye2[3])
                    sunglass(frame, x, y, w, h, sunglasses, a=0.6)
                    sunglasses_applied = True
            #noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            #mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        cv2.imshow('fraem', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def overlay_mustache(frame, x, y, w, h, mustache, alpha_transparency=0):
    new_w = int(w * 0.5)
    new_h = int(h * 0.2)
    mustache_resized = cv2.resize(mustache, (new_w, new_h))
    mustache_rgb = mustache_resized[:, :, :3]
    mustache_alpha = mustache_resized[:, :, 3]
    y = y + int(0.6 * h) 
    x = x + int(0.15 * w)
    roi = frame[y:y+new_h, x:x+new_w]
    mask = (mustache_alpha / 255.0) * alpha_transparency
    inverse_mask = 1.0 - mask

    for c in range(3):
        roi[:, :, c] = (mask * mustache_rgb[:, :, c] + inverse_mask * roi[:, :, c])
    
    frame[y:y+new_h, x:x+new_w] = roi

def filter_mustache():
    while True:
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        profile_faces = profile_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(profile_faces) == 0:
            profile_faces = np.empty((0, 4), dtype=int)
        if len(faces) > 0:
            combined_faces = np.vstack((faces, profile_faces))
        else:
            combined_faces = profile_faces
        for (fx, fy, fw, fh) in combined_faces:
            overlay_mustache2(frame, fx, fy, fw, fh, mustache, alpha_transparency=1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def overlay_mustache2(frame, x, y, w, h, mustache, a=1):
    new_w = int(w * 0.5)
    new_h = int(h * 0.2)  
    mustache_resized = cv2.resize(mustache, (new_w, new_h))
    mustache_rgb = mustache_resized[:, :, :3]
    mustache_alpha = mustache_resized[:, :, 3]
    y = y + int(0.7 * h)
    x = x + int((w - new_w) / 2) 

    roi = frame[y:y+new_h, x:x+new_w]
    mask = (mustache_alpha / 255.0) * a
    inverse_mask = 1.0 - mask

    for c in range(3):
        roi[:, :, c] = (mask * mustache_rgb[:, :, c] + inverse_mask * roi[:, :, c])
    frame[y:y+new_h, x:x+new_w] = roi

cat_filter = cv2.imread("C:\\Users\\shini\\OneDrive\\Desktop\\cat.png", cv2.IMREAD_UNCHANGED)
print(cat_filter.shape)

# Function to overlay the cat filter on detected faces
def overlay_cat_filter(frame, x, y, w, h, cat_filter, alpha_transparency=1.0):
    # Resize the cat filter image to fit the face width
    new_w = int(w * 1.2)  # Resize width to 1.2x face width (adjust as necessary)
    new_h = int(h * 1.5)  # Resize height to 1.5x face height to include ears and whiskers
    cat_filter_resized = cv2.resize(cat_filter, (new_w, new_h))

    # Extract the RGB and alpha channels from the cat filter
    cat_rgb = cat_filter_resized[:, :, :3]
    cat_alpha = cat_filter_resized[:, :, 3]

    # Adjust the position to overlay the filter correctly (above the face for the ears)
    y = y - int(0.4 * h)  # Move the filter up to cover the ears (adjust as needed)
    
    # Get the region of interest (ROI) on the frame where the filter will be placed
    roi = frame[y:y+new_h, x:x+new_w]

    # Create a mask based on the filter's alpha channel and adjust transparency
    mask = (cat_alpha / 255.0) * alpha_transparency
    inverse_mask = 1.0 - mask

    # Blend the filter with the frame
    for c in range(3):  # Loop over RGB channels
        roi[:, :, c] = (mask * cat_rgb[:, :, c] + inverse_mask * roi[:, :, c])

    # Place the blended region back into the frame
    frame[y:y+new_h, x:x+new_w] = roi

# Usage example inside your video capture loop
def filter_cat():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (fx, fy, fw, fh) in faces:
            # Overlay the cat filter on each detected face
            overlay_cat_filter(frame, fx, fy, fw, fh, cat_filter, alpha_transparency=1.0)

        # Display the frame with the filter applied
        cv2.imshow('Cat Filter - Press "q" to Exit', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the filter function
filter_cat()

#filter_mustache()
#filter_sunglass()
cap.release()
cv2.destroyAllWindows()
