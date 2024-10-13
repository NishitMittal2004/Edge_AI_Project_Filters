import cv2
import numpy as np
import pyaudio
import random
import imutils

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

def detect_audio_expression(data):
    max_amplitude = np.max(np.abs(data))
    max_amplitude /= 5

    if max_amplitude > 2000:
        return 'high'
    elif max_amplitude > 1000:
        return 'Medium'
    elif max_amplitude > 500:
        return 'low'
    return 'none'

def apply_transparent_background(frame, background_color, alpha=0.5):
    overlay = np.full_like(frame, background_color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)



while True:
    ret, frame = cap.read()
    data = np.frombuffer(stream.read(1024), dtype=np.int16)
    expression = detect_audio_expression(data)

    if expression == 'high':
        frame = apply_transparent_background(frame, (0, 0, 255), alpha=0.5)
        #frame = shake_frame(frame, intensity=10)  # Shake for high expression
    elif expression == 'Medium':
        frame = apply_transparent_background(frame, (255, 0, 0), alpha=0.5)
        frame = imutils.rotate(frame, angle=random.uniform(-10, 10))  # Dizzy effect for medium expression
    elif expression == 'low':
        frame = apply_transparent_background(frame, (128, 128, 128), alpha=0.5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# stream.stop_stream()
# stream.close()
# p.terminate()