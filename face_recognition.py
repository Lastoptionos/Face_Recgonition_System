import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('dataset/face_recognition_model.h5')

# Load labels
with open('dataset/labels.txt', 'r') as file:
    labels = file.read().splitlines()

# Start video capture for real-time recognition
cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100)) / 255.0
        face = np.expand_dims(face, axis=[0, -1])

        prediction = model.predict(face)
        label_id = np.argmax(prediction)
        confidence = np.max(prediction)

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label_text = f"{labels[label_id]}: {confidence:.2f}"
        cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-Time Recognition', img)

    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


