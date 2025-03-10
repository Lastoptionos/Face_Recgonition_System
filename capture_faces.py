# import cv2
# import os
# import time
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# # Function to load data for training
# def load_data(data_dir, target_size=(100, 100)):
#     images = []
#     labels = []
#     for filename in os.listdir(data_dir):
#         if filename.endswith(".jpg"):
#             label = int(filename.split('.')[1])
#             img_path = os.path.join(data_dir, filename)
#             image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#             if image is not None:
#                 image = cv2.resize(image, target_size)
#                 images.append(image)
#                 labels.append(label)
#     return np.array(images), np.array(labels)

# # Capture Faces
# def capture_faces():
#     cam = cv2.VideoCapture(0)
#     face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#     face_id = input('\n Enter user ID (e.g., 0, 1, 2, ...) and press <return> ==> ')
#     face_name = input(f'\n Enter name for user ID {face_id} and press <return> ==> ')

#     print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

#     if not os.path.exists('dataset'):
#         os.mkdir('dataset')

#     file_path = 'dataset/labels.txt'
#     with open(file_path, 'a+') as file:
#         file.seek(0)  # Move to the start of the file to read its content
#         if file.read():  # If the file is not empty, add a newline before appending
#             file.write('\n')
#         file.write(face_name)

#     count = 0
#     frame_count = 0
#     start_time = time.time()

#     while True:
#         ret, img = cam.read()
#         img = cv2.flip(img, 1)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_detector.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             gray_face = gray[y:y+h, x:x+w]
#             cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray_face)
#             count += 1

#             frame_count += 1
#             end_time = time.time()
#             elapsed_time = end_time - start_time
#             fps = frame_count / elapsed_time
#             cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             cv2.putText(img, f"Img Captured: {count}", (img.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('image', 800, 600)
#         cv2.moveWindow('image', 0, 0)  # Place the window at the top of the screen
#         cv2.imshow('image', img)

#         k = cv2.waitKey(20) & 0xff
#         if k == 27 or count >= 100:  # Press 'ESC' to stop
#             break

#     print("\n [INFO] Exiting Program and cleanup stuff")
#     cam.release()
#     cv2.destroyAllWindows()

# # Train Model
# def train_model():
#     print("\n [INFO] Loading data and preparing for training...")
#     data_dir = 'dataset'
#     images, labels = load_data(data_dir)
#     images = images / 255.0
#     images = np.expand_dims(images, axis=-1)
#     labels = to_categorical(labels)

#     X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

#     print("\n [INFO] Training the model...")
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], 1)),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dense(labels.shape[1], activation='softmax')
#     ])

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#     # Check if the file exists, delete if it does
#     model_file = 'face_recognition_model.h5'
#     if os.path.exists(model_file):
#         os.remove(model_file)
#         print(f"{model_file} already existed and was deleted.")

#     # Save the model
#     model.save(model_file)
#     print(f"\n [INFO] Model saved as {model_file}")

import cv2
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Function to load data for training
def load_data(data_dir, target_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            label = int(filename.split('.')[1])
            img_path = os.path.join(data_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                image = cv2.resize(image, target_size)
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Capture Faces and Automatically Train Model
def capture_faces_and_train_model():
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

    face_id = input('\n Enter user ID (e.g., 0, 1, 2, ...) and press <return> ==> ')
    face_name = input(f'\n Enter name for user ID {face_id} and press <return> ==> ')

    print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    file_path = 'dataset/labels.txt'
    with open(file_path, 'a+') as file:
        file.seek(0)  # Move to the start of the file to read its content
        if file.read():  # If the file is not empty, add a newline before appending
            file.write('\n')
        file.write(face_name)

    count = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray_face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray_face)
            count += 1

            frame_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Img Captured: {count}", (img.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 600)
        cv2.moveWindow('image', 0, 0)  # Place the window at the top of the screen
        cv2.imshow('image', img)

        k = cv2.waitKey(20) & 0xff
        if k == 27 or count >= 100:  # Press 'ESC' to stop
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

    # Once faces are captured, train the model automatically
    print("\n [INFO] Starting model training...")
    train_model()

# Train Model
def train_model():
    print("\n [INFO] Loading data and preparing for training...")
    data_dir = 'dataset'
    images, labels = load_data(data_dir)
    images = images / 255.0
    images = np.expand_dims(images, axis=-1)
    labels = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    print("\n [INFO] Training the model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(labels.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model and store history
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the model
    model_file = 'dataset/face_recognition_model.h5'
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"{model_file} already existed and was deleted.")
    model.save(model_file)
    print(f"\n [INFO] Model saved as {model_file}")
    
    # Evaluate the model on the test set
    print("\n [INFO] Evaluating the model...")
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    # Calculate performance metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Display metrics in command line
    print("\n[INFO] Performance Metrics:")
    print(f"Accuracy: {overall_accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Classification report
    print("\n[INFO] Detailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(labels.shape[1])])
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n[INFO] Confusion Matrix:")
    print(cm)

    # Create the directory for performance-related files if not exist
    performance_dir = 'dataset/model_performance'
    if not os.path.exists(performance_dir):
        os.mkdir(performance_dir)

    # Save the classification report and confusion matrix
    report_file = os.path.join(performance_dir, 'model_report.txt')
    with open(report_file, 'w') as f:
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {overall_accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1-Score: {f1:.2f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"\n [INFO] Training report saved as {report_file}")

    # Plot and save performance graphs
    print("\n [INFO] Plotting training performance...")
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(performance_dir, 'training_performance.png'))
    plt.show()

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(labels.shape[1])])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(performance_dir, 'confusion_matrix.png'))
    plt.show()

# Main execution
if __name__ == "__main__":
    capture_faces_and_train_model()



