# Face Recognition System Using CNN Algorithm

## Overview
#### This project implements a Face Recognition System using a Convolutional Neural Network (CNN). The system is trained to identify and classify faces based on a dataset of images. It utilizes deep learning techniques to extract and compare facial features for recognition.

## Features
- Face detection and recognition using CNN
- Model training on a dataset of labeled images
- Real-time face recognition via webcam
- Image pre-processing and feature extraction
- Uses TensorFlow/Keras and OpenCV for implementation

## Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib

#### Install dependencies using:
pip install tensorflow opencv-python numpy matplotlib

## Dataset
#### The system requires a dataset of face images categorized into different labeled folders. You can use datasets like Labeled Faces in the Wild (LFW) or create your own by collecting images.

## Model Architecture

### The CNN model consists of:
- Convolutional layers for feature extraction
- Max-pooling layers for dimensionality reduction
- Fully connected layers for classification
- Softmax activation for multi-class classification

## Results
### The system will output the recognized person's name along with the confidence score. Incorrect classifications can be improved by increasing the dataset size and refining the model.

## Future Improvements
- Improve accuracy with deeper networks
- Implement face embedding techniques (e.g., FaceNet)
- Deploy as a web or mobile application
- Add support for multiple faces in a single frame



