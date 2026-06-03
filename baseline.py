# This code serves as a baseline 3D CNN video classification model that will be used to compare with topologically 
# enhanced models. 

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Input
from sklearn.metrics import classification_report

# Define global variables
num_frames = 8
resolution = 56
num_epochs = 10
layer_1_filters = 32
layer_2_filters = 64
layer_3_filters = 128

num_classes = 101

def main():

    # Firstly, lets import all video files from UCF101
    path = Path('/Users/darcysprigg/Coding/Co-op summer 2026/UCF101')

    # Load all video data
    video_frames, encoded_labels, class_names = load_data(path, num_classes, num_frames)

    # Split video data giving 70% for training, 20% for testing, and 10% for validation
    video_frames, video_frames_test, encoded_labels, encoded_labels_test = train_test_split(
        video_frames, encoded_labels, test_size=0.2, random_state=42)
    
    video_frames, video_frames_val, encoded_labels, encoded_labels_val = train_test_split(
        video_frames, encoded_labels, test_size=0.125, random_state=42)
    
    # Create input shape with a specified frame/pixel count and 3 channels
    input_shape = (num_frames, resolution, resolution, 3)

    # Call helper function to create model
    model = create_3dCNN_model(input_shape, num_classes)

    # Train the model using training and validation data
    history = model.fit(video_frames, encoded_labels, validation_data=(video_frames_val, encoded_labels_val), 
                        epochs=num_epochs, batch_size=8)

    # Evaluate the models performance using the testing data
    loss, accuracy = model.evaluate(video_frames_test, encoded_labels_test)

    # Print the accuracy of the model
    print(f'Test Accuracy: {accuracy:.2f}')

    # Obtain predictions from the model using testing data
    encoded_predict = model.predict(video_frames_test)

    # Convert predicted classes and true classes to class labels
    encoded_pred_classes = np.argmax(encoded_predict, axis=1)
    encoded_true_classes = np.argmax(encoded_labels_test, axis=1)

    # Create a classification report
    class_report = classification_report(encoded_true_classes, encoded_pred_classes, target_names = class_names)

    # Print report
    print(class_report)

# Function made to extract 16 frames from a video file
def extract_frames(path, num_frames):

    # Open specified video file and initialize a list of frames
    capture = cv2.VideoCapture(path)
    frames = []

    # Get total number of frames in video
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the interval at which the frames will be extracted
    frame_interval = max(total_frames // num_frames, 1)

    # Loop through frames and extract desired ones
    for i in range(num_frames):

        # Go to specified frame index
        capture.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)

        # Get specified frame
        ret, frame = capture.read()
        
        # Check if frame was successfully read
        if not ret:
            break
        
        # Resize frame to specifed size
        frame = cv2.resize(frame, (resolution, resolution))

        # Add frame to frames list
        frames.append(frame)

    # Close video file
    capture.release()

    # Fill in missing frames with blank frames
    while len(frames) < num_frames:
        frames.append(np.zeros((resolution, resolution, 3), np.uint8))

    # Return array of frames
    return np.array(frames)

# This function loads video data and prepares it for training a 3D CNN model
def load_data(path_dir, num_classes, num_frames):

    # Initalize frame and label lists
    video_frames = []
    labels = []

    # Loop through video categories
    for category in tqdm(os.listdir(path_dir)):

        # Define path to given category
        category_path = os.path.join(path_dir, category)

        if os.path.isdir(category_path):

            # Loop through each video within the specified category
            for video in os.listdir(category_path):
                
                # Define path to video
                video_path = os.path.join(category_path, video)

                # Call helper function to get array of video frames
                frames = extract_frames(video_path, num_frames)

                # Append frames to video_frames list
                video_frames.append(frames)

                # Append category name to labels
                labels.append(category)

    # Make video_frames list an array
    video_frames = np.array(video_frames)

    # Get label codes and class names
    codes, class_names = pd.factorize(np.array(labels))

    # Convert labels into one-hot encoded format
    encoded_labels = to_categorical(codes, num_classes)

    # Return video frames, encoded labels and class names
    return video_frames, encoded_labels, class_names

# Function that builds a 3D CNN video classification model
def create_3dCNN_model(input_shape, num_classes):

    # Initialize model
    model = Sequential()

    # Add input shape to model
    model.add(Input(input_shape))

    # Adding a 3D convolutional layer containing a specified number of filters, a kernel size of (3,3,3) 
    # and relu activation
    model.add(Conv3D(layer_1_filters, (3, 3, 3), activation='relu', padding='same'))
    # Adding a 3D max pooling layer with a pool size of (2,2,2)
    model.add(MaxPooling3D((2, 2, 2)))
    # Adding a batch normalization layer
    model.add(BatchNormalization())

    # Adding a 3D convolutional layer containing a specified number of filters, a kernel size of (3,3,3) 
    # and relu activation
    model.add(Conv3D(layer_2_filters, (3, 3, 3), activation='relu', padding='same'))
    # Adding a 3D max pooling layer with a pool size of (2,2,2)
    model.add(MaxPooling3D((2, 2, 2)))
    # Adding a batch normalization layer
    model.add(BatchNormalization())

    # Adding a 3D convolutional layer containing a specified number of filters, a kernel size of (3,3,3) 
    # and relu activation
    model.add(Conv3D(layer_3_filters, (3, 3, 3), activation='relu', padding='same'))
    # Adding a 3D max pooling layer with a pool size of (2,2,2)
    model.add(MaxPooling3D((2, 2, 2)))
    # Adding a batch normalization layer
    model.add(BatchNormalization())

    # Adding a flatten layer
    model.add(Flatten())
    # Adding dense layer with 512 units and relu activation
    model.add(Dense(512, activation='relu'))
    # Adding dropout layer with a dropout rate of 0.5
    model.add(Dropout(0.5))
    # Adding output layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    # Compiling the model with adam optimizer, categorical crossentropy loss and accuracy metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Return model
    return model

if __name__ == "__main__":
    main()