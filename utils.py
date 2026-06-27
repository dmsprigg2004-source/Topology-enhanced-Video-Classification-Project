# Copyright 2022 The TensorFlow Authors
# Copyright 2026 Darcy Sprigg

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import cv2
import einops
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
import os
from pathlib import Path
import shutil
import copy
import math

# ----------------------------------  DATA LOADING AND PREPROCESSING CODE --------------------------------------------

# Function that defines and returns test settings
def get_test_settings():

    # Define desired test settings
    num_categories = 5
    splits = {"train": 70, "val": 10, "test": 20}
    epochs = 40
    height = 112
    width = 112
    n_frames = 16
    batch_size = 8

    # Calculate steps per epoch and validation steps
    steps_per_epoch = (splits['train'] * num_categories) // batch_size
    validation_steps = (splits['val'] * num_categories) // batch_size

    # Return test settings
    return num_categories, splits, epochs, height, width, n_frames, batch_size, steps_per_epoch, validation_steps

# Call above function to get test settings
num_categories, splits, epochs, height, width, n_frames, batch_size, steps_per_epoch, validation_steps = get_test_settings()

# Function that gets a list of files to be used in either training, validation or testing as well as a dictionary 
# specifying which files are unused
def split_class_lists(category_dict, split_count):

    # Initialize list and remainder dictionary
    split_files = []
    remainder = {}

    # Loop through each category
    for category in category_dict:

        # Add needed files to list
        split_files.extend(category_dict[category][:split_count])

        # Add item to remainder dictionary specifying which files have not been chosen in that category
        remainder[category] = category_dict[category][split_count:]

    # Return list and remainder dictionary
    return split_files, remainder

# Function to create a new subset directory
def create_subset_dir(category_dict, categories_list, split_files, split_name):

    # Define path for new directory
    new_dir_path = Path(f'./{split_name}')

    # If directory exists, delete it
    if new_dir_path.is_dir():
        shutil.rmtree(new_dir_path)
    
    # Make new directory
    new_dir_path.mkdir()
    
    # Loop through categories
    for category in categories_list:

        # Make a new directory inside previously made directory
        new_category_dir_path = Path(f'./{split_name}/{category}')
        new_category_dir_path.mkdir()

        # Loop through files in each category
        for file in category_dict[category]:

            # Loop through split_files
            for split_file in split_files:

                # If the current file is to be used in this split, copy it to the new directory
                if file == split_file:
                    needed_file = Path(f'./UCF101/{category}/{file}')
                    shutil.copy(needed_file, new_category_dir_path)

    # Return path to new directory
    return new_dir_path

# Function that creates subset directories for training, validation and testing
def create_subset_dirs(num_categories, UCF101_dir, splits):

    # Initialize dicitonary and category count
    category_dict = {}
    category_count = 0

    # Loop through categories in UCF101
    for category in os.listdir(UCF101_dir):

        # Define path to category
        category_path = os.path.join(UCF101_dir, category)

        # If path leads to a directory, add category to dictionary and copy all videos into a list under that category
        if os.path.isdir(category_path):
            category_dict[category] = []

            for video in os.listdir(category_path):
                category_dict[category].append(video)

            # Add 1 to count
            category_count += 1

        # Once reached specified category count, break
        if category_count == num_categories:
           break
    
    # Get list of category names within category_dict
    categories_list = list(category_dict.keys())[:num_categories]

    # Create random order for videos within category_dict
    for category in categories_list:
        new_files_for_class = category_dict[category]
        random.shuffle(new_files_for_class)
        category_dict[category] = new_files_for_class

    # Initialize dictionary and make a copy of category_dict
    subset_dirs = {}
    category_dict_copy = copy.deepcopy(category_dict)

    # Loop through keys and values within splits dicitonary
    for split_name, split_count in splits.items():

        # Get files to be used in this split
        split_files, category_dict_copy = split_class_lists(category_dict_copy, split_count)

        # Create directory of this split
        split_dir = create_subset_dir(category_dict, categories_list, split_files, split_name)

        # Print message indicating directory was made
        print(f"{split_name} directory created")

        # Add directory to output directory
        subset_dirs[split_name] = split_dir

    # Return subset directories
    return subset_dirs

# Function that formats a specified frame
def format_frames(frame, output_size):

    # Change frame data type and resize it
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)

    # Return frame
    return frame

# Function that extracts frames from a video file
def frames_from_video_file(video_path, n_frames, output_size = (height,width)):

    # Initalize output list 
    result = []

    # Initalize new VideoCapture object
    src = cv2.VideoCapture(str(video_path))  

    # Define length needed for a frame step of 15
    need_length = 1 + (n_frames - 1) * 15

    # Get number of frames in video
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    # If video length is less than what is needed for a frame step of 15, calculate new frame step
    if video_length < need_length:

        # Calculate valid frame step
        frame_step = max(1, math.floor(video_length / n_frames))

        # Define start as first frame
        start = 0

    else:
        # Set frame step to 15
        frame_step = 15

        # Define start to a random position within a valid range
        max_start = video_length - need_length
        start = random.randint(0, max_start)

    # Set starting position using start variable
    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Read video frame
    ret, frame = src.read()

    # Format frame and append to output list
    result.append(format_frames(frame, output_size))

    # Loop one time for each remaining needed frame
    for _ in range(n_frames - 1):
        
        # Initialize empty lists
        ret_list = []
        frame_list = []

        # Loop a specified number of times to skip frames
        for _ in range(frame_step):

            # Read video frame
            ret, frame = src.read()

            # Append values to lists
            ret_list.append(ret)
            frame_list.append(frame)

        # Loop once for each frame just read
        for i in range(frame_step):

            # If current frame was extracted, format, append to result, and break the loop
            if ret_list[len(ret_list) - 1 - i] == True:
                frame = format_frames(frame_list[len(frame_list) - 1 - i], output_size)
                result.append(frame)
                break
            
            # If no frames were successfully extracted, add fully black image to result
            if i == frame_step - 1:
                result.append(np.zeros_like(result[0]))

    # Release VideoCapture object
    src.release()

    # Convert result to an array and rearrange color channels
    result = np.array(result)[..., [2, 1, 0]]

    # Return array
    return result

# Define FrameGenerator class
class FrameGenerator:

    # __init__ function to initialize instance attributes
    def __init__(self, path, n_frames, training = False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    # Function that returns lists of paths to video files and class names for each video
    def get_files_and_class_names(self):

        # Create lists of video paths and video class names
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths] 

        # Return lists
        return video_paths, classes

    # __call__ function that yields video frames with their respective label
    def __call__(self):

        # Call function to get video paths and class names
        video_paths, classes = self.get_files_and_class_names()

        # Create a list of tuples containing video paths and their respective class
        pairs = list(zip(video_paths, classes))

        # If training is True, mix up pairs within pairs list
        if self.training:
            random.shuffle(pairs)

        # Loop through each tuple in pairs list
        for path, name in pairs:

            # Get video frames
            video_frames = frames_from_video_file(path, self.n_frames) 

            # Get label
            label = self.class_ids_for_name[name]

            # Yield video frames and its respective label
            yield video_frames, label

# ------------------------------  END OF DATA LOADING AND PREPROCESSING CODE ----------------------------------------

# ---------------------------------------  MODEL CREATION CODE ------------------------------------------------------

# Define class to create a custom (2+1)D convolutional layer
class Conv2Plus1D(keras.layers.Layer):

    # __init__ function to initialize instance attributes
    def __init__(self, filters, kernel_size, padding):

        # Initialize instance attributes within keras.layers.Layer
        super().__init__()

        # Create small neural network with one 2D and one 1D convolutional layer
        self.seq = keras.Sequential([  
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            layers.Conv3D(filters=filters, 
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
            ])
    
    # Function that creates above neural network with an input variable then returns 
    def call(self, x):
        return self.seq(x)

# Define class that creates the main branch of the residual block
class ResidualMain(keras.layers.Layer):

    # __init__ function to initialize instance attributes
    def __init__(self, filters, kernel_size):

        # Initialize instance attributes within keras.layers.Layer
        super().__init__()

        # Create small neural network using Conv2Plus1D, layer normalization and ReLU activation
        self.seq = keras.Sequential([
                  Conv2Plus1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same'),
                  layers.LayerNormalization(),
                  layers.ReLU(),
                  Conv2Plus1D(filters=filters, 
                              kernel_size=kernel_size,
                              padding='same'),
                  layers.LayerNormalization()
                  ])
    
    # Function that creates above neural network with an input variable then returns 
    def call(self, x):
        return self.seq(x)
    
    # Define class that updates a branches shape to a specified size
class Project(keras.layers.Layer):

    # __init__ function to initialize instance attributes
    def __init__(self, units):

        # Initialize instance attributes within keras.layers.Layer
        super().__init__()

        # Create small neural network with a dense layer of specified size and layer normalization
        self.seq = keras.Sequential([
                  layers.Dense(units),
                  layers.LayerNormalization()
                  ])

    # Function that creates above neural network with an input variable then returns 
    def call(self, x):
        return self.seq(x)
    
    # Function that creates a residual block
def add_residual_block(input, filters, kernel_size):

    # Create the main branch of the residual block and save it to a variable
    out = ResidualMain(filters, kernel_size)(input)
    
    # Save input to new variable
    res = input

    # If residual branch and main branch don't have the same shape, update residual branch
    if out.shape[-1] != input.shape[-1]:

        # Call function to change the residual branches shape
        res = Project(out.shape[-1])(res)

    # Add layers together and return
    return layers.add([res, out])

# Define class to resize videos
class ResizeVideo(keras.layers.Layer):

    # __init__ function to initialize instance attributes
    def __init__(self, height, width):

        # Initialize instance attributes within keras.layers.Layer
        super().__init__()

        # Initialize remaining instance attributes
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    # Function that resizes frames within videos
    def call(self, video):

        # Define original shape of video
        old_shape = einops.parse_shape(video, 'b t h w c')

        # Flatten batch and time dimensions so each frame may be treated independently
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')

        # Resize images
        images = self.resizing_layer(images)

        # Revert to original video shape
        videos = einops.rearrange(
                images, '(b t) h w c -> b t h w c',
                t = old_shape['t'])
        
        # Return videos
        return videos
    
    # Function that creates a 3D CNN model using a training dataset and a specified input shape
def create_3D_CNN(x_ds, input_shape):

    # Create input layer with input shape and copy to "x" variable
    input = layers.Input(shape=(input_shape[1:]))
    x = input

    # Create a (2+1)D convolutional layer with the Conv2Plus1D class
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    # Normalize activations within neural network layer
    x = layers.BatchNormalization()(x)
    # Implement ReLU activation layer
    x = layers.ReLU()(x)
    # Resize video to half of its current hight and width
    x = ResizeVideo(height // 2, width // 2)(x)

    # Add residual block with a specified number of filters and specific kernel size
    x = add_residual_block(x, 16, (3, 3, 3))
    # Resize video to one quarter of its current hight and width
    x = ResizeVideo(height // 4, width // 4)(x)

    # Add residual block with a specified number of filters and specific kernel size
    x = add_residual_block(x, 32, (3, 3, 3))
    # Resize video to one eighth of its current height and width
    x = ResizeVideo(height // 8, width // 8)(x)

    # Add residual block with a specified number of filters and specific kernel size
    x = add_residual_block(x, 64, (3, 3, 3))
    # Resize video to one sixteenth of its current height and width
    x = ResizeVideo(height // 16, width // 16)(x)

    # Add residual block with a specified number of filters and specific kernel size
    x = add_residual_block(x, 128, (3, 3, 3))

    # Add layer that downsamples model data
    x = layers.GlobalAveragePooling3D()(x)

    # Add layer to reshape data into one dimension
    x = layers.Flatten()(x)

    # Add a dense layer the size of the number of categories
    x = layers.Dense(num_categories)(x)

    # Define model using starting and ending points
    model = keras.Model(input, x)

    # Initailize weights within neural network using a sample of video frames
    frames, label = next(iter(x_ds))
    model.build(frames)

    # Return model
    return model

# -------------------------------------  END OF MODEL CREATION CODE ---------------------------------------------------

# ---------------------------------------  MODEL EVALUATION CODE ------------------------------------------------------

# Function that generates plots showing how accuracy and loss change throughout training
def plot_history(history):

    # Initialize subplots within new figure window and set figure size
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)

    # Generate first subplot
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label = 'train')
    ax1.plot(history.history['val_loss'], label = 'test')
    ax1.set_ylabel('Loss')
    max_loss = max(history.history['loss'] + history.history['val_loss'])
    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation']) 

    # Generate second subplot
    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'],  label = 'train')
    ax2.plot(history.history['val_accuracy'], label = 'test')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    # Save figure and close figure window
    plt.savefig("./History_Plots/history_plot.png")
    plt.close()

    # Print message indicating plot was saved
    print("History plot saved")

# Function that retrieves actual and predicted values from a dataset
def get_actual_predicted_labels(dataset, model): 

    # Obtain actual and predicted values and format them
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)
    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    # return actual and predicted values
    return actual, predicted

# Function that plots a confusion matrix
def plot_confusion_matrix(actual, predicted, labels, ds_type):

    # Generate confusion matrix
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize':(12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix of action recognition for ' + ds_type)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # Save figure and close figure window
    plt.savefig(f"./Confusion_Matrices/confusion_matrix_{ds_type}.png")
    plt.close() 
    
    # Print message indicating confusion matrix was saved
    print(f"Confusion matrix for {ds_type} saved")

# Function that calcuates precision and recall values
def calculate_precision_recall(y_actual, y_pred, labels):
  
    # Compute confusion matrix and diagonal of the matrix
    cm = tf.math.confusion_matrix(y_actual, y_pred)
    tp = np.diag(cm)

    # Initialize precision and recall dictionaries
    precision = dict()
    recall = dict()

    # For loop to calculate every precision and recall value
    for i in range(len(labels)):

        # Calculate false positives and false negatives
        col = cm[:, i]
        fp = np.sum(col) - tp[i]
        row = cm[i, :]
        fn = np.sum(row) - tp[i]

        # Calculate precision but avoid division by 0
        if tp[i] + fp != 0:
            precision[labels[i]] = tp[i] / (tp[i] + fp)
        else:
            precision[labels[i]] = None

        # Calculate recall but avoid division by 0
        if tp[i] + fn != 0:
            recall[labels[i]] = tp[i] / (tp[i] + fn)
        else:
            recall[labels[i]] = None
    
    # Return precision and recall dicitonaries
    return precision, recall

# Function that calculates F1 scores
def calculate_F1_scores(precision, recall):

    # Initialize dictionary
    F1_scores = {}

    # Loop through items in precision dictionary
    for key, value in precision.items():

        # Define precision and recall values
        precision_val = value
        recall_val = recall[key]

        # If precision or recall are None, add None to dictionary
        if precision_val is None or recall_val is None:
            F1_scores[key] = None
        else:
            # Calculate F1 score and add to dicitonary
            F1_scores[key] = 2 * ((precision_val * recall_val)/(precision_val + recall_val))

    # Return dicitonary
    return F1_scores

# Function that prints classification metrics
def print_classification_metrics(model_accuracy, precision, recall, F1_scores):

    # Print line for readability
    print("------------------------------------------------------------------------")

    # Print model accuracy
    print("Model accuracy:\n")
    print(model_accuracy)

    # Loop through precision values and print them
    print("\nPrecision values:\n")
    for key, value in precision.items():
        print(f"{key}: {value}")
    
    # Loop through recall values and print them
    print("\nRecall values:\n")
    for key, value in recall.items():
        print(f"{key}: {value}")

    # Loop through F1 scores and print them
    print("\nF1 scores:\n")
    for key, value in F1_scores.items():
        print(f"{key}: {value}")

    # Print line for readability
    print("------------------------------------------------------------------------")

# --------------------------------------- END OF MODEL EVALUATION CODE -------------------------------------------------