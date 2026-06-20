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

from tqdm import tqdm
import random
import pathlib
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
from gtda.images import ImageToPointCloud
import gudhi as gd
from gudhi.representations import PersistenceImage

# Defining test settings
num_categories = 1
splits = {"train": 70, "val": 10, "test": 20}
epochs = 1
height = 112
width = 112
n_frames = 10
batch_size = 8

# Choose test from list
tests = ["Baseline", "Concatenation", "3 Channel Concatenation"]
chosen_test = tests[2]

def main():

    # Defining path to video data
    UCF101_dir = pathlib.Path('./UCF101')

    # Calculate steps per epoch and validation steps
    steps_per_epoch = (splits['train'] * num_categories) // batch_size
    validation_steps = (splits['val'] * num_categories) // batch_size
    
    # Create subset directories
    subset_dirs = create_subset_dirs(num_categories = num_categories, UCF101_dir = UCF101_dir, splits = splits)

    # Define output signature
    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))
    
    # Generate training, validation and testing datasets
    train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['train'], n_frames, training=True), 
                                              output_signature = output_signature)
    val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['val'], n_frames), output_signature = output_signature)
    test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['test'], n_frames), output_signature = output_signature)

    # Test baseline model if selected
    if chosen_test == "Baseline":
        test_baseline_model(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds)

    # Test concatenation based fusion if selected
    elif chosen_test == "Concatenation":
        test_concatenation_based_fusion(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds)

    # Test 3 channel concatenation based fusion if selected
    elif chosen_test == "3 Channel Concatenation":
        test_3_channel_concatenation_based_fusion(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds)

    return 

# ----------------------------------  DATA LOADING AND PREPROCESSING CODE --------------------------------------------

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

        # Add 1 to count
        category_count += 1

        # Define path to category
        category_path = os.path.join(UCF101_dir, category)

        # If path leads to a directory, add category to dictionary and copy all videos into a list under that category
        if os.path.isdir(category_path):
            category_dict[category] = []

            for video in os.listdir(category_path):
                category_dict[category].append(video)

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
def frames_from_video_file(video_path, n_frames, output_size = (height,width), frame_step = 15):

    # Initalize output list 
    result = []

    # Initalize new VideoCapture object
    src = cv2.VideoCapture(str(video_path))  

    # Get number of frames in video
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    # Define variable for needed number of frames
    need_length = 1 + (n_frames - 1) * frame_step

    # If needed number of frames is longer than video length, set start to 0
    if need_length > video_length:
        start = 0

    # Else, define a maximum starting position and set start to a random number between that and 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    # Set starting position using start variable
    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Read video frame
    ret, frame = src.read()

    # Format frame and append to output list
    result.append(format_frames(frame, output_size))

    # Loop one time for each remaining needed frame
    for _ in range(n_frames - 1):

        # Loop a specified number of times to skip frames
        for _ in range(frame_step):

            # Read video frame
            ret, frame = src.read()

        # If frame was extracted, format and add to result, otherwise add fully black image to result
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
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
    # Resize video to one eighth of its current hight and width
    x = ResizeVideo(height // 8, width // 8)(x)

    # Add residual block with a specified number of filters and specific kernel size
    x = add_residual_block(x, 64, (3, 3, 3))
    # Resize video to one sixteenth of its current hight and width
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


# ------------------------------------ TOPOLOGICAL FEATURE EXTRACTION CODE ---------------------------------------------

# Function that generates point clouds from the data
def generate_point_clouds(video_frames):

    # Create object for ImageToPointCloud class
    itpc = ImageToPointCloud()

    # Initalize list of binary frames
    binary_frames = []

    # For loop to access individual frames
    for frame in video_frames:

        # If baseline test is selected, preprocess data
        if chosen_test == "Baseline":
        
            # Convert to NumPy ndarray
            np_frame = frame.numpy()

            # Convert frame to greyscale form
            grey_image = cv2.cvtColor(np_frame, cv2.COLOR_RGB2GRAY)

            # Rescale grey image
            frame = (grey_image * 255).astype(np.uint8)

        # Convert frame to binary image
        ret, binary_image = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

        # Append binary image to list
        binary_frames.append(binary_image)

    # Convert list to array
    binary_frames_arr = np.array(binary_frames)

    # Generate point clouds from binary frames
    point_clouds = itpc.fit_transform(binary_frames_arr, y=None)

    # Return point clouds
    return point_clouds

# Function that generates persistence diagrams and simplex trees from point clouds
def generate_pds_sts(point_clouds):

    # Initialize lists of persistence diagrams and simplex trees
    persistence_diagrams = []
    simplex_trees = []
    
    # Loop through each point cloud in the input
    for point_cloud in point_clouds:

        # Define number of points in point cloud
        num_points = point_cloud.shape[0]

        # If point cloud has zero points, append a "None" value to lists and continue loop
        if num_points == 0:
            persistence_diagrams.append(None)
            simplex_trees.append(None)
            continue
        
        # If point cloud has more than 1000 points, resize to 1000 by choosing 1000 points at random
        if num_points > 1000:
            indices = np.random.choice(range(0, num_points), 1000)
            point_cloud = point_cloud[indices]

        # Create a simplex tree from point cloud
        simplex_tree = gd.AlphaComplex(points=point_cloud).create_simplex_tree()

        # Create persistence diagram from simplex tree
        persistence_diagram = simplex_tree.persistence()

        # Append persistence diagram to list
        persistence_diagrams.append(persistence_diagram)

        # Append simplex tree to list
        simplex_trees.append(simplex_tree)

    # Return lists
    return simplex_trees, persistence_diagrams

# Function that generates persistence images from the data
def generate_persistence_images(simplex_trees):

    # Initalize list of persistence images
    persistence_images = []

    # Loop through inputted simplex trees
    for tree in simplex_trees:

        # If tree has "None" value, append None to list and continue
        if tree is None:
            persistence_images.append(None)
            continue

        # Create persistence image
        persitence_image = PersistenceImage(bandwidth=1, weight=lambda x: x[1]**2,
                                        im_range=[0,18,0,18], resolution=[height,width])
        persitence_image = persitence_image.fit_transform([tree.persistence_intervals_in_dimension(1)])

        # Append image to list
        persistence_images.append(persitence_image)

    # Return persistence images
    return persistence_images

# Function that takes a dataset and returns topological features of the data
def tf_extraction_ds(x_ds, name):
    
    # Initialize dictionary and count variable
    frames_dict = {}
    count = 0

    # Loop through all video frames in the dataset to create a frame dicitonary
    for frames, label in x_ds:
        frames_dict[f"{label}.{count}"] = frames
        count += 1
    
    # Initialize all dictionaries and list
    point_cloud_dict = {}
    simplex_trees_dict = {}
    persistence_diagrams_dict = {}
    persistence_images_list = []

    # Loop through frame dictionary to create point cloud dictionary
    for label, frames in tqdm(frames_dict.items(), desc= f"{name} - Generating point clouds"):
        point_cloud_dict[label] = generate_point_clouds(frames)

    # Loop through point cloud dictionary to create simplex tree and persistence diagram dictionaries
    for label, point_clouds in tqdm(point_cloud_dict.items(), desc= f"{name} - Generating simplex trees and pesistence diagrams"):
        simplex_trees, persistence_diagrams = generate_pds_sts(point_clouds)
        simplex_trees_dict[label] = simplex_trees
        persistence_diagrams_dict[label] = persistence_diagrams
    
    # Loop through simplex tree dictionary to create persistence images list
    for label, simplex_trees in tqdm(simplex_trees_dict.items(), desc= f"{name} - Generating persistence images"):

        # Generate batch of persistence images
        persistence_images_batch = generate_persistence_images(simplex_trees)

        # Add persistence images to output list
        persistence_images_list.extend(persistence_images_batch)

    # Return dicitonaries and list
    return persistence_images_list

# Function that takes a list of frames and returns topological features of the data
def tf_extraction_list(frame_list, name):

    # Generate point clouds
    print(f"{name} - Genrating point clouds...")
    point_clouds = generate_point_clouds(frame_list)
    print("Done")

    # Generate simplex trees and persistence diagrams
    print(f"{name} - Generating simplex trees and pesistence diagrams...")
    simplex_trees, persistence_diagrams = generate_pds_sts(point_clouds)
    print("Done")

    # Generate persistence images
    print(f"{name} - Generating persistence images...")
    persistence_images_list = generate_persistence_images(simplex_trees)
    print("Done")

    # Return list of persistence images
    return persistence_images_list

# ----------------------------------- END OF TOPOLOGICAL FEATURE EXTRACTION CODE --------------------------------------

# ---------------------------------------- CONCATENATION BASED FUSION CODE --------------------------------------------

# Define Concatenated_frame_generator class that generates concatenations of video frames and persistence images
class Concatenated_frame_generator:

    # __init__ function to initialize instance attributes
    def __init__(self, x_ds, pis_list):
        self.x_ds = x_ds
        self.pis_list = pis_list

    # __call__ function that yields video frames with their respective label
    def __call__(self):
        
        # Initialize index tracker
        index = 0

        # Loop through frames and labels within inputted dataset
        for frames, label in self.x_ds:
            
            # Initialize list of concatenated frames
            concatenated_frames = []

            # Loop through frames
            for frame in frames:
                
                # Define current persistence image with current index
                cur_pi = self.pis_list[index]

                # If current persistence image is "None," set current persistence image tensor to a tensor of desired shape filled with zeros
                if cur_pi is None:
                    cur_pi_tensor = tf.zeros((height,width,1), dtype = tf.float32)

                else:
                    # Convert current persistence image to a tensor
                    cur_pi_tensor = tf.convert_to_tensor(cur_pi, dtype = tf.float32)

                    # Reshape tensor to desired shape
                    cur_pi_tensor = tf.reshape(cur_pi_tensor, [height, width, 1])

                # Concatenate video frame and current persistence image tensor
                concatenated_frame = tf.concat([frame, cur_pi_tensor], axis = -1)

                # Append concatenated frame to concatenated frames list
                concatenated_frames.append(concatenated_frame)
                
                # Set index for next loop
                index += 1

            # Make concatenated frames list an array
            concatenated_frames_array = np.array(concatenated_frames)

            # Yield concatenated frames array and its respective label
            yield concatenated_frames_array, label

# Funciton that splits video frames into frames representing one colour channel each
def split_frames(x_ds):

    # Initialize lists
    red = []
    blue = []
    green = []

    # Loop through frames within inputted dataset
    for frames, label in x_ds:

        # Loop through individual frames
        for frame in frames:

            # Convert to NumPy ndarray
            np_frame = frame.numpy()

            # Rescale image
            np_frame = (np_frame * 255).astype(np.uint8)

            # Split into colour channels
            r, g, b = cv2.split(np_frame)

            # Append channels to lists
            red.append(r)
            blue.append(b)
            green.append(g)
    
    # Define all black channel with the same shape as colour channels
    black = np.zeros(red[0].shape, np.uint8)

    # Loop through colour channel lists
    for i in range(len(red)):
        
        # Select specific channel
        r = red[i]
        b = blue[i]
        g = green[i]

        # Merge colour channels with fully black channel
        new_r = cv2.merge([r, black, black])
        new_g = cv2.merge([black, g, black])
        new_b = cv2.merge([black, black, b])

        # Update channels to new values
        red[i] = new_r
        blue[i] = new_b
        green[i] = new_g

    # Return lists
    return red, green, blue

class Three_channel_concatenated_frame_generator:

    # __init__ function to initialize instance attributes
    def __init__(self, x_ds, red_pis_list, green_pis_list, blue_pis_list):
        self.x_ds = x_ds
        self.red_pis_list = red_pis_list
        self.green_pis_list = green_pis_list
        self.blue_pis_list = blue_pis_list

    # __call__ function that yields video frames with their respective label
    def __call__(self):
        
        # Initialize index tracker
        index = 0

        # Loop through frames and labels within inputted dataset
        for frames, label in self.x_ds:
            
            # Initialize list of concatenated frames
            concatenated_frames = []

            # Loop through frames
            for frame in frames:
                
                # Define current persistence images with current index
                red_cur_pi = self.red_pis_list[index]
                green_cur_pi = self.green_pis_list[index]
                blue_cur_pi = self.blue_pis_list[index]

                # If current persistence image is "None," set current persistence image tensor to a tensor of desired shape filled with zeros
                if red_cur_pi is None or green_cur_pi is None or blue_cur_pi is None:
                    cur_pi_tensor = tf.zeros((height,width,3), dtype = tf.float32)

                    # Concatenate video frame and current persistence image tensor
                    concatenated_frame = tf.concat([frame, cur_pi_tensor], axis = -1)

                else:
                    # Convert current persistence images to tensors
                    red_cur_pi_tensor = tf.convert_to_tensor(red_cur_pi, dtype = tf.float32)
                    green_cur_pi_tensor = tf.convert_to_tensor(green_cur_pi, dtype = tf.float32)
                    blue_cur_pi_tensor = tf.convert_to_tensor(blue_cur_pi, dtype = tf.float32)

                    # Reshape tensors to desired shape
                    red_cur_pi_tensor = tf.reshape(red_cur_pi_tensor, [height, width, 1])
                    green_cur_pi_tensor = tf.reshape(green_cur_pi_tensor, [height, width, 1])
                    blue_cur_pi_tensor = tf.reshape(blue_cur_pi_tensor, [height, width, 1])

                    concatenated_frame = tf.concat([frame, red_cur_pi_tensor, green_cur_pi_tensor, blue_cur_pi_tensor], axis = -1)

                # Append concatenated frame to concatenated frames list
                concatenated_frames.append(concatenated_frame)
                
                # Set index for next loop
                index += 1

            # Make concatenated frames list an array
            concatenated_frames_array = np.array(concatenated_frames)

            # Yield concatenated frames array and its respective label
            yield concatenated_frames_array, label

# ------------------------------------- END OF CONCATENATION BASED FUSION CODE ----------------------------------------

# ------------------------------------------------ TEST BASED CODE ----------------------------------------------------

# Function to test baseline video classificaion model
def test_baseline_model(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds):

    # Make versions of datasets that repeat for training
    repeat_train_ds = train_ds.repeat().batch(batch_size)
    repeat_val_ds = val_ds.repeat().batch(batch_size)

    # Batch data into desired sizes
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    # Define input shape
    input_shape = (None, n_frames, height, width, 3)

    # Call function to create the 3D CNN model
    model = create_3D_CNN(train_ds, input_shape)

    # Prepare model for training with the Adam optimizer and SparseCategoricalCrossentropy loss function
    model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.0001), 
                metrics = ['accuracy'])
    
    # Train the model and obtain model history using model.fit()
    history = model.fit(x = repeat_train_ds, epochs = epochs, validation_data = repeat_val_ds, steps_per_epoch = steps_per_epoch, 
                        validation_steps = validation_steps)
    
    # Call function to plot history of model training performance
    plot_history(history)

    # Evaluate model to get accuracy and loss values
    model_accuracy_and_loss = model.evaluate(test_ds, return_dict=True)

    # Obtain rounded model accuracy value
    model_accuracy = round(model_accuracy_and_loss["accuracy"], 2)

    # Use FrameGenerator class to obtain class labels from training data
    fg = FrameGenerator(subset_dirs['train'], n_frames, training=True)
    labels = list(fg.class_ids_for_name.keys())

    # Call funciton to get actual and predicted values from the training dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(train_ds, model)
    plot_confusion_matrix(actual, predicted, labels, 'training')

    # Call funciton to get actual and predicted values from the test dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(test_ds, model)
    plot_confusion_matrix(actual, predicted, labels, 'test')

    # Call function to calculate precision and recall values
    precision, recall = calculate_precision_recall(actual, predicted, labels)

    # Call function to calculate F1 scores
    F1_scores = calculate_F1_scores(precision, recall)

    # Call function to print classificaiton metrics
    print_classification_metrics(model_accuracy, precision, recall, F1_scores)

    return

# Function to test concatenation based fusion
def test_concatenation_based_fusion(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds):

    # Get topological features from the datasets
    train_persistence_images_list = tf_extraction_ds(train_ds, "Training")
    val_persistence_images_list = tf_extraction_ds(val_ds, "Validation")
    test_persistence_images_list = tf_extraction_ds(test_ds, "Test")
    
    # Define output signature
    output_signature = (tf.TensorSpec(shape = (None, None, None, 4), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))

    # Generate training, validation and testing datasets
    train_concatenated_frames = tf.data.Dataset.from_generator(Concatenated_frame_generator(train_ds, train_persistence_images_list), 
                                                                output_signature = output_signature)
    val_concatenated_frames = tf.data.Dataset.from_generator(Concatenated_frame_generator(val_ds, val_persistence_images_list), 
                                                                output_signature = output_signature)
    test_concatenated_frames = tf.data.Dataset.from_generator(Concatenated_frame_generator(test_ds, test_persistence_images_list), 
                                                                output_signature = output_signature)

    # Make versions of datasets that repeat for training
    repeat_train_concatenated_frames = train_concatenated_frames.repeat().batch(batch_size)
    repeat_val_concatenated_frames = val_concatenated_frames.repeat().batch(batch_size)

    # Batch data into desired sizes
    train_concatenated_frames = train_concatenated_frames.batch(batch_size)
    val_concatenated_frames = val_concatenated_frames.batch(batch_size)
    test_concatenated_frames = test_concatenated_frames.batch(batch_size)

    # Define input shape
    input_shape = (None, n_frames, height, width, 4)

    # Call function to create the 3D CNN model
    model = create_3D_CNN(train_concatenated_frames, input_shape)

    # Prepare model for training with the Adam optimizer and SparseCategoricalCrossentropy loss function
    model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.0001), metrics = ['accuracy'])

    # Train the model and obtain model history using model.fit()
    history = model.fit(x = repeat_train_concatenated_frames, epochs = epochs, validation_data = repeat_val_concatenated_frames, 
                        steps_per_epoch = steps_per_epoch, validation_steps = validation_steps)

    # Call function to plot history of model training performance
    plot_history(history)

    # Evaluate model to get accuracy and loss values
    model_accuracy_and_loss = model.evaluate(test_concatenated_frames, return_dict=True)

    # Obtain rounded model accuracy value
    model_accuracy = round(model_accuracy_and_loss["accuracy"], 2)

    # Use FrameGenerator class to obtain class labels from training data
    fg = FrameGenerator(subset_dirs['train'], n_frames, training=True)
    labels = list(fg.class_ids_for_name.keys())

    # Call funciton to get actual and predicted values from the training dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(train_concatenated_frames, model)
    plot_confusion_matrix(actual, predicted, labels, 'training')

    # Call funciton to get actual and predicted values from the test dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(test_concatenated_frames, model)
    plot_confusion_matrix(actual, predicted, labels, 'test')

    # Call function to calculate precision and recall values
    precision, recall = calculate_precision_recall(actual, predicted, labels)

    # Call function to calculate F1 scores
    F1_scores = calculate_F1_scores(precision, recall)

    # Call function to print classificaiton metrics
    print_classification_metrics(model_accuracy, precision, recall, F1_scores)

    return

# Function to test 3 channel concatenation based fusion
def test_3_channel_concatenation_based_fusion(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds):

    # Split frames into images representing single colour channels
    red_train_list, green_train_list, blue_train_list = split_frames(train_ds)
    red_val_list, green_val_list, blue_val_list = split_frames(val_ds)
    red_test_list, green_test_list, blue_test_list = split_frames(test_ds)

    # Get topological features of training data
    red_train_pi_list = tf_extraction_list(red_train_list, "Training, red")
    green_train_pi_list = tf_extraction_list(green_train_list, "Training, green")
    blue_train_pi_list = tf_extraction_list(blue_train_list, "Training, blue")

    # Get topological features of validaiton data
    red_val_pi_list = tf_extraction_list(red_val_list, "Validation, red")
    green_val_pi_list = tf_extraction_list(green_val_list, "Validation, green")
    blue_val_pi_list = tf_extraction_list(blue_val_list, "Validation, blue")

    # Get topological features of testing data
    red_test_pi_list = tf_extraction_list(red_test_list, "Test, red")
    green_test_pi_list = tf_extraction_list(green_test_list, "Test, green")
    blue_test_pi_list = tf_extraction_list(blue_test_list, " Test, blue")

    # Define output signature
    output_signature = (tf.TensorSpec(shape = (None, None, None, 6), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))

    # Generate training, validation, and testing datasets
    train_concatenated_frames = tf.data.Dataset.from_generator(Three_channel_concatenated_frame_generator(train_ds, red_train_pi_list, 
                                                            green_train_pi_list, blue_train_pi_list), output_signature = output_signature)
    val_concatenated_frames = tf.data.Dataset.from_generator(Three_channel_concatenated_frame_generator(val_ds, red_val_pi_list, 
                                                            green_val_pi_list, blue_val_pi_list), output_signature = output_signature)
    test_concatenated_frames = tf.data.Dataset.from_generator(Three_channel_concatenated_frame_generator(test_ds, red_test_pi_list, 
                                                            green_test_pi_list, blue_test_pi_list), output_signature = output_signature)
    
    # Make versions of datasets that repeat for training
    repeat_train_concatenated_frames = train_concatenated_frames.repeat().batch(batch_size)
    repeat_val_concatenated_frames = val_concatenated_frames.repeat().batch(batch_size)

    # Batch data into desired sizes
    train_concatenated_frames = train_concatenated_frames.batch(batch_size)
    val_concatenated_frames = val_concatenated_frames.batch(batch_size)
    test_concatenated_frames = test_concatenated_frames.batch(batch_size)

    # Define input shape
    input_shape = (None, n_frames, height, width, 6)

    # Call function to create the 3D CNN model
    model = create_3D_CNN(train_concatenated_frames, input_shape)

    # Prepare model for training with the Adam optimizer and SparseCategoricalCrossentropy loss function
    model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.0001), metrics = ['accuracy'])

    # Train the model and obtain model history using model.fit()
    history = model.fit(x = repeat_train_concatenated_frames, epochs = epochs, validation_data = repeat_val_concatenated_frames, 
                        steps_per_epoch = steps_per_epoch, validation_steps = validation_steps)

    # Call function to plot history of model training performance
    plot_history(history)

    # Evaluate model to get accuracy and loss values
    model_accuracy_and_loss = model.evaluate(test_concatenated_frames, return_dict=True)

    # Obtain rounded model accuracy value
    model_accuracy = round(model_accuracy_and_loss["accuracy"], 2)

    # Use FrameGenerator class to obtain class labels from training data
    fg = FrameGenerator(subset_dirs['train'], n_frames, training=True)
    labels = list(fg.class_ids_for_name.keys())

    # Call funciton to get actual and predicted values from the training dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(train_concatenated_frames, model)
    plot_confusion_matrix(actual, predicted, labels, 'training')

    # Call funciton to get actual and predicted values from the test dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(test_concatenated_frames, model)
    plot_confusion_matrix(actual, predicted, labels, 'test')

    # Call function to calculate precision and recall values
    precision, recall = calculate_precision_recall(actual, predicted, labels)

    # Call function to calculate F1 scores
    F1_scores = calculate_F1_scores(precision, recall)

    # Call function to print classificaiton metrics
    print_classification_metrics(model_accuracy, precision, recall, F1_scores)

    return

# -------------------------------------------- END OF TEST BASED CODE -------------------------------------------------

if __name__ == "__main__":
    main()