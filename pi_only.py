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

import pathlib
import cv2
import numpy as np
import tensorflow as tf
import keras
from gtda.images import ImageToPointCloud
import gudhi as gd
from gudhi.representations import PersistenceImage
import open3d
import random

from utils import create_subset_dirs
from utils import FrameGenerator
from utils import create_3D_CNN
from utils import plot_history
from utils import get_actual_predicted_labels
from utils import plot_confusion_matrix
from utils import calculate_precision_recall
from utils import calculate_F1_scores
from utils import get_test_settings
from utils import early_stoppage
from utils import create_metrics_test_settings_spreadsheet
from utils import save_results
from utils import frames_from_video_file

import math

# Get test settings
num_categories, splits, epochs, height, width, n_frames, batch_size, steps_per_epoch, validation_steps = get_test_settings()

# Define chosen test
chosen_test = "PI_only"

# Choose whether to save results to folder with specific name
save_output = True
results_file_name = "test"

def main():

    # Defining path to video data
    UCF101_dir = pathlib.Path('./UCF101')
    
    # Create subset directories
    subset_dirs = create_subset_dirs(num_categories = num_categories, UCF101_dir = UCF101_dir)

    # Obtain early stoppage callback
    callback = early_stoppage()

    # Define output signature
    output_signature = (tf.TensorSpec(shape = (None, None, None, 1), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))

    # Generate training, validation and testing datasets
    train_ds = tf.data.Dataset.from_generator(PI_generator(subset_dirs['train'], n_frames, training=True), 
                                            output_signature = output_signature)
    val_ds = tf.data.Dataset.from_generator(PI_generator(subset_dirs['val'], n_frames), output_signature = output_signature)
    test_ds = tf.data.Dataset.from_generator(PI_generator(subset_dirs['test'], n_frames), output_signature = output_signature)

    print("Datasets created")

    # Test pi only
    test_pi_only(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds, callback)

    # Save results to folder if wanted
    if save_output:
        save_results(results_file_name)

    return 

# ------------------------------------ TOPOLOGICAL FEATURE EXTRACTION CODE ---------------------------------------------

# Function that downsamples a given point cloud
def downsample_point_cloud(point_cloud):

    # Define number of points in point cloud
    num_points = point_cloud.shape[0]

    # If point cloud has zero points, return "None"
    if num_points == 0:
        return None
    
    # Create object for PointCloud class
    pcd = open3d.geometry.PointCloud()

    # Make point cloud 3d by adding a third dimension with zeros
    three_dimensional_pc = np.column_stack((point_cloud, np.zeros(point_cloud.shape[0])))

    # Define point cloud points
    pcd.points = open3d.utility.Vector3dVector(three_dimensional_pc)

    # Downsample point cloud using a voxel of size 1.5
    downsampled_point_cloud = pcd.voxel_down_sample(voxel_size=1.5)

    # Convert point cloud to an array
    np_point_cloud = np.asarray(downsampled_point_cloud.points)

    # Return downsampled point cloud
    return np_point_cloud

# Function that generates point clouds from the data
def generate_point_clouds(video_frames, chosen_test):

    # Create object for ImageToPointCloud class
    itpc = ImageToPointCloud()

    # Initalize list of binary frames
    binary_frames = []

    # For loop to access individual frames
    for frame in video_frames:

        # Convert frame to greyscale form
        grey_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Rescale grey image
        frame = (grey_image * 255).astype(np.uint8)

        # Convert frame to binary image
        binary_image = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Append binary image to list
        binary_frames.append(binary_image)

    # Convert list to array
    binary_frames_arr = np.array(binary_frames)

    # Generate point clouds from binary frames
    point_clouds = itpc.fit_transform(binary_frames_arr, y=None)

    # Create list of downsampled point clouds
    downsampled_pcs = [downsample_point_cloud(point_cloud) for point_cloud in point_clouds]

    # Return point clouds
    return downsampled_pcs

# Function that generates a simplex tree from a point cloud
def generate_st(point_cloud):

    # If point cloud has "None" value, return None
    if point_cloud is None:
        return None

    # Create a simplex tree from point cloud
    simplex_tree = gd.AlphaComplex(points=point_cloud).create_simplex_tree()

    # Create persistence diagram from simplex tree
    persistence_diagram = simplex_tree.persistence()

    # Return simplex tree
    return simplex_tree

# Function that generates a persistence image from a simplex tree
def generate_persistence_image(simplex_tree):

    # If tree has "None" value, return None
    if simplex_tree is None:
        return None
    
    # Obtain intervals in dimensions 1 and 0 of simplex tree
    intervals1 = simplex_tree.persistence_intervals_in_dimension(1)
    intervals0 = simplex_tree.persistence_intervals_in_dimension(0)

    # Initialize filtered intervals list
    filtered_intervals = []

    # Loop through intervals in dimension 1 and add those whose birth and death are over
    # 0.75 apart to filtered intervals list
    for interval in intervals1:
        if interval[1] - interval[0] > 0.75:
            filtered_intervals.append(interval)

    # Loop through intervals in dimension 0 and add those whose birth and death are over
    # 0.75 apart and whose death value is not infinite to filtered intervals list
    for interval in intervals0:
        if not math.isinf(interval[1]) and interval[1] - interval[0] > 0.75:
            filtered_intervals.append(interval)

    # Make filtered intervals list an array
    filtered_intervals = np.array(filtered_intervals)

    # Create persistence image
    persistence_image = PersistenceImage(bandwidth=0.8, weight=lambda x: x[1]**2,
                                    im_range=[0,8,0,8], resolution=[height,width])
    persistence_image = persistence_image.fit_transform([filtered_intervals])

    # Return persistence image
    return persistence_image

# ----------------------------------- END OF TOPOLOGICAL FEATURE EXTRACTION CODE --------------------------------------

# ----------------------------------------------- PI ONLY BASED CODE --------------------------------------------------

# Define PI_generator class that generates persistence images with corresponding label
class PI_generator:

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

    # __call__ function that yields persistence images with the respective label
    def __call__(self):

        # Call function to get video paths and class names
        video_paths, classes = self.get_files_and_class_names()

        # Create a list of tuples containing video paths and their respective class
        pairs = list(zip(video_paths, classes))

        # If training is True, mix up pairs within pairs list
        if self.training:
            random.seed(0)
            random.shuffle(pairs)

        # Loop through each tuple in pairs list
        for path, name in pairs:

            # Get video frames
            video_frames = frames_from_video_file(path, self.n_frames) 

            # Get label
            label = self.class_ids_for_name[name]
            
            # Initialize lists
            simplex_trees_list = []
            persistence_images_list = []
            
            # Get point clouds from video frames
            point_clouds = generate_point_clouds(video_frames, chosen_test)

            # Generate simplex trees from point clouds and add to list
            for point_cloud in point_clouds:
                simplex_tree = generate_st(point_cloud)
                simplex_trees_list.append(simplex_tree)

            # Loop through simplex tree list to generate persistence images list
            for simplex_tree in simplex_trees_list:

                # Generate persistence image
                persistence_image = generate_persistence_image(simplex_tree)

                # Add persistence image to output list
                persistence_images_list.append(persistence_image)

            # Initialize list
            reshaped_pis = []

            # Loop through persistence images to reshape them
            for pi in persistence_images_list:

                # If current persistence image is "None," set current persistence image to a tensor of desired shape filled with zeros
                if pi is None:
                    pi = tf.zeros((height,width,1), dtype = tf.float32)

                else:
                    # Reshape to desired shape
                    pi = tf.reshape(pi, [height, width, 1])

                # Add reshaped pi to list
                reshaped_pis.append(pi)

            # Check that video frames are aligned with corresponding persistence images
            if len(video_frames) != len(reshaped_pis):
                print("ERROR: NUMBER OF VIDEO FRAMES DOES NOT MATCH NUMBER OF GENERATED PERSISTENCE IMAGES")

            # Yield video frames, persistence images and the respective label
            yield reshaped_pis, label

# Define Standardized_PI_generator class that generates persistence images
class Standardized_PI_generator:

    # __init__ function to initialize instance attributes
    def __init__(self, pi_list):
        self.pi_list = pi_list

    # __call__ function that yields persistence images and their respective label
    def __call__(self):

        # Loop through persistence images and labels within inputted list
        for (pis, label) in self.pi_list:

            # Yield persistence images and respective label
            yield pis, label

# ------------------------------------------- END OF PI ONLY BASED CODE -----------------------------------------------

# ------------------------------------------------ TEST BASED CODE ----------------------------------------------------

# Function to test pi only
def test_pi_only(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds, callback):

    # Initialize list
    train_persistence_images = []

    # Loop through training dataset
    for pis, label in train_ds:

        # Loop through persistence images and add them to list
        for pi in pis:
            if pi is not None:
                pi = pi.numpy()
                train_persistence_images.append(pi)

    # Obtain all pixel values from persistence images
    concatenated_pi_list = np.concatenate([pi.ravel() for pi in train_persistence_images if pi is not None])

    # Calculate IQR and median of pixel values
    IQR = np.percentile(concatenated_pi_list, 75) - np.percentile(concatenated_pi_list, 25)
    median_pi_val = np.median(concatenated_pi_list)

    # Initialize lists
    train_standardized_pi_list = []
    val_standardized_pi_list = []
    test_standardized_pi_list = []

    # Loop through frames, persistence images and labels within training dataset
    for pis, label in train_ds:

        # Initialize standardized persistence images list
        standardized_pis = []

        # Loop through frames and persistence images within dataset
        for pi in pis:

            # Convert persistence image to numpy
            numpy_pi = pi.numpy()

            # Standardize persistence image
            standardized_pi = (numpy_pi - median_pi_val) / IQR

            # Convert persistence image to a tensor
            pi_tensor = tf.convert_to_tensor(standardized_pi, dtype = tf.float32)

            # Add frame to list
            standardized_pis.append(pi_tensor)

        # Make standardized persistence images list an array
        standardized_pis_array = np.array(standardized_pis)

        # Add standardized persistence images array and its respective label to list
        train_standardized_pi_list.append((standardized_pis_array, label))

    # Loop through frames, persistence images and labels within training dataset
    for pis, label in val_ds:

        # Initialize standardized persistence images list
        standardized_pis = []

        # Loop through frames and persistence images within dataset
        for pi in pis:

            # Convert persistence image to numpy
            numpy_pi = pi.numpy()

            # Standardize persistence image
            standardized_pi = (numpy_pi - median_pi_val) / IQR

            # Convert persistence image to a tensor
            pi_tensor = tf.convert_to_tensor(standardized_pi, dtype = tf.float32)

            # Add frame to list
            standardized_pis.append(pi_tensor)

        # Make standardized persistence images list an array
        standardized_pis_array = np.array(standardized_pis)

        # Add standardized persistence images array and its respective label to list
        val_standardized_pi_list.append((standardized_pis_array, label))

    # Loop through frames, persistence images and labels within training dataset
    for pis, label in test_ds:

        # Initialize standardized persistence images list
        standardized_pis = []

        # Loop through frames and persistence images within dataset
        for pi in pis:

            # Convert persistence image to numpy
            numpy_pi = pi.numpy()

            # Standardize persistence image
            standardized_pi = (numpy_pi - median_pi_val) / IQR

            # Convert persistence image to a tensor
            pi_tensor = tf.convert_to_tensor(standardized_pi, dtype = tf.float32)

            # Add frame to list
            standardized_pis.append(pi_tensor)

        # Make standardized persistence images list an array
        standardized_pis_array = np.array(standardized_pis)

        # Add standardized persistence images array and its respective label to list
        test_standardized_pi_list.append((standardized_pis_array, label))
        
    # Define output signature
    output_signature = (tf.TensorSpec(shape = (None, None, None, 1), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))

    # Generate training, validation and testing datasets
    train_standardized_pis = tf.data.Dataset.from_generator(Standardized_PI_generator(train_standardized_pi_list), 
                                                                output_signature = output_signature)
    val_standardized_pis = tf.data.Dataset.from_generator(Standardized_PI_generator(val_standardized_pi_list), 
                                                                output_signature = output_signature)
    test_standardized_pis = tf.data.Dataset.from_generator(Standardized_PI_generator(test_standardized_pi_list), 
                                                                output_signature = output_signature)

    # Make versions of datasets that repeat for training
    repeat_train_standardized_pis = train_standardized_pis.repeat().batch(batch_size)
    repeat_val_standardized_pis = val_standardized_pis.repeat().batch(batch_size)

    # Batch data into desired sizes
    train_standardized_pis = train_standardized_pis.batch(batch_size)
    val_standardized_pis = val_standardized_pis.batch(batch_size)
    test_standardized_pis = test_standardized_pis.batch(batch_size)

    # Define input shape
    input_shape = (None, n_frames, height, width, 1)

    # Call function to create the 3D CNN model
    model = create_3D_CNN(train_standardized_pis, input_shape)

    # Prepare model for training with the Adam optimizer and SparseCategoricalCrossentropy loss function
    model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.0001), metrics = ['accuracy'])

    # Train the model and obtain model history using model.fit()
    history = model.fit(x = repeat_train_standardized_pis, epochs = epochs, validation_data = repeat_val_standardized_pis, 
                        steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, callbacks=[callback])

    # Call function to plot history of model training performance
    plot_history(history)

    # Evaluate model to get accuracy and loss values
    model_accuracy_and_loss = model.evaluate(test_standardized_pis, return_dict=True)

    # Obtain model accuracy
    model_accuracy = model_accuracy_and_loss["accuracy"]

    # Use FrameGenerator class to obtain class labels from training data
    fg = FrameGenerator(subset_dirs['train'], n_frames, training=True)
    labels = list(fg.class_ids_for_name.keys())

    # Call funciton to get actual and predicted values from the training dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(train_standardized_pis, model)
    plot_confusion_matrix(actual, predicted, labels, 'training')

    # Call funciton to get actual and predicted values from the test dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(test_standardized_pis, model)
    plot_confusion_matrix(actual, predicted, labels, 'test')

    # Call function to calculate precision and recall values
    precision, recall = calculate_precision_recall(actual, predicted, labels)

    # Call function to calculate F1 scores
    F1_scores = calculate_F1_scores(precision, recall)

    # Call function to create spreadsheet of classification metrics and test settings
    create_metrics_test_settings_spreadsheet(model_accuracy, precision, recall, F1_scores)

    return
# -------------------------------------------- END OF TEST BASED CODE -------------------------------------------------

if __name__ == "__main__":
    main()