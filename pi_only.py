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
import pathlib
import cv2
import numpy as np
import tensorflow as tf
import keras
from gtda.images import ImageToPointCloud
import gudhi as gd
from gudhi.representations import PersistenceImage
import open3d

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

import math

# Get test settings
num_categories, splits, epochs, height, width, n_frames, batch_size, steps_per_epoch, validation_steps = get_test_settings()

# Define chosen test
chosen_test = "PI Only"

# Choose whether to save results to folder with specific name
save_output = True
results_file_name = "test"

def main():

    # Defining path to video data
    UCF101_dir = pathlib.Path('./UCF101')
    
    # Create subset directories
    subset_dirs = create_subset_dirs(num_categories = num_categories, UCF101_dir = UCF101_dir, splits = splits)

    # Define output signature
    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))
    
    # Generate training, validation and testing datasets
    train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['train'], n_frames, training=True), 
                                              output_signature = output_signature)
    val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['val'], n_frames), output_signature = output_signature)
    test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['test'], n_frames), output_signature = output_signature)

    # Obtain early stoppage callback
    callback = early_stoppage()

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

    # Downsample point cloud using a voxel of size 3
    downsampled_point_cloud = pcd.voxel_down_sample(voxel_size=3)

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

        # Convert to NumPy ndarray
        np_frame = frame.numpy()

        # Convert frame to greyscale form
        grey_image = cv2.cvtColor(np_frame, cv2.COLOR_RGB2GRAY)

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
    
    # Code to be revised later
    intervals1 = simplex_tree.persistence_intervals_in_dimension(1)
    intervals0 = simplex_tree.persistence_intervals_in_dimension(0)

    filtered_intervals = []

    for interval in intervals1:
            filtered_intervals.append(interval)

    for interval in intervals0:
        if not math.isinf(interval[1]):
            filtered_intervals.append(interval)

    filtered_intervals = np.array(filtered_intervals)

    filtered_2 = []

    for interval in filtered_intervals:
        if interval[1] - interval[0] > .75:
            filtered_2.append(interval)

    filtered_2 = np.array(filtered_2)

    # Create persistence image
    persistence_image = PersistenceImage(bandwidth=0.8, weight=lambda x: x[1]**2,
                                    im_range=[0,8,0,8], resolution=[height,width])
    persistence_image = persistence_image.fit_transform([filtered_2])

    # Return persistence image
    return persistence_image

# Function that takes a dataset and returns a list of persistence images
def tf_extraction_ds(x_ds, name, training_val):
    
    # Initialize dictionary and count variable
    frames_dict = {}
    count = 0

    # Loop through all video frames in the dataset to create a frame dicitonary
    for frames, label in x_ds:
        frames_dict[f"{label}.{count}"] = frames
        count += 1
    
    # Initialize lists
    point_cloud_list = []
    simplex_trees_list = []
    persistence_images_list = []

    # Loop through frame dictionary to generate point cloud list
    for label, frames in tqdm(frames_dict.items(), desc= f"{name} - Generating point clouds"):
        point_cloud_list.extend(generate_point_clouds(frames, chosen_test))

    # Loop through point cloud list to generate simplex tree list
    for point_cloud in tqdm(point_cloud_list, desc= f"{name} - Generating simplex trees"):
        simplex_tree = generate_st(point_cloud)
        simplex_trees_list.append(simplex_tree)
    
    # Loop through simplex tree list to generate persistence images list
    for simplex_tree in tqdm(simplex_trees_list, desc= f"{name} - Generating persistence images"):

        # Generate persistence image
        persistence_image = generate_persistence_image(simplex_tree)

        # Add persistence image to output list
        persistence_images_list.append(persistence_image)

    # If training is true get values for standardization
    if training_val == True:

        # Obtain all pixel values from persistence images
        concatenated_pi_list = np.concatenate([pi.ravel() for pi in persistence_images_list if pi is not None])

        # Calculate IQR and median of pixel values
        IQR = np.percentile(concatenated_pi_list, 75) - np.percentile(concatenated_pi_list, 25)
        median_pi_val = np.median(concatenated_pi_list)

        # Return persistence images list along with median and IQR of pixel values
        return persistence_images_list, median_pi_val, IQR

    # Return persistence images list
    return persistence_images_list

# ----------------------------------- END OF TOPOLOGICAL FEATURE EXTRACTION CODE --------------------------------------

# ---------------------------------------- PERSISTENCE IMAGE GENERATION CODE --------------------------------------------

# Define Persistence_image_generator class that generates persistence images
class Persistence_image_generator:

    # __init__ function to initialize instance attributes
    def __init__(self, x_ds, pis_list):
        self.x_ds = x_ds
        self.pis_list = pis_list

    # __call__ function that yields persistence images with their respective label
    def __call__(self):
        
        # Initialize index tracker
        index = 0

        # Loop through frames and labels within inputted dataset
        for frames, label in self.x_ds:
            
            # Initialize list of persistence image tensors
            pi_tensor_list = []

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

                # Append persistence image tensor to list
                pi_tensor_list.append(cur_pi_tensor)
                
                # Set index for next loop
                index += 1

            # Make persistence image tensor list into an array
            pi_tensor_array = np.array(pi_tensor_list)

            # Yield persistence image tensor array and its respective label
            yield pi_tensor_array, label

# ------------------------------------- PERSISTENCE IMAGE GENERATION CODE ----------------------------------------

# ------------------------------------------------ TEST BASED CODE ----------------------------------------------------

# Function to test persistence image only test
def test_pi_only(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds, callback):

    # Get topological features from the datasets
    train_persistence_images_list, median, IQR = tf_extraction_ds(train_ds, "Training", training_val = True)
    val_persistence_images_list = tf_extraction_ds(val_ds, "Validation", training_val = False)
    test_persistence_images_list = tf_extraction_ds(test_ds, "Test", training_val = False)

    # Standardize persistence images and make into a list
    train_standardized_pi_list = [(pi - median) / IQR if pi is not None else None for pi in train_persistence_images_list]
    vali_standardized_pi_list = [(pi - median) / IQR if pi is not None else None for pi in val_persistence_images_list]
    test_standardized_pi_list = [(pi - median) / IQR if pi is not None else None for pi in test_persistence_images_list]
    
    # Define output signature
    output_signature = (tf.TensorSpec(shape = (None, None, None, 1), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))

    # Generate training, validation and testing datasets
    train_persistence_images = tf.data.Dataset.from_generator(Persistence_image_generator(train_ds, train_standardized_pi_list), 
                                                                output_signature = output_signature)
    val_persistence_images = tf.data.Dataset.from_generator(Persistence_image_generator(val_ds, vali_standardized_pi_list), 
                                                                output_signature = output_signature)
    test_persistence_images = tf.data.Dataset.from_generator(Persistence_image_generator(test_ds, test_standardized_pi_list), 
                                                                output_signature = output_signature)

    # Make versions of datasets that repeat for training
    repeat_train_persistence_images = train_persistence_images.repeat().batch(batch_size)
    repeat_val_persistence_images = val_persistence_images.repeat().batch(batch_size)

    # Batch data into desired sizes
    train_persistence_images = train_persistence_images.batch(batch_size)
    val_persistence_images = val_persistence_images.batch(batch_size)
    test_persistence_images = test_persistence_images.batch(batch_size)

    # Define input shape
    input_shape = (None, n_frames, height, width, 1)

    # Call function to create the 3D CNN model
    model = create_3D_CNN(train_persistence_images, input_shape)

    # Prepare model for training with the Adam optimizer and SparseCategoricalCrossentropy loss function
    model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.0001), metrics = ['accuracy'])

    # Train the model and obtain model history using model.fit()
    history = model.fit(x = repeat_train_persistence_images, epochs = epochs, validation_data = repeat_val_persistence_images, 
                        steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, callbacks=[callback])

    # Call function to plot history of model training performance
    plot_history(history)

    # Evaluate model to get accuracy and loss values
    model_accuracy_and_loss = model.evaluate(test_persistence_images, return_dict=True)

    # Obtain model accuracy
    model_accuracy = model_accuracy_and_loss["accuracy"]

    # Use FrameGenerator class to obtain class labels from training data
    fg = FrameGenerator(subset_dirs['train'], n_frames, training=True)
    labels = list(fg.class_ids_for_name.keys())

    # Call funciton to get actual and predicted values from the training dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(train_persistence_images, model)
    plot_confusion_matrix(actual, predicted, labels, 'training')

    # Call funciton to get actual and predicted values from the test dataset, then plot confusion matrix
    actual, predicted = get_actual_predicted_labels(test_persistence_images, model)
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