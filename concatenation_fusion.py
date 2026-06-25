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

from utils import create_subset_dirs
from utils import FrameGenerator
from utils import create_3D_CNN
from utils import plot_history
from utils import get_actual_predicted_labels
from utils import plot_confusion_matrix
from utils import calculate_precision_recall
from utils import calculate_F1_scores
from utils import print_classification_metrics
from utils import get_test_settings

# Get test settings
num_categories, splits, epochs, height, width, n_frames, batch_size, steps_per_epoch, validation_steps = get_test_settings()

# Choose test from list
tests = ["Concatenation", "3 Channel Concatenation"]
chosen_test = tests[1]

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

    # Test concatenation based fusion if selected
    if chosen_test == "Concatenation":
        test_concatenation_based_fusion(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds)

    # Test 3 channel concatenation based fusion if selected
    elif chosen_test == "3 Channel Concatenation":
        test_3_channel_concatenation_based_fusion(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds)

    return 

# ------------------------------------ TOPOLOGICAL FEATURE EXTRACTION CODE ---------------------------------------------

# Function that generates point clouds from the data
def generate_point_clouds(video_frames, chosen_test):

    # Create object for ImageToPointCloud class
    itpc = ImageToPointCloud()

    # Initalize list of binary frames
    binary_frames = []

    # For loop to access individual frames
    for frame in video_frames:

        # If concatenation test is selected, preprocess data
        if chosen_test == "Concatenation":
        
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
        point_cloud_dict[label] = generate_point_clouds(frames, chosen_test)

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
    point_clouds = generate_point_clouds(frame_list, chosen_test)
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