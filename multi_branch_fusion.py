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
import pathlib
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import concatenate

from utils import create_subset_dirs
from utils import FrameGenerator
from utils import plot_history
from utils import get_actual_predicted_labels
from utils import plot_confusion_matrix
from utils import calculate_precision_recall
from utils import calculate_F1_scores
from utils import print_classification_metrics
from utils import ResizeVideo
from utils import add_residual_block
from utils import Conv2Plus1D
from utils import frames_from_video_file
from utils import get_test_settings

from concatenation_fusion import generate_point_clouds
from concatenation_fusion import generate_pds_sts
from concatenation_fusion import generate_persistence_images

# Get test settings
num_categories, splits, epochs, height, width, n_frames, batch_size, steps_per_epoch, validation_steps = get_test_settings()

def main():

    # Defining path to video data
    UCF101_dir = pathlib.Path('./UCF101')
    
    # Create subset directories
    subset_dirs = create_subset_dirs(num_categories = num_categories, UCF101_dir = UCF101_dir, splits = splits)

    # Define output signature
    output_signature = ((tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32), tf.TensorSpec(shape = (None, None, None, 1), dtype = tf.float32)),
                         tf.TensorSpec(shape = (), dtype = tf.int16))
    
    # Generate training, validation and testing datasets
    train_ds = tf.data.Dataset.from_generator(Frame_PI_Generator(subset_dirs['train'], n_frames, training=True), 
                                              output_signature = output_signature)
    val_ds = tf.data.Dataset.from_generator(Frame_PI_Generator(subset_dirs['val'], n_frames), output_signature = output_signature)
    test_ds = tf.data.Dataset.from_generator(Frame_PI_Generator(subset_dirs['test'], n_frames), output_signature = output_signature)

    # Call function to test multi-branch fusion model
    test_multi_branch_fusion_model(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds)

    return

# Function to test multi-branch fusion model
def test_multi_branch_fusion_model(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds):

    # Make version of datasets that repeat for training
    repeat_train_ds = train_ds.repeat().batch(batch_size)
    repeat_val_ds = val_ds.repeat().batch(batch_size)
    
    # Batch data into desired sizes
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    # Define input shapes
    input_shape_x_ds = (None, n_frames, height, width, 3)
    input_shape_pi_ds = (None, n_frames, height, width, 1)

    # Call function to create the multi-branch 3D CNN model
    model = create_multi_branch_3D_CNN(train_ds, input_shape_x_ds, input_shape_pi_ds)

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

# ---------------------------------------  MULTI BRANCH MODEL CREATION CODE ------------------------------------------------
    
# Function that creates a multi-branch 3D CNN model using a training dataset and specified input shapes
def create_multi_branch_3D_CNN(x_ds, input_shape_x_ds, input_shape_pi_ds):

    # Initialize a counter and outputs
    count = 0
    output_1 = None
    output_2 = None

    # Create input layers using input shapes
    x_ds_input = layers.Input(shape=(input_shape_x_ds[1:]))
    pi_ds_input = layers.Input(shape=(input_shape_pi_ds[1:]))

    # Run loop once for each input shape
    for shape in [x_ds_input, pi_ds_input]:
        
        # Assign variable "x" to specific shape
        x = shape

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

        # Assign "x" to specific output variable
        if count == 0:
            output_1 = x
            count += 1
        else:
            output_2 = x

    # Concatenate outputs
    output = concatenate([output_1, output_2])

    # Define model using starting and ending points
    model = keras.Model(inputs=[x_ds_input, pi_ds_input], outputs = output)

    # Initailize weights within neural network using a sample of video frames and persistence images
    (frames, pis), label = next(iter(x_ds))
    model.build([frames, pis])

    # Return model
    return model

# ---------------------------------------  END OF MULTI BRANCH MODEL CREATION CODE ------------------------------------------

# ---------------------------------------------  MULTI BRANCH FUSION CODE ---------------------------------------------------

# Define Frame_PI_Generator class
class Frame_PI_Generator:

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

            # Get label
            label = self.class_ids_for_name[name]

            # Get video frames
            video_frames = frames_from_video_file(path, self.n_frames) 

            # Obtain persistence images based on video frames and store them to a list
            point_clouds = generate_point_clouds(video_frames)
            simplex_trees, persistence_diagrams = generate_pds_sts(point_clouds)
            persistence_images = generate_persistence_images(simplex_trees)

            # Initialize list for persistence image tensors
            pi_tensors = []

            # Loop through persistence images within list
            for pi in persistence_images:
                
                # If current persistence image is "None," set current persistence image tensor to a tensor of desired shape filled with zeros
                if pi is None:
                    cur_pi_tensor = tf.zeros((height,width,1), dtype = tf.float32)

                else:
                    # Convert current persistence image to a tensor
                    cur_pi_tensor = tf.convert_to_tensor(pi, dtype = tf.float32)

                    # Reshape tensor to desired shape
                    cur_pi_tensor = tf.reshape(cur_pi_tensor, [height, width, 1])

                # Append persistence image tensor to list
                pi_tensors.append(cur_pi_tensor)

            # Convert list to an array
            pi_array = np.array(pi_tensors)

            # Yield video frames and persistence images with their respective label
            yield (video_frames, pi_array), label

# -------------------------------------------  END OF MULTI BRANCH FUSION CODE ----------------------------------------------

if __name__ == "__main__":
    main()