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
import tensorflow as tf
import keras

from utils import create_subset_dirs
from utils import FrameGenerator
from utils import create_3D_CNN
from utils import plot_history
from utils import get_actual_predicted_labels
from utils import plot_confusion_matrix
from utils import calculate_precision_recall
from utils import calculate_F1_scores
from utils import print_classification_metrics

# Defining test settings
num_categories = 1
splits = {"train": 70, "val": 10, "test": 20}
height = 112
width = 112
epochs = 1
n_frames = 10
batch_size = 8

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

    # Test baseline model
    test_baseline_model(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds)

    return 

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

# -------------------------------------------- END OF TEST BASED CODE -------------------------------------------------

if __name__ == "__main__":
    main()