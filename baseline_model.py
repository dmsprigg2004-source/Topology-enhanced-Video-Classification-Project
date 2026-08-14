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
from utils import get_test_settings
from utils import early_stoppage
from utils import create_metrics_test_settings_spreadsheet
from utils import save_results

# Get test settings
num_categories, splits, epochs, height, width, n_frames, batch_size, dataset_split, steps_per_epoch, validation_steps = get_test_settings()

# Choose whether to save results to folder with specific name
save_output = True
results_file_name = "test"

def main():

    # Defining path to video data
    UCF101_dir = pathlib.Path('./UCF101')
    
    # Create subset directories
    subset_dirs = create_subset_dirs(num_categories = num_categories, UCF101_dir = UCF101_dir)

    # Define output signature
    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))
    
    # Generate training, validation and testing datasets
    train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['train'], n_frames, training=True), 
                                              output_signature = output_signature)
    val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['val'], n_frames), output_signature = output_signature)
    test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['test'], n_frames), output_signature = output_signature)

    # Initialize lists
    train_ds_list = []
    val_ds_list = []
    test_ds_list = []

    # Loop through training, validation and testing datasets and add values to lists
    for video_frames, label in train_ds:
        train_ds_list.append((video_frames, label))

    for video_frames, label in val_ds:
        val_ds_list.append((video_frames, label))

    for video_frames, label in test_ds:
        test_ds_list.append((video_frames, label))

    # Create new datasets
    new_train_ds = tf.data.Dataset.from_generator(FrameGenerator2(train_ds_list), 
                                              output_signature = output_signature)
    new_val_ds = tf.data.Dataset.from_generator(FrameGenerator2(val_ds_list), 
                                              output_signature = output_signature)
    new_test_ds = tf.data.Dataset.from_generator(FrameGenerator2(test_ds_list), 
                                              output_signature = output_signature)

    print("Datasets created")

    # Obtain early stoppage callback
    callback = early_stoppage()

    # Test baseline model
    test_baseline_model(steps_per_epoch, validation_steps, subset_dirs, new_train_ds, new_val_ds, new_test_ds, callback)

    # Save results to folder if wanted
    if save_output:
        save_results(results_file_name)

    return 

# Define FrameGenerator2 class
class FrameGenerator2:

    # __init__ function to initialize instance attributes
    def __init__(self, frame_list):
        self.frame_list = frame_list

    # __call__ function that yields video frames with their respective label
    def __call__(self):

        for (frames, label) in self.frame_list:
            
            # Yield video frames and its respective label
            yield frames, label

# Function to test baseline video classificaion model
def test_baseline_model(steps_per_epoch, validation_steps, subset_dirs, train_ds, val_ds, test_ds, callback):

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
                        validation_steps = validation_steps, callbacks=[callback])
    
    # Call function to plot history of model training performance
    plot_history(history)

    # Evaluate model to get accuracy and loss values
    model_accuracy_and_loss = model.evaluate(test_ds, return_dict=True)

    # Obtain model accuracy
    model_accuracy = model_accuracy_and_loss["accuracy"]

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

    # Call function to create spreadsheet of classification metrics and test settings
    create_metrics_test_settings_spreadsheet(model_accuracy, precision, recall, F1_scores)

    return

if __name__ == "__main__":
    main()