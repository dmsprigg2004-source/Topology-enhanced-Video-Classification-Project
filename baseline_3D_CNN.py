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
import itertools
import collections

import cv2
import einops
import numpy as np
import remotezip as rz
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
import random

num_categories = 3
splits = {"train": 35, "val": 5, "test": 10}
epochs = 1
height = 56
width = 56 
n_frames = 8
batch_size = 8

def main():

    UCF101_dir = pathlib.Path('/Users/darcysprigg/Coding/Co-op summer 2026/UCF101')

    subset_dirs = create_subset_dirs(num_categories = num_categories, UCF101_dir = UCF101_dir, splits = splits)

    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))
    
    train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['train'], n_frames, training=True), 
                                              output_signature = output_signature)

    val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['val'], n_frames), output_signature = output_signature)

    test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_dirs['test'], n_frames), output_signature = output_signature)

    # train_frames_dict, train_pc_dict, train_sts_dict, train_pds_dict, train_pis_dict = tf_extraction(train_ds, "Training")
    # val_frames_dict, val_pc_dict, val_sts_dict, val_pds_dict, val_pis_dict = tf_extraction(val_ds, "Validation")
    # test_frames_dict, test_pc_dict, test_sts_dict, test_pds_dict, test_pis_dict = tf_extraction(test_ds, "Test")

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    input_shape = (None, n_frames, height, width, 3)
    input = layers.Input(shape=(input_shape[1:]))
    x = input

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)

    model = keras.Model(input, x)

    frames, label = next(iter(train_ds))
    model.build(frames)

    model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = keras.optimizers.Adam(learning_rate = 0.0001), 
                  metrics = ['accuracy'])
    
    history = model.fit(x = train_ds, epochs = epochs, validation_data = val_ds)
    
    plot_history(history)

    model.evaluate(test_ds, return_dict=True)

    fg = FrameGenerator(subset_dirs['train'], n_frames, training=True)
    labels = list(fg.class_ids_for_name.keys())     

    actual, predicted = get_actual_predicted_labels(train_ds, model)
    plot_confusion_matrix(actual, predicted, labels, 'training')

    actual, predicted = get_actual_predicted_labels(test_ds, model)
    plot_confusion_matrix(actual, predicted, labels, 'test')

    precision, recall = calculate_classification_metrics(actual, predicted, labels)
    
    print(precision)

    print(recall)

    return 

def tf_extraction(x_ds, name):
    
    frames_dict = {}
    count = 0

    for frames, label in x_ds:
        frames_dict[f"{label}.{count}"] = frames
        count += 1
    
    point_cloud_dict = {}
    simplex_trees_dict = {}
    persistence_diagrams_dict = {}
    persistence_images_dict = {}

    for label, frames in tqdm(frames_dict.items(), desc= f"{name} - Genrating point clouds"):
        point_cloud_dict[label] = generate_point_clouds(frames)

    for label, point_clouds in tqdm(point_cloud_dict.items(), desc= f"{name} - Generating simplex trees and pesistence images"):
        simplex_trees, persistence_diagrams = generate_pds_sts(point_clouds)
        simplex_trees_dict[label] = simplex_trees
        persistence_diagrams_dict[label] = persistence_diagrams

    for label, simplex_trees in tqdm(simplex_trees_dict.items(), desc= f"{name} - Generating persistence images"):
        persistence_images_dict[label] = generate_persistence_images(simplex_trees)

    return frames_dict, point_cloud_dict, simplex_trees_dict, persistence_diagrams_dict, persistence_images_dict
    

def split_class_lists(files_for_class, count):
    split_files = []
    remainder = {}

    for cls in files_for_class:
        split_files.extend(files_for_class[cls][:count])
        remainder[cls] = files_for_class[cls][count:]

    return split_files, remainder

def create_subset_dir(category_dict, categories_list, split_files, split_name):

    new_dir_path = Path(f'/Users/darcysprigg/Coding/Co-op summer 2026/{split_name}')

    if new_dir_path.is_dir():
        shutil.rmtree(new_dir_path)
    
    new_dir_path.mkdir()
   
    for category in categories_list:

        new_category_dir_path = Path(f'/Users/darcysprigg/Coding/Co-op summer 2026/{split_name}/{category}')
        new_category_dir_path.mkdir()

        for file in category_dict[category]:
            for split_file in split_files:
                if file == split_file:
                    needed_file = Path(f'/Users/darcysprigg/Coding/Co-op summer 2026/UCF101/{category}/{file}')
                    shutil.copy(needed_file, new_category_dir_path)

    return new_dir_path

def create_subset_dirs(num_categories, UCF101_dir, splits):

    category_dict = {}
    category_count = 0

    for category in os.listdir(UCF101_dir):

        category_count += 1

        category_path = os.path.join(UCF101_dir, category)

        if os.path.isdir(category_path):
            category_dict[category] = []

            for video in os.listdir(category_path):
                category_dict[category].append(video)

        if category_count == num_categories:
           break

    categories_list = list(category_dict.keys())[:num_categories]

    for category in categories_list:
        new_files_for_class = category_dict[category]
        random.shuffle(new_files_for_class)
        category_dict[category] = new_files_for_class

    subset_dirs = {}
    category_dict_copy = copy.deepcopy(category_dict)

    for split_name, split_count in splits.items():

        split_files, category_dict_copy = split_class_lists(category_dict_copy, split_count)

        split_dir = create_subset_dir(category_dict, categories_list, split_files, split_name)

        print(f"{split_name} directory created")

        subset_dirs[split_name] = split_dir

    return subset_dirs

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)

    return frame

def frames_from_video_file(video_path, n_frames, output_size = (height,width), frame_step = 15):
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))

    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result

class FrameGenerator:
    def __init__(self, path, n_frames, training = False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames) 
            label = self.class_ids_for_name[name]
            yield video_frames, label

class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq = keras.Sequential([  
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            layers.Conv3D(filters=filters, 
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
            ])
  
    def call(self, x):
        return self.seq(x)

class ResidualMain(keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
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
    
    def call(self, x):
        return self.seq(x)

class Project(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
                  layers.Dense(units),
                  layers.LayerNormalization()
                  ])

    def call(self, x):
        return self.seq(x)

def add_residual_block(input, filters, kernel_size):
    out = ResidualMain(filters, kernel_size)(input)
  
    res = input
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
                images, '(b t) h w c -> b t h w c',
                t = old_shape['t'])
        return videos

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label = 'train')
    ax1.plot(history.history['val_loss'], label = 'test')
    ax1.set_ylabel('Loss')
    
    max_loss = max(history.history['loss'] + history.history['val_loss'])

    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation']) 

    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'],  label = 'train')
    ax2.plot(history.history['val_accuracy'], label = 'test')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.savefig("./History Plots/history_plot.png")

    print("History plot saved")

def get_actual_predicted_labels(dataset, model): 
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels, ds_type):
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

def calculate_classification_metrics(y_actual, y_pred, labels):
  cm = tf.math.confusion_matrix(y_actual, y_pred)
  tp = np.diag(cm)
  precision = dict()
  recall = dict()
  for i in range(len(labels)):
    col = cm[:, i]
    fp = np.sum(col) - tp[i]
    
    row = cm[i, :]
    fn = np.sum(row) - tp[i]

    if tp[i] + fp != 0:
    
        precision[labels[i]] = tp[i] / (tp[i] + fp)

    else:
        precision[labels[i]] = None

    if tp[i] + fn != 0:

        recall[labels[i]] = tp[i] / (tp[i] + fn)
    
    else:
        recall[labels[i]] = None
  
  return precision, recall


# Function that generates point clouds from the data
def generate_point_clouds(video_frames):

    # Create object for ImageToPointCloud class
    itpc = ImageToPointCloud()

    # Initalize list of binary frames
    binary_frames = []

    # Create for loop to access individual frames
    for frame in video_frames:
        
        np_frame = frame.numpy()
            
        # Convert frame to greyscale form
        grey_image = cv2.cvtColor(np_frame, cv2.COLOR_RGB2GRAY)

        grey_image = (grey_image * 255).astype(np.uint8)

        # Convert greyscale image to binary image
        ret, binary_image = cv2.threshold(grey_image, 127, 255, cv2.THRESH_BINARY)

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
        
        if point_cloud.shape[0] == 0:

            persistence_diagrams.append(None)
            simplex_trees.append(None)
            
            continue
        
        num_points = point_cloud.shape[0]
        
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

        if tree is None:
            persistence_images.append(None)
            continue

        # Create persistence image
        persitence_image = PersistenceImage(bandwidth=0.15, weight=lambda x: x[1]**2,
                                        im_range=[0,1.5,0,1.5], resolution=[100,100])
        persitence_image = persitence_image.fit_transform([tree.persistence_intervals_in_dimension(1)])

        # Append image to list
        persistence_images.append(persitence_image)

    # Return persistence images
    return persistence_images

if __name__ == "__main__":
    main()