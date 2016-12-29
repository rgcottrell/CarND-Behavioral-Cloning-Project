#!/usr/bin/env python

# Import libraries necessary for this project.
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, ELU, Flatten
from keras.optimizers import Adam
from PIL import Image
from sklearn.model_selection import train_test_split

# Location of the simulator data.
DATA_FILE = 'driving_log.csv'

# Load the training data from the simulator.
cols = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']
data = pd.read_csv(DATA_FILE, names=cols, header=1)

# Separate the image paths and steering angles.
images = data[['Center Image', 'Left Image', 'Right Image']]
angles = data['Steering Angle']

# Split the data into training and validation sets.
images_train, images_validation, angles_train, angles_validation = train_test_split(images, angles, test_size=0.15, random_state=42)

# Define the model
model = Sequential()
model.add(Conv2D(3, 1, 1, input_shape=(90, 320, 3)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Conv2D(3, 5, 5, subsample=(2, 2)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Conv2D(48, 3, 3))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Conv2D(64, 3, 3))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Conv2D(128, 3, 3))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(1))

# Select the optimizer and compile the model.
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)

# Helper to load an normalize images.
def load_image(path, flip=False):
    # Read the image from disk, and flip it if requested.
    image = Image.open(path.strip())
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Normalize the image pixels to range -1 to 1.
    image = np.array(image, np.float32)
    image /= 127.5
    image -= 1.
    
    # Slice off the top and bottom pixels to remove the sky
    # and the hood of the car.
    image = image[40:130, :]
    
    # Return the normalized image.
    return image

# Data generator.
def generate_batches(images, angles, batch_size=64, augment=True):
    # Create an array of sample indexes.
    indexes = np.arange(len(images))
    batch_images = []
    batch_angles = []
    sample_index = 0
    while True:
        # Reshuffle the indexes after each pass through the samples to minimize
        # overfitting on the data.
        np.random.shuffle(indexes)
        for i in indexes:
            # Increment the number of samples. 
            sample_index += 1
            
            # Load the center image and weight.
            center_image = load_image(images.iloc[i]['Center Image'])
            center_angle = float(angles.iloc[i])
            batch_images.append(center_image)
            batch_angles.append(center_angle)
            
            # Add augmentation if requested
            if augment:
                # Load the flipped image and invert angle
                flipped_image = load_image(images.iloc[i]['Center Image'], True)
                flipped_angle = -1. * center_angle
                batch_images.append(flipped_image)
                batch_angles.append(flipped_angle)

                # Load the left image and adjust angle
                left_image = load_image(images.iloc[i]['Left Image'])
                left_angle = min(1.0, center_angle + 0.25)
                batch_images.append(left_image)
                batch_angles.append(left_angle)
                # Load the right image and adjust angle
                right_image = load_image(images.iloc[i]['Right Image'])
                right_angle = max(-1.0, center_angle - 0.25)
                batch_images.append(right_image)
                batch_angles.append(right_angle)
            
            # If we have processed batch_size samples or this is the last batch
            # of the epoch, then submit the batch. Note that due to augmentation
            # there may be more than batch_size elements in the batch.
            if (sample_index % batch_size) == 0 or (sample_index % len(images)) == 0:
                yield np.array(batch_images), np.array(batch_angles)
                batch_images = []
                batch_angles = []

# Instantiate data generators for training and validation.
nb_epoch = 25
samples_per_epoch = 4 * len(images_train)
generator_train = generate_batches(images_train, angles_train)
nb_val_samples = len(images_validation)
generator_validation = generate_batches(images_validation, angles_validation, augment=False)

# Run the model.
history = model.fit_generator(
    generator_train, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
    validation_data=generator_validation, nb_val_samples=nb_val_samples)

# Save the generated model and weights.
