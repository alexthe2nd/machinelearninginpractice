'''
Augmentation
Provides a function that uses image augmentation techniques.


'''

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Augment dataset.

'''


def get_data(dataset, dataset_labels, augmentation_factor=1, use_random_rotation=True, use_random_shear=True,
			 use_random_shift=True,
			 use_random_zoom=True):
	augmented_image = []
	augmented_image_labels = []

	for num in range(0, dataset.shape[0]):

		for i in range(0, augmentation_factor):
			# original image:
			augmented_image.append(dataset[num])
			augmented_image_labels.append(dataset_labels[num])

			if use_random_rotation:
				augmented_image.append(
					tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1,
																		 channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shear:
				augmented_image.append(
					tf.contrib.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1,
																	  channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shift:
				augmented_image.append(
					tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1,
																	  channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_zoom:
				augmented_image.append(
					tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], 0.9, row_axis=0, col_axis=1,
																	 channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

	return np.array(augmented_image), np.array(augmented_image_labels)
