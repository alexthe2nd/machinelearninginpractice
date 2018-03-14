"""
Pre-processing class.

"""
import dill
import gzip
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from imgaug import augmenters as iaa

from tqdm import tqdm

class PreProcesser():
	def __init__(self):
		pass

	@staticmethod
	def save(dataset, filename):
		"""
		Writes a data set to a file.
		:param dataset:
		:return:
		"""
		assert isinstance(filename, str)
		print("Saving augmented images")
		file = gzip.open(filename, 'wb')
		dill.dump(dataset, file)
		file.close()
		print("Saving completed")

	@staticmethod
	def load(filename):
		"""
		Loads a data set.
		:return: data set
		"""
		print("Loading augmented images")
		assert isinstance(filename, str)
		file = gzip.open(filename, 'rb')
		dataset = dill.load(file)
		file.close()
		print("Loading completed")
		return dataset

	@staticmethod
	def augment_mnist_data(dataset, augmenter, augmented_ratio=1):
		"""
		Augments a data set and returns it.
		:param augmenter: The augmenter
		:param augmented_ratio: Returns how many times the data set needs to be replicated.
		:param data set: The data set that needs to be augmented.
		:return:
		"""
		train_images = []
		train_labels = []

		training_length = len(dataset.train.images)
		# Loop all training images
		for i in tqdm(range(augmented_ratio * training_length), desc="Augmenting images", unit="image"):
			# Augment images
			train_images.append(
				np.reshape(augmenter.augment_image(dataset.train.images[i % training_length].reshape(28, 28, 1)), 784))

			# Append corresponding label
			train_labels.append(dataset.train.labels[i % training_length])

		train = Dataset(train_images, train_labels)
		validation = dataset.validation
		test = dataset.test
		return base.Datasets(train=train, validation=validation, test=test)

	@staticmethod
	def load_files(folder='../MNIST_data/', source_url=None):
		"""
		Returns MNIST data set form specified folder. Downloads the files if the data is not present.
		:param source_url: The URL to download the data set from if no files are present yet.
		:param folder: The target and source folder for the data set.
		:return:
		"""

		if source_url:
			return read_data_sets(folder, source_url=source_url, one_hot=False)
		else:
			return read_data_sets(folder, one_hot=False)

class Dataset:
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels

	def next_batch(num, data, labels):
		idx = np.arange(0, len(data))
		np.random.shuffle(idx)
		idx = idx[:num]
		data_shuffle = [data[i] for i in idx]
		labels_shuffle = [labels[i] for i in idx]

		return np.asarray(data_shuffle), np.asarray(labels_shuffle)


if __name__ == '__main__':
	dataset = PreProcesser.load_files()
	augmenter = iaa.OneOf([
		iaa.Sometimes(0.3, [
			iaa.Affine(rotate=(-0.15, 0.15))
		])
	])
	aug_dataset = PreProcesser.augment_mnist_data(dataset, augmenter=augmenter, augmented_ratio=2)
	PreProcesser.save(aug_dataset, '../MNIST_data/saved_dataset_1.pkl')
