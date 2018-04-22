import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa

from mnist.preprocess import PreProcesser


def plot(image):
	# The rest of columns are pixels
	pixels = image * 255

	# Make those columns into a array of 8-bits pixels
	# This array will be of 1D with length 784
	# The pixel intensity values are integers from 0 to 255
	pixels = np.array(pixels, dtype='uint8')

	# Reshape the array into 28 x 28 array (2-dimensional array)
	pixels = pixels.reshape((28, 28))

	# Plot
	plt.title('PLOT')
	plt.imshow(pixels, cmap='gray')
	plt.show()

if __name__ == '__main__':
	dataset = PreProcesser.load_files(folder='./MNIST_data/')

	# Original MNIST dataset
	PreProcesser.save(dataset, 'MNIST_data/DC1/dc_default.pkl')

	#plot(dataset.train.images[0][:])

	double_augmenter = iaa.OneOf([

	])
	double_dataset = PreProcesser.augment_mnist_data(dataset, augmenter=double_augmenter, augmented_ratio=4)
	PreProcesser.save(double_dataset, 'MNIST_data/DC1/dc_double.pkl')

	flip_augmenter = iaa.OneOf([
		iaa.Fliplr(0.5)
	])
	flipped_dataset = PreProcesser.augment_mnist_data(double_dataset, augmenter=flip_augmenter, augmented_ratio=1)
	PreProcesser.save(flipped_dataset, 'MNIST_data/DC1/dc_flip.pkl')

	# Random crop
	# Picks a 24x24 patch from the image and scales it back up to 28x28
	offset = 3 # nr of pxs the crop will be offset from the side (pick 3 or 4, picking 2 results in center crops only)
	# What we really want here is take all 5 patches so the dataset size increases fivefold.
	# However, that requires creating five augmenters, because of how the framework works right now.
	crop_augmenter = iaa.OneOf([
		# topleft
		iaa.Sequential([
			iaa.Crop(px=(4 - offset, offset, offset, 4 - offset)),
			iaa.Affine(scale=1.166666667)
		]),
		# topright
		iaa.Sequential([
			iaa.Crop(px=(4 - offset, 4 - offset, offset, offset)),
			iaa.Affine(scale=1.166666667)
		]),
		# botleft
		iaa.Sequential([
			iaa.Crop(px=(offset, offset, 4 - offset, 4 - offset)),
			iaa.Affine(scale=1.166666667)
		]),
		# botright
		iaa.Sequential([
			iaa.Crop(px=(offset, 4 - offset, 4 - offset, offset)),
			iaa.Affine(scale=1.166666667)
		]),
		# center
		iaa.Sequential([
			iaa.Crop(px=(2, 2, 2, 2)),
			iaa.Affine(scale=1.166666667)
		])
	])
	cropped_dataset = PreProcesser.augment_mnist_data(flipped_dataset, augmenter=crop_augmenter, augmented_ratio=1)
	PreProcesser.save(cropped_dataset, 'MNIST_data/DC1/dc_crop.pkl')

	#plot(cropped_dataset.train.images[0][:])


	# Rotate
	rotate_augmenter = iaa.OneOf([
		iaa.Affine(rotate=(-15, 15))
	])

	rotated_dataset = PreProcesser.augment_mnist_data(cropped_dataset, augmenter=rotate_augmenter, augmented_ratio=1)
	PreProcesser.save(rotated_dataset, 'MNIST_data/DC/dc_rotate.pkl')

	#plot(rotated_dataset.train.images[0][:])

	# PiecewiseAffine
	piecewise_affine_augmenter = iaa.OneOf([
		iaa.PiecewiseAffine(scale=(0.0, 0.1))
	])
	piecewise_affine_dataset = PreProcesser.augment_mnist_data(rotated_dataset, augmenter=piecewise_affine_augmenter, augmented_ratio=1)
	PreProcesser.save(piecewise_affine_dataset, 'MNIST_data/DC/dc_piecewise_affine.pkl')

	#plot(piecewise_affine_dataset.train.images[0][:])