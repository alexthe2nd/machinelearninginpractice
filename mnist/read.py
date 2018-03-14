import matplotlib.pyplot as plt
import numpy as np

from mnist.preprocess import PreProcesser

data = PreProcesser.load('MNIST_data/DC1/dc_piecewise_affine.pkl')
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

plot(data.train.images[0][:])