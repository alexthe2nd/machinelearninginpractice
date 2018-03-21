import pandas as pd

import imgaug.augmenters as iaa
import numpy as np
from tqdm import tqdm


def get_crops(x_train, y_train, offset=4):
	"""Creates 5 crops for each image, along with corresponding ground truths

    Parameters
    ---------
    x_train: List of training set images

    y_train: List of ground truths that correspond to x_tain

    offset (Default = 4): Amount of pixels to cut off (other side is cut off by 4-offset),
                          setting offset to 2 will result in 5 equivalent images (centered).

    """
	topleft = iaa.Sequential([
		iaa.Crop(px=(4 - offset, offset, offset, 4 - offset)),
		iaa.Affine(scale=1.166666667)
	])
	topright = iaa.Sequential([
		iaa.Crop(px=(4 - offset, 4 - offset, offset, offset)),
		iaa.Affine(scale=1.166666667)
	])
	botleft = iaa.Sequential([
		iaa.Crop(px=(offset, offset, 4 - offset, 4 - offset)),
		iaa.Affine(scale=1.166666667)
	])
	botright = iaa.Sequential([
		iaa.Crop(px=(offset, 4 - offset, 4 - offset, offset)),
		iaa.Affine(scale=1.166666667)
	])
	center = iaa.Sequential([
		iaa.Crop(px=(2, 2, 2, 2)),
		iaa.Affine(scale=1.166666667)
	])
	augs = [topleft, topright, botleft, botright, center]

	aug_imgs = []
	for aug in tqdm(augs):
		aug_imgs.append(aug.augment_images(x_train * 255))

	aug_x_train = [item for sublist in aug_imgs for item in sublist]
	aug_y_train = y_train * 5

	return aug_x_train, aug_y_train


def augment(train_images, train_labels, augmenter):
	result_images = train_images
	train_labels = train_labels * 2
	for i in tqdm(range(len(train_images))):
		result_images.append(augmenter.augment_image(train_images[i]))

	return result_images, train_labels


def save_csv(df, train_images, train_labels, filename='train_augmented.csv'):
	train_images = np.reshape(train_images, (len(train_images), 784))
	set = np.concatenate((train_labels, train_images), axis=1)
	newdf = pd.DataFrame(set, columns=df.columns.get_values())

	newdf.to_csv('~/Caddy/MNIST/{}'.format(filename), index=False)


def main():
	train = pd.read_csv('train.csv')
	train_images = train.values[:, 1:] / 255
	train_labels = train.values[:, :1].tolist()

	train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))

	train_images, train_labels = get_crops(train_images, train_labels)

	save_csv(train, train_images, train_labels, 'train_org_crops.csv')

	rotate_augmenter = iaa.OneOf([
		iaa.Affine(rotate=(-15, 15))
	])
	train_images, train_labels = augment(train_images, train_labels, rotate_augmenter)

	save_csv(train, train_images, train_labels, 'train_org_crops_rot.csv')

	piecewise_affine_augmenter = iaa.OneOf([
		iaa.PiecewiseAffine(scale=(0.0, 0.1))
	])

	train_images, train_labels = augment(train_images, train_labels, piecewise_affine_augmenter)

	save_csv(train, train_images, train_labels, 'train_org_crops_rot_pw.csv')

	sharpen_augmenter = iaa.OneOf([
		iaa.Sharpen(alpha=1, lightness=500)
	])

	train_images, train_labels = augment(train_images, train_labels, sharpen_augmenter)

	save_csv(train, train_images, train_labels, 'train_org_crops_rot_pw_sharpen.csv')

if __name__ == '__main__':
	main()
