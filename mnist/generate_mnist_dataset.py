from mnist.helpers.preprocess import PreProcesser
from imgaug import augmenters as iaa

if __name__ == '__main__':
	dataset = PreProcesser.load_files(folder='./MNIST_data/')
	augmenter = iaa.OneOf([

	])
	aug_dataset = PreProcesser.augment_mnist_data(dataset, augmenter=augmenter, augmented_ratio=1)
	PreProcesser.save(aug_dataset, 'MNIST_data/dataset_ratio_1_no_augmentation.pkl')