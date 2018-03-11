from mnist.helpers.preprocess import PreProcesser
from imgaug import augmenters as iaa

if __name__ == '__main__':
	dataset = PreProcesser.load_files(folder='./MNIST_data/')
	augmenter = iaa.OneOf([
		iaa.Sometimes(0.3,
					  [iaa.Affine(rotate=(-0.15, 0.15))]
					  )
	])
	aug_dataset = PreProcesser.augment_mnist_data(dataset, augmenter=augmenter,
												  augmented_ratio=1)
	print("created dataset")
	print(type(aug_dataset))
	PreProcesser.save(aug_dataset,
					  'MNIST_data/dataset_ratio_1_no_augmentation.pkl')