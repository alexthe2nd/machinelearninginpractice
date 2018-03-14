# Preprocessing
Ciresan et al. (2012) [1] achieved low error rate using augmentation methods. They randomly translate up to 5% of the images. In addition, scaling (±15%), rotation (±5◦) improved the results more.
Simard et al. (2003) [2] illustrate augmentation using affine (translation, rotation, scaling, horizontal shearing) and elastic deformations. Using a CNN with affine and elastic deformations they report an error rate of respectively 0.6% and 0.4%.
LeCun et al. (1998) [3] also show a decreasing error rate using affine deformations.

Thus, we may conclude that augmentation of training data may have an impact on the error rate.

## Usage
### Installing
Run this before running your program:
```
pip install imgaug
```
### Augmenting a dataset
As we want the augmented dataset to be the same for every training session, a dataset can be generated and saved. We use `imgaug`, a library that can deform images. In the example, every training sample is rotated between -15◦ and 15◦, with a 30% chance of that happening. Furthermore, we can define a `augmentation_size`, which indicates how big the newly generated data set will be in comparison to the initial data set. For more information about defining other deformations, see [imgaug](http://imgaug.readthedocs.io/en/latest/source/augmenters.html).
```
# Load MNIST files
dataset = PreProcesser.load_files()

# Define deformations
augmenter = iaa.Sequential([
	iaa.Sometimes(0.3, [
		iaa.Affine(rotate=(-0.15, 0.15))
	])
])

# Augment dataset and make it twice as big
aug_dataset = PreProcesser.augment_mnist_data(dataset, augmenter=augmenter, augmented_ratio=2)
```

### Using the Zalando dataset
As the MNIST dataset is too easy, we may want to use other datasets. Luckily, Zalando Research has made a [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). This dataset can be loaded using the following commands. Of course, you can apply augmentation to this dataset.

```
# Load Fashion
dataset = PreProcesser.load_files(folder='../Fashion-MNIST/', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
```

### Saving the augmented dataset
Saving the augmented dataset is done easily using [Pickle](https://docs.python.org/3/library/pickle.html). The data is gzipped to reduce on disk space.
```
PreProcesser.save(aug_dataset, '../MNIST_data/saved_dataset.pkl')
```

### Loading the augmented dataset
When you want to load the augmented dataset in your program, you can easily open it using the following command.
```
aug_dataset = PreProcesser.load('../MNIST_data/saved_dataset.pkl')
```

## References
[1] Ciregan, D., Meier, U., & Schmidhuber, J. (2012, June). Multi-column deep neural networks for image classification. In Computer vision and pattern recognition (CVPR), 2012 IEEE conference on (pp. 3642-3649). IEEE.
[2] Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003, August). Best practices for convolutional neural networks applied to visual document analysis. In ICDAR (Vol. 3, pp. 958-962).
[3] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
