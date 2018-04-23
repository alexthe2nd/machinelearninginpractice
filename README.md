# iMaterialist Challenge (Fashion) at FGVC5
_URL: [https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/)_

## Planning

### April 23
_Still waiting for access to the Google Cloud Platform._

### Meeting April 23th
Laurens introduces Zaheer into the challenge and the dataset. 
Zaheer wants us to find the description of the labels. 

Laurens is thinking about making a batch-generator that augments images that contain few occuring classes.

Convolutional network with whitespace or resized images. 

Zaheer advises to first create a simple network that tries to tackle the problem. Then, we'll decide what to happen. 

A few people should start working on a simple network and other people should work on resizing the images.

Region of interest (

### Tasks April 30th
#### Baseline group
(Ankur, Laurens, Luca, Tristan)
* Fully convolutional network and see how to handle the output
* Pre-trained fully convolutional neural network with a threshold.
* Possibly different thresholds for different labels
* Batch generator (Laurens)

#### Resizing group
(Alex, Brigel, Rick)
Stretching or cropping. ImageNet uses 224x224 dimensions, so anything bigger would . Bigger is fine as well. 
