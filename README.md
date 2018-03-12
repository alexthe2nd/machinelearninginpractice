# Machine Learning in Practice

Team coach: Zaheer Babar, [z.babar@cs.ru.nl](z.babar@cs.ru.nl).


## Digit Recognition competition on [kaggle.com](https://kaggle.com/)

## Planning
### Deadline March 5th
5th of March individual goal:
* Get TensowFlow running with a digit recognition algorithm
* Learn how to export your model so you don't have to retrain the CNN again

### Deadline March 12th
***Improving the training set*** X people are going to improve the training dataset by (de)centering it and mutating it. Other methods have used this approach to get more training samples. Rick and Tristan. Afterwards, improve Zalando.

***Getting a list of potential improvements*** Literature research on potential improvements. Title + short description + link to source material. Ankur and Brigel are going to research improvements on NN, Luca, Alex are going to research improvements on other techniques. Deadline next week.

### Deadline March 19th
We've split up the groups in a few groups that will work on various topics. For these topics, some interesting papers have been added in the folder `/literature/`. Results will be compared to the baseline CNN as described in the Tensorflow tutorial, implemented in [mnist/initial.py](mnist/initial.py).

| Who               	| What                                                                       	| Dataset             	|
|-------------------	|----------------------------------------------------------------------------	|---------------------	|
| Laurens & Tristan 	| Various methods of Augmentation on Baseline and DropConnect + Presentation 	| MNIST and Zalando   	|
| Rick              	| On The Fly Augmentation                                                    	| MNIST               	|
| Alex & Luca       	| Random Forests + XGBoost                                                   	| Non-Augmented MNIST 	|
| Ankur & Brigel    	| BinaryConnect + Network in Network                                         	| Non-Augmented MNIST 	|

You'll be documenting your process and describe your findings. This will consists of one paragraph, which answers the following questions:
* What has been done in other research and what contributed to their success?
* What have you done to improve on results? What is the stucture of your neural network and what hyperparameters did you use?
* What results (% error rate) did you get?

## Description
### Running
The system can be easily run by copying the code into [Google Colab](colab.research.google.com), which runs it extremely fast.

### Initial version
The initial version was run on 5000 iterations, with minibatches of 128. This resulted in a test accuracy of 0.9895 (1.05% error rate).

The network was constructed as follows:
reshape -> conv1_5x5 -> max_pool_2x2 -> conv2_5x5 -> max_pool_2x2 -> fully_connected_layer_1024 -> dropout_0.5 -> fully_connected_layer_10

### DropConnect version
The DropConnect version (see [Regularization of Neural Networks using DropConnect](https://cs.nyu.edu/~wanli/dropc/)) replaces the Dropout layer of the initial version with DropConnect 0.5. After 5000 iteratoins, an test accuracy of 0.99 (1.0% error rate) was achieved.

### Augmentation
A detailed description for augmentation is specified in [mnist/helpers](mnist/helpers).

## Links
Some usefull links are included below.
[Official digit recognition TensorFlow tutorial](https://www.tensorflow.org/tutorials/layers)

[Best MNIST results so far](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)

[How to implement DropConnect in TensorFlow](https://nickcdryan.wordpress.com/2017/06/13/dropconnect-implementation-in-python-and-tensorflow/)

[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
Colab now supports running TensorFlow computations on a GPU. Simply select "GPU" in the Accelerator drop-down in Notebook Settings (either through the Edit menu or the command palette at cmd/ctrl-shift-P).

[Use Tensorboard in combination with Google Colaboratory](https://stackoverflow.com/questions/47818822/can-i-use-tensorboard-with-google-colab)

[Augmentation](http://imgaug.readthedocs.io/en/latest/source/augmenters.html)
