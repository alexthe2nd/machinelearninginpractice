# Machine Learning in Practice

Team coach: Zaheer Babar, [z.babar@cs.ru.nl](z.babar@cs.ru.nl).


## Digit Recognition competition on [kaggle.com](https://kaggle.com/)
5th of March individual goal:
* Get TensowFlow running with a digit recognition algorithm
* Learn how to export your model so you don't have to retrain the CNN again


### Before next week

### Planning
***Improving the training set*** X people are going to improve the training dataset by (de)centering it and mutating it. Other methods have used this approach to get more training samples. Rick and Tristan. Afterwards, improve Zalando.

***Getting a list of potential improvements*** Literature research on potential improvements. Title + short description + link to source material. Ankur and Brigel are going to research improvements on NN, Luca, Alex are going to research improvements on other techniques. Deadline next week.

### Next week
***Split up*** we'll assign potential improvements to groups of two and get performance.

***Discuss results*** How is the model built (#num layers, size, etc.), dataset used. Discuss further.


After everybody has been set up with the system, we're going to tweak the neural network. 
We can possibly split up the group in two subgroups. One subgroup will continue on tinkering the neural network, the other group will use other methods. In this way, we can use ensemble learning.

[Official digit recognition TensorFlow tutorial](https://www.tensorflow.org/tutorials/layers)

[Best MNIST results so far](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)

[How to implement DropConnect in TensorFlow](https://nickcdryan.wordpress.com/2017/06/13/dropconnect-implementation-in-python-and-tensorflow/)

[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
Colab now supports running TensorFlow computations on a GPU. Simply select "GPU" in the Accelerator drop-down in Notebook Settings (either through the Edit menu or the command palette at cmd/ctrl-shift-P).

[Use Tensorboard in combination with Google Colaboratory](https://stackoverflow.com/questions/47818822/can-i-use-tensorboard-with-google-colab)
