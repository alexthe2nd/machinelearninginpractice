"""
While trying to run XGBoost I have encountered some issues. I couldn't import XGBoost.
Some solutions are available on the internet, you should solve it yourself.

Eventually, I ran the following command
>> exec(open("A4/E25/xgboost.py").read())
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import tensorflow as tf

def main(unused_argv):
	# Load training data
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

	test_data = mnist.test.images
	test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	# Fit model on training data
	model = XGBClassifier(eta=0.3, objective='multi:softprob', num_class=10, max_depth=10)
	model.fit(train_data, train_labels)

	# Make predictions for evaluation data
	test_prediction = model.predict(test_data)
	test_prediction = [round(value) for value in test_prediction]

	# Evaluate predictions
	accuracy = accuracy_score(test_labels, test_prediction)
	print("Accuracy: {0:.2f}".format(accuracy * 100))

if __name__ == "__main__":
	tf.app.run()
