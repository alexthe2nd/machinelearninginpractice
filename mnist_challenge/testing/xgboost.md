# XGBoost testing

*After repeated environment issues, I drunkenly fixed everything on Saturday night.*

Initial testing results had an accuracy of 93.81%. 

~~I will attempt to implement some improvements on Sunday night and Monday morning, and find out how it performs.~~ Responsibilities popped up at home, and I couldn't work on it anymore. Sorry about that. 

*- Alex S.*

Basic script, with some unused imports: 
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
   
  # Fit model on training data
  model = XGBClassifier()
  model.fit(train_data, train_labels)
  
  # Make predictions for evaluation data
  eval_pred = model.predict(eval_data)
  predictions = [round(value) for value in eval_pred]
  
  # Evaluate predictions
  accuracy = accuracy_score(eval_labels, predictions)
  print("Accuracy: %.2f%%" % (accuracy * 100.0))

if __name__ == "__main__":
  tf.app.run()
```
