## Data transformations for training and testing

When more training data is not available, difference transformations to the existing training and test data
are used to augment the sets and hence improve the current state of the art deep convolutional neural network
based image classification pipeline.

- Extending image crops into extra pixels
- Color manipulations 

- Predictions at multiple scales 
- Predictions with multiple views 
- Reducing the number of predictions

In practice, previously trained models can be used to initialize higher resolution models and cut
training time down substantially to 30 epochs from 90 epochs. Higher resolution models are
complementary to base models and a single high resolution model is as valuable as four
additional base models

Paper: https://arxiv.org/ftp/arxiv/papers/1312/1312.5402.pdf
