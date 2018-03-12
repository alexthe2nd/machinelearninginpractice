## Data transformations for training and testing

When more training data is not available, different transformations to the existing training and test data
are used to augment the sets and hence improve the current state of the art deep convolutional neural network
based image classification pipeline.

Transformations:

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

## Singular Value Bounding 

The paper investigates network properties that can lead to good performance. Research is inspired by the usage of
orthogonal matrices to initialize networks in order to inspect how orthogonal weight matrices perform when network 
training converges. The authors of the paper propose to constrain the solutionsof weight matrices in the orthogonal 
feasible set during the whole process of network training, and achieve this by a simple yet effective method called 
Singular Value Bounding (SVB). In SVB, all singular values of each weight matrix are simply  bounded in a narrow band 
around the value of 1. Based on the same motivation they propose using Bounded Batch Normalization (BBN). Experiments on
benchmark image classification datasets show the efficacy of our proposed SVB and BBN. The results achieved are 3.06% error
rate on CIFAR10 and 16.90% on CIFAR100, using off-the-shelf network architectures (Wide ResNets).  

Paper: https://arxiv.org/pdf/1611.06013.pdf
