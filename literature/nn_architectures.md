## BinaryConnect
**Error:** 1.01%  
**Paper:** http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-binary-weights-during-propagations.pdf

Trains the network with only two possible weight values {-1, 1}. This works as a kind of regularizer for
 the network. After each forward+backward propagation the values of the weights is replaced with one of 
the binary values. The paper talks about 2 possible methods to do that: deterministically by having a set 
threshold or stochastically by computing the probability as sigmoid of the weight.

## Network in Network
**Error:** 0.47%  
**Paper:** https://arxiv.org/pdf/1312.4400.pdf

This paper suggests replacing the convolution layers with `mlpconv` layer. The `mlpconv` layer works similar to a
convolution layer but instead of having filters for each subsection of the image, `mlpconv` train a MLP for each of these subsections in the image and then applies Average Pooling on the reults of these MLPs, similar to CNNs.
