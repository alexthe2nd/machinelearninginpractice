# TODO
* BatchGenerator --> Generates train samples
  * Resizes images to a suitable format 
  * Applies augmentation
* Setup your (pretrained) model, extend it with layers
  * Apply sigmoid in the final layer instead of softmax
  * Use binary_crossentropy as loss function
