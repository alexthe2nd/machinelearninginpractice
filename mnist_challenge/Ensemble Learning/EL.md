## Layers after 
For Xception, we choose to apply a fully connected layer (size 512), as this shows improved results (https://arxiv.org/pdf/1610.02357.pdf)

For VGG, we also apply a fully connected layer (size 512). 

## Unfreezing layers (FreezeOut)
Freezing the first layers and unfreezing the last few dense layers will decrease the calculation time.(https://arxiv.org/pdf/1706.04983.pdf)

## Numerical results
VGG19 512 dense, frozen convs 0.98614

VGG19 2x 128, 2x 0.5 dropout, frozen convs 0.98185

VGG19 unfrozen, dense 512, 0.98857

VGG16 last 3 layers unfrozen, rest frozen, 512 Dense 0.9921

Best result so far: 3 convs, random overlapping fractional max pooling, dense + dropconnect, normal + rotate + affine training, default validation 0.99071

XCeption didn't work.
