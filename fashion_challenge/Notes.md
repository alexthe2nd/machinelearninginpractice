_Nothing here yet._ 
## Meeting May 7th
Laurens updates Zaheer about the progress.

Overfitting threshold on a validation set reduces the performance on the test set.

Zaheer proposes resampling (easy) or clustering (difficult) to make it more balanced. Here, we combine all the images that contain $x$ - $y$ (range) occurences of label Z. He will send us some papers. 

Zaheer asks about class balancing. A general approach can be a different threshold for different labels. 

### To research
* Hypercolumns. 
* Global average pooling to prevent OOM-exception.
* 

### To implement
* _Class balancing (preprocessing)._
* Generate baseline test on GCP, with Exception, VGG16, VGG19 and other pre-trained models to get a baseline.
* Cross-validation (however the number-of-labels-per-sample shouldn't be a problem)
