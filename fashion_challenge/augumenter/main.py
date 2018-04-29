from augument import aug
import os

# Set the local directory of images.
# TODO: Change to your actual directory.
img_dir = 'Train/'
# Testing with one image || spec_image = 'id_7_labels_[114, 222, 113, 176, 214, 87].jpg'
# List contents of the image folder.
files = os.listdir(img_dir)

# Augument each image.
for f in files:
    if f.endswith('.jpg'):
        image = aug(img_dir + f)
        # Comment the following line out after testing.
        image.show()

"""
Usage of image augumenter:
- Takes image name, size, and bg color as arguments.
- 'size' defaults to '299', and 'bg_color' to 'white'.

Example function call:
    image = aug('test.jpg', 512, "black")
"""
