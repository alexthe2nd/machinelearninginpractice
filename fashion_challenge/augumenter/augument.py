# Uses the Pillow library (pip3 install pillow).
from PIL import Image
from PIL import ImageFilter
from random import randint

# Weighted random image effect, tends towards horizontal flip.
def random_effect(im):
    r = randint(0,9)
    if (r <= 5):
        # Contrast Stretching
        return apply_contrast_stretching(im)
    elif (r == 6):
        # Horizontal flip.
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    elif (r == 7):
        # Smooth - lesser blur.
        return im.filter(ImageFilter.SMOOTH)
    elif (r == 8):
        # Blur.
        return im.filter(ImageFilter.BLUR)
    else:
        # Sharpen.
        return im.filter(ImageFilter.SHARPEN)

        """
        # Greyscale
        img_bw = image.convert('L')
        """

# Crop and resize
def make_square(im, min_size, fill_color):
    x, y = im.size
    size = max(min_size, x, y)

    # Create new blank image and paste source image in its center.
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))

    # Workaround for resizing. Remove if you already feed it resized images.
    resized_im = new_im.resize((min_size, min_size))
    return resized_im

# Process red band of the image
def normalize_red(intensity):
    min_pixel_in = 86 
    max_pixel_in = 230

    min_pixel_out = 0
    max_pixel_out = 255 

    out_pixel = (intensity - min_pixel_in) * (((max_pixel_out - min_pixel_out)/(max_pixel_in - min_pixel_in))+ min_pixel_out)
    return out_pixel
         
# Process green band of the image
def normalize_green(intensity):
    min_pixel_in = 90
    max_pixel_in = 225

    min_pixel_out = 0
    max_pixel_out = 255 

    out_pixel = (intensity - min_pixel_in) * (((max_pixel_out - min_pixel_out)/(max_pixel_in - min_pixel_in))+ min_pixel_out)
    return out_pixel

def normalize_blue(intensity):
     
    min_pixel_in = 100 
    max_pixel_in = 210

    min_pixel_out = 0
    max_pixel_out = 255 

    out_pixel = (intensity - min_pixel_in) * (((max_pixel_out - min_pixel_out)/(max_pixel_in - min_pixel_in))+ min_pixel_out)
    return out_pixel

def apply_contrast_stretching(image_object):
    
    # Split R, G, B bands from the image. 

    multi_bands = image_object.split()

    # Apply contrast stretching on each color band.

    norm_red   = multi_bands[0].point(normalize_red)
    norm_green = multi_bands[1].point(normalize_green)
    norm_blue  = multi_bands[2].point(normalize_blue)

    # Merge the above normalization to retrieve new constrast stretched image.

    contrast_stretched_img = Image.merge("RGB", (norm_red, norm_green, norm_blue))

    return contrast_stretched_img

# Augumenter
def aug(image, min_size = 599, fill_color = "white"):
    im = Image.open(image)
    augumented = make_square(random_effect(im), min_size, fill_color)
    augumented.show()
    return augumented
