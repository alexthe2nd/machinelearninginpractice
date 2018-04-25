from PIL import Image
from PIL import ImageFilter
from random import randint

# Weighted random image effect, tends towards horizontal flip.
def random_effect(im):
    r = randint(0,9)
    if (r < 6):
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

# Augumenter
def aug(image, min_size = 299, fill_color = "white"):
    im = Image.open(image)
    augumented = make_square(random_effect(im), min_size, fill_color)
    # augumented.show()
    return augumented
