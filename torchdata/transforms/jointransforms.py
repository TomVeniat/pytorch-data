import numbers
import random

from PIL import ImageOps, Image


class JointRandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
        the given size. size can be a tuple (target_height, target_width)
        or an integer, in which case the target will be of a square shape (size, size)
        """
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, input, target):
        if self.padding > 0:
            input = ImageOps.expand(input, border=self.padding, fill=0)
            target = ImageOps.expand(target, border=self.padding, fill=0)

        w, h = input.size
        th, tw = self.size
        if w == tw and h == th:
            return input, target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return input.crop((x1, y1, x1 + tw, y1 + th)), target.crop((x1, y1, x1 + tw, y1 + th))


class JointRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, input, target):
        if random.random() < 0.5:
            return input.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
        return input, target


class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for jt in self.transforms:
            img, label = jt(img, label)
        return img, label


class JointPad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number)
        self.padding = padding
        self.fill = fill

    def __call__(self, input, target):
        return ImageOps.expand(input, border=self.padding, fill=self.fill), ImageOps.expand(target, border=self.padding, fill=self.fill)