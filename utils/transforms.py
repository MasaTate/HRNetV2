import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch
import random
from typing import Tuple
import numpy as np
from PIL import Image

class PairRandomCrop:
    """Crop the given PIL Image at a random location and size.
    Args:
        min_size (sequence): Desired minimum output size of the crop.
        max_size (sequence): Desired maximum output size of the crop.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """
    def __init__(self, size=(512, 1024), padding=0, pad_if_needed=False):
        assert isinstance(size, Tuple) and len(size) == 2
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        
    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h -th)
        j = random.randint(0, w - tw)
        
        return i, j, th, tw
        
    def __call__(self, img, trg):
        """
        input is PIL Image.
        Be careful that <PIL Image>.size will return the tupple of (width, height)
        """
        
        assert img.size == trg.size, 'size of image and target should be the same. %s, %s'%(img.size, trg.size)
        
        if self.padding > 0:
            img = F.pad(img, self.padding)
            trg = F.pad(trg, self.padding)
            
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0] / 2)))
            trg = F.pad(trg, padding=int((1 + self.size[1] - trg.size[0] / 2)))
            
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            trg = F.pad(trg, padding=int((1 + self.size[0] - trg.size[1]) / 2))
            
        i, j, h, w = self.get_params(img, self.size)
        
        return F.crop(img, i, j, h, w), F.crop(trg, i, j, h, w)
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1}'.format(self.size, self.padding)

class PairRandomScale:
    """Scale the given PIL Image at random rate
    scale_range(tuple):Desired minimum and maximum scaling rate
    """
    def __init__(self, scale_range=(0.5, 2)):
        assert isinstance(scale_range, Tuple)
        self.scale_range = scale_range
        self.rate = 1.0
        
    def __call__(self, img, trg):
        """
        input is PIL Image.
        Be careful that <PIL Image>.size will return the tupple of (width, height)
        """
        assert img.size == trg.size, 'size of image and target should be the same. %s, %s'%(img.size, trg.size)
        
        self.rate = random.uniform(self.scale_range[0], self.scale_range[1])
        out_size = ( int(img.size[1] * self.rate), int(img.size[0] * self.rate) )
        img = F.resize(img, out_size, F.InterpolationMode.BILINEAR)
        trg = F.resize(trg, out_size, F.InterpolationMode.NEAREST)
        return img, trg
    
    def __repr__(self):
        return self.__class__.__name__ + 'rate={}'.format(self.rate)


class PairRandomHorizontalFlip:
    def __init__(self, p=0.5):
        assert p <= 1.0 and p >= 0, 'p should be in the range of [0.0, 1.0]'
        self.p = p
        
    def __call__(self, img, trg):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(trg)
        return img, trg
    
class PairCompose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, trg):
        for t in self.transforms:
            img, trg = t(img, trg)
        return img, trg
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '     {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class PairToTensor:
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize=normalize
        self.target_type = target_type
        
    def __call__(self, img, trg):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor. 
        Returns:
            Tensor: Converted image and label
        """
        if self.normalize:
            return F.to_tensor(img), torch.from_numpy(np.array(trg, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1)), torch.from_numpy(np.array(trg, dtype=self.target_type))
        
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class PairNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, img_tensor, trg):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        return F.normalize(img_tensor, self.mean, self.std), trg
    