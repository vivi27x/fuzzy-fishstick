# Create a transform which adds channels for magnitude, fourier spectrum and Local Binary Pattern (LBP) to the input image
import numpy as np
from skimage.feature import local_binary_pattern as lbp
import cv2 as cv
import einops
import torch

def _grayscale(image):
    return cv.cvtColor((image.cpu().numpy() * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)

def _FFT(gray):
    return 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9)
    
def _LBP(gray):
    return lbp(gray, 3, 6, method='uniform')    

def _add_new_channels(image):
    gray = _grayscale(image)

    new_channels = []
    new_channels.append(_FFT(gray))
    new_channels.append(_LBP(gray))

    new_channels = np.stack(new_channels, axis=2)/255
    return torch.from_numpy(new_channels).to('cpu').float()

def add_new_channels(image):
    image_copied = einops.rearrange(image, "c h w -> h w c")
    new_channels = _add_new_channels(image_copied)
    image_copied = torch.concatenate([image_copied, new_channels], dim=-1)
    image_copied = einops.rearrange(image_copied, "h w c -> c h w")

    return image_copied