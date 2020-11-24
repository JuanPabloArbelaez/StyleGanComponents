import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.stats import truncnorm



def get_truncated_noise(n_samples, z_dim, truncation):
    """Function for creating truncated noise vectors: Given dimensions (n_samples, z_dim)
        and truncation value, cretes a tensor of that shape filled with random
        numbers from the truncated normal distribution.

    Args:
        n_samples (int): the number of samples to generate
        z_dim (int): the dimension of the noise vector
        truncation (float): the truncation value
    """
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
    
    return truncated_noise


class MappingLayers(nn.Module):
    """Mapping Layers Class

    Args:
        z_dim (int): the deimension of the noise vector
        hidden_dim (int): the inner dimension
        w_dim (int): the dimension of the intermediate noise vector (w)
    """
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, w_dim),
        )

    def forward(self, noise):
        """Method for completing a forward pass of MappingLayers:
            Given an inital noise tensor, returns the intermediate noise tensor.

        Args:
            noise (tensor): A noise tensor with dimensions (n_samples, z_dim)
        """
        return self.mapping(noise)


class InjectNoise(nn.Module):
    """Inject Noise Class

    Args:
        channels (int): the number of channels the image has
    """
    def __init__(self, channels):
        super().__init__()
        weight = torch.empty(channels).normal_(mean=0, std=1)
        self.weight = nn.Parameter(torch.reshape(weight, (1, channels, 1, 1)))

    def forward(self, image):
        """Function for completing a forward pass of InjectNoise: Given an image,
            returns the image with random noise added.

        Args:
            image (tensor): the feature map of shape (n_samples, channels, width, height)
        """
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        noise = torch.randn(noise_shape, device=image.device)
        return image + (self.weight * noise)


class AdaIN(nn.Module):
    """AdaIn Class

    Args:
        channels (int): the number of channels the image has
        w_dim (int): the dimension of the intermediate noise vector
    """
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        """Method for completing a forward pass of AdaIN: Given an image and the intermediate noise vector w,
            returns the normalized image that has been scaled an shifted by the style

        Args:
            image (tensor): the feature map of shape (n_samples, channels, width, height)
            w (tensor): the intermediate noise vector
        """
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        transformed_image = (normalized_image * style_scale) + style_shift
        
        return transformed_image


class MicroStyleGANGeneratorBlock(nn.Module):
    """Micro StyleGAN Generator Block Class

    Args:
        in_chan (int): the number of channels in the input
        out_chan (int): the number of channels wanted in the output
        w_dim (int): the dimension of the intermediate noise vector
        kernel_size (int): the size of the convolutional kernel
        starting_size (int): the size of the starting image
    """
    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample

        if self.use_upsample:
            self.upsample = nn.Upsample((starting_size, starting_size), mode="bilinear")
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=1)
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, w):
        """Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x, and w,
        computes a StyleGAN generator block

        Args:
            x (tensor): the input into the generator, a feature map of sjhape (n_samples, channels, width, height)
            w (tensor): the intermediate noise vector
        """

        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.adain(x, w)
        return x


class MicroStyleGANGenerator(nn.Module):
    """Micro StyleGAN Generator Class

    Args:
        z_dim (int): the dimension of the noise vector
        map_hidden_dim (int): the mapping inner dimension
        w_dim (int): the dimension of the intermediate noise vector
        in_chan (int): the number of channels in the input
        out_chan (int): the number of channels wanted in the output
        kernel_size (int): the size of the convolutional kernel
        hidden_chan (int): the inner dimension
    """
    def __init__(self, z_dim, map_hidden_dim, w_dim, in_chan, out_chan, kernel_size, hidden_chan):
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8) 
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16) 

        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = 0.2

    def upsample_to_match_size(self, smaller_image, bigger_image):
        """Function for upsampling an image to the size of another: Given two images (smaller and bigger),
            upsamples the first to have the same dims as the second one.

        Args:
            smaller_image (tensor): the smaller image to upsample
            bigger_image (tensor): the bigger images, whose dimension will be upsampled to
        """
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode="bilinear")

    def forward(self, noise, return_intermediate=False):
        """Method for completing a forward pass of MicroStyleGANGenerator: Given noise, 
            computes a StyleGan interation.

        Args:
            noise (tensor): Noise tensor with dimensions (n_samples, z_dim)
            return_intermediate (bool, optional): True to return images (for testing)
        """
        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)
        x_small = self.block1(x, w)
        x_small_image = self.block1_to_image(x_small)
        x_big = self.block2(x_small, w)
        x_big_image = self.block2_to_image(x_big)
        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image)

        interpolation = torch.lerp(x_small_upsample, x_big_image, self.alpha)

        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
        return interpolation
