import torch
import torch.nn as nn
from common_net import LeakyReLUINSConv2d, ReLUBNNConvTranspose2d
from configuration import *


# Generator for 64x64 input size image
class Generator(nn.Module):
    def __init__(self, hidden_dim=BASE_CHANNAL_DIM, noise_size=100):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.noise_size = noise_size

        # Project random noise to input size
        self.projector = nn.Sequential(
            nn.Linear(self.noise_size, 4 * 4 * 8 * self.hidden_dim),
            nn.BatchNorm1d(4 * 4 * 8 * self.hidden_dim),
            nn.ReLU(True),
        )

        # Deconvolution block
        self.deconv = nn.Sequential(

            # 128 * 8 x 4 x 4 -> 128 * 4 x 8 x 8
            ReLUBNNConvTranspose2d(8 * self.hidden_dim, 4 * self.hidden_dim,
                                   kernel_size=5, stride=2, padding=2),

            # 128 * 4 x 8 x 8 -> 128 * 2 x 16 x 16
            ReLUBNNConvTranspose2d(4 * self.hidden_dim, 2 * self.hidden_dim,
                                   kernel_size=5, stride=2, padding=2),

            # 128 * 2 x 16 x 16 -> 128 x 32 x 32
            ReLUBNNConvTranspose2d(2 * self.hidden_dim, 1 * self.hidden_dim,
                                   kernel_size=5, stride=2, padding=2),

            # 128 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(self.hidden_dim, 3, 2, stride=2),
            nn.Tanh(),
        )

    def forward(self, input):
        # Feedforward process
        output = self.projector(input)
        output = output.view(-1, 8 * self.hidden_dim, 4, 4)
        output = self.deconv(output)
        return output


# Discriminator for 64x64 input images
class Discriminator(nn.Module):
    def __init__(self, hidden_dim=BASE_CHANNAL_DIM, input_channel=3):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_channel = input_channel

        # Convolutional block
        self.conv_block = nn.Sequential(

            # 3 x 64 x 64 -> 128 x 32 x 32
            nn.Conv2d(self.input_channel, self.hidden_dim, 5, 2, padding=2),
            nn.LeakyReLU(0.2),

            # 128 x 32 x 32 -> 256 x 16 x 16
            LeakyReLUINSConv2d(self.hidden_dim,
                               2 * self.hidden_dim, 5, 2, padding=2),

            # 256 x 16 x 16 -> 512 x 8 x 8
            LeakyReLUINSConv2d(2 * self.hidden_dim,
                               4 * self.hidden_dim, 5, 2, padding=2),

            # 512 x 8 x 8 -> 1024 x 4 x 4
            LeakyReLUINSConv2d(4 * self.hidden_dim,
                               8 * self.hidden_dim, 5, 2, padding=2),

            # 1024 x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(8 * self.hidden_dim, 1, 4, 1, padding=0),
        )



    def forward(self, input):
        output = self.conv_block(input)

        return output.view(-1)
