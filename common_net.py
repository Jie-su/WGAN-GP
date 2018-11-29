import torch.nn as nn


class LeakyReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding):
        super(LeakyReLUINSConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False)]
        model += [nn.InstanceNorm2d(n_out, affine=True)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ReLUBNNConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding):
        super(ReLUBNNConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size,
                                     stride=stride, padding=padding,
                                     output_padding=1, bias=False)]
        model += [nn.BatchNorm2d(n_out, affine=False)]
        model += [nn.ReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
