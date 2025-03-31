import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
class VAE_Encoder(nn.Sequential):  # encoder class inheriting from parent nn.sequential class
    def __init__(self):
        super().__init__(
            #main idea --> is to decrease the image size and increasing the features
            # (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            #(Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            #(Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            #(Batch_Size, 128, Height / 2, Width / 2) /2 due to strides
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            #(Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),
            #(Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),
            # (Batch_Size,(Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            #(Batch_Size, 512, Height / 4, Width / 4) since prev is /2
            VAE_ResidualBlock(256, 512),
            #(Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),
            #(Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            #(Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            #(Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            #(Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            #(Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512),
            #(Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # adding the normalization  shape remains same (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512),

            # (Batch_Size, 512, Height / 8, Width / 8)
            # as per the paper
            nn.SiLU(),

            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8).
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # paddinf handling the shape

            # (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        # x-->(Batch_Size, Channel, Height, Width)
        # noise--> same size of encoder output(Batch_Size, 4, Height / 8, Width / 8)

        for module in self: # loop through each of the layer

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric,(Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).# Pad with zeros on the right and bottom.
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)
        # the output of the last layer of our module x is ->(Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # variance clamping
        log_variance = torch.clamp(log_variance, -30, 20)
        #(Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        #(Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()  #mean to std dev

        # Transform N(0, 1) -> N(mean, stdev) according to forward algo--> sampling from mean to stdev
        # (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise   #sampling task

        # Scale by a constant as per the paper
        x *= 0.18215

        return x

