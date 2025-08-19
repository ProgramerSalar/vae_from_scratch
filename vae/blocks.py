import torch 
from torch import nn 




def get_input_layer(
        in_channels: int,
        out_channels: int,
        norm_norm_groups: int,
        layer_type: str,
        norm_type: str = 'group',
        affine: bool = True
):
    
    if layer_type == 'conv':
        input_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    elif layer_type == 'pixel_shuffle':
        input_layer = nn.Sequential(
            nn.PixelUnshuffle(2)
        )