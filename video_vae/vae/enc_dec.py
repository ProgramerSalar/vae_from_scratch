import torch 
from torch import nn 
from typing import List, Tuple

from conv import CausalConv3d
from blocks import CausalDownBlock3d

class CausalEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channels: List = [128, 256, 512, 512],
                 down_num_layers: int = 2,
                 encoder_num_layers: int = 4,
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32,
                 add_height_width_2x: Tuple[bool, ...] = (True, True, True, False),
                 add_frame_2x: Tuple[bool, ...] = (True, True, True, False),
                 double_z: bool = True):
        
        super().__init__()

        # [2, 3, 8, 256, 256] -> [2, 128, 8, 256, 256]
        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channels=channels[0])
        
        self.encoder_block_layers = nn.ModuleList([])
        output_channels = channels[0]
        for i in range(encoder_num_layers):
            input_channels = output_channels
            output_channels = channels[i]

            # [128] -> [128]
            # [128] -> [256]
            # [256] -> [512]
            # [512] -> [512]

            self.encoder_block_layers.append(
                CausalDownBlock3d(in_channels=input_channels,
                                  out_channels=output_channels,
                                  num_layers=down_num_layers,
                                  dropout=dropout,
                                  eps=eps,
                                  scale_factor=scale_factor,
                                  norm_num_groups=norm_num_groups,
                                  add_height_width_2x=add_height_width_2x,
                                  add_frame_2x=add_frame_2x)
            )