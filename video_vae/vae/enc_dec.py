import torch 
from torch import nn 
from typing import List, Tuple

from .conv import CausalConv3d, CausalGroupNorm
from .blocks import CausalDownBlock3d, CausalMiddleBlock3d

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


        # <-- Mid block --> 
        self.mid_block_layer = CausalMiddleBlock3d(in_channels=channels[-1],
                                                   attention_head_dim=512,
                                                   norm_num_groups=norm_num_groups,
                                                   dropout=dropout,
                                                   scale_factor=scale_factor,
                                                   eps=eps)
        
        self.conv_norm_out = CausalGroupNorm(in_channels=channels[-1],
                                             num_groups=norm_num_groups,
                                             eps=eps)
        
        self.act_fn = nn.SiLU()
        conv_output_channels = 2 * out_channels if double_z else out_channels

        self.conv_output = CausalConv3d(in_channels=channels[-1],
                                        out_channels=conv_output_channels,
                                        kernel_size=3,
                                        stride=1)
        
        self.gradient_checkpointing = False


    def forward(self, 
                x: torch.FloatTensor) -> torch.FloatTensor:
        
        sample = self.conv_in(x)

        if self.training and self.gradient_checkpointing:

            pass 

        else:
            # down
            for down_block in self.encoder_block_layers:
                sample = down_block(sample)

            # middle block 
            sample = self.mid_block_layer(sample)


        # post process 
        sample = self.conv_norm_out(sample)
        sample = self.act_fn(sample)
        sample = self.conv_output(sample)

        return sample

        
