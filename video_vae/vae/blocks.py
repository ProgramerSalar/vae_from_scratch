import torch 
from torch import nn 

from resnet import CausalResnetBlock3D, CausalHeightWidth2x, CausalFrame2x

class CausalDownBlock3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 eps: float = 1e-5,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32,
                 add_height_width_2x: bool = True,
                 add_frame_2x: bool = True):
        
        super().__init__()
        self.add_height_width_2x = add_height_width_2x
        self.add_frame_2x = add_frame_2x

        self.block_layers = nn.ModuleList([])
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            self.block_layers.append(
                    CausalResnetBlock3D(in_channels=input_channels,
                                        out_channels=out_channels,
                                        dropout=dropout,
                                        eps=eps,
                                        scale_factor=scale_factor,
                                        norm_num_groups=norm_num_groups)
            )

        ## <-- Decrease the (height, width) dim --> ##
        if self.add_height_width_2x:
            self.height_width_dims = nn.ModuleList([
                CausalHeightWidth2x(
                    in_channels=input_channels,
                    out_channels=out_channels
                )
            ])

        ## <-- Descrease the (frame) dim --> ##
        if self.add_frame_2x:
            self.frame_dims = nn.ModuleList([
                CausalFrame2x(in_channels=input_channels,
                              out_channels=out_channels)
            ])


    def forward(self, x):

        for block_layer in self.block_layers:

            x = block_layer(x)

        if self.add_height_width_2x:
            for height_width_dim in self.height_width_dims:
                x = height_width_dim(x)


        if self.add_frame_2x:
            for frame_dim in self.frame_dims:
                x = frame_dim(x)

        return x 
    



        