# Copyright 2025 The savitri-AI Team. All rights reserved.
#
#    company: https://github.com/savitri-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch 
from torch import nn 
from typing import List, Union, Tuple

from .conv import CausalConv3d, CausalGroupNorm
from .blocks import CausalBlock3d, CausalMiddleBlock3d, CausalUpperBlock




class CausalEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channels: List = [128, 256, 512, 512],
                 num_layers: int = 2,
                 encoder_num_layers: int = 4,
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32,
                 add_height_width_2x: Tuple[bool, bool, bool, bool] = (True, True, True, False),
                 add_frame_2x: Tuple[bool, bool, bool, bool] = (True, True, True, False),
                 double_z: bool = True
                 ):
        super().__init__()

        # [2, 3, 8, 256, 256] -> [2, 3, 128, 256, 256]
        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channels=channels[0],
                                    )
        
        
        
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
                CausalBlock3d(in_channels=input_channels,
                            out_channels=output_channels,
                            num_layers=num_layers,
                            dropout=dropout,
                            eps=eps,
                            scale_factor=scale_factor,
                            norm_num_groups=norm_num_groups,
                            add_height_width_2x=add_height_width_2x[i],
                            add_frame_2x=add_frame_2x[i])
            )

        # mid block 
        self.mid_block_layer = CausalMiddleBlock3d(in_channels=channels[-1],
                                                   attention_head_dim=512,
                                                   norm_num_groups=norm_num_groups,
                                                   dropout=dropout,
                                                   scale_factor=scale_factor,
                                                   eps=eps)
        
        self.conv_norm_output = CausalGroupNorm(in_channels=channels[-1],
                                                num_groups=norm_num_groups,
                                                eps=eps)
        
        self.act_fn = nn.SiLU()
        conv_output_channels = 2 *  out_channels if double_z else out_channels
        self.conv_output = CausalConv3d(in_channels=channels[-1],
                                        out_channels=conv_output_channels,
                                        kernel_size=3,
                                        stride=1)
        
        self.gradient_checkpointing = False


    def forward(self, 
                sample: torch.FloatTensor) -> torch.FloatTensor:
        
        # [2, 3, 8, 256, 256] -> [2, 128, 8, 256, 256]
        sample = self.conv_in(sample)
        
        
        for encoder_block_layer in self.encoder_block_layers:
            # [2, 128, 8, 256, 256] -> [2, 128, 4, 128, 128]
            # [2, 128, 4, 128, 128] -> [2, 256, 2, 64, 64]
            # [2, 256, 2, 64, 64] -> [2, 512, 1, 32, 32]
            # [2, 512, 1, 32, 32] -> [2, 512, 1, 32, 32]
            sample = encoder_block_layer(sample)

        
        # mid block 
        sample = self.mid_block_layer(sample)

        sample = self.conv_norm_output(sample)
        sample = self.act_fn(sample)
        sample = self.conv_output(sample)
            
        # [2, 3, 8, 256, 256] -> [2, 2*3, 1, 32, 32]
        return sample
    

class CausalDecoder(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                channels: List = [128, 256, 512, 512],
                num_layers: int = 2,
                decoder_num_layers: int = 4,
                dropout: float = 0.0,
                eps: float = 1e-5,
                scale_factor: float = 1.0,
                norm_num_groups: int = 32,
                add_height_width_2x: Tuple[bool, bool, bool, bool] = (True, True, True, False),
                add_frame_2x: Tuple[bool, bool, bool, bool] = (True, True, True, False),
                
                 ):
        
        super().__init__()

        # [2, 6, 1, 256, 256] -> [2, 512, 1, 256, 256]
        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channels=channels[-1])
        

        self.mid_block_layer = CausalMiddleBlock3d(in_channels=channels[-1],
                                                   attention_head_dim=512,
                                                   norm_num_groups=norm_num_groups,
                                                   dropout=dropout,
                                                   scale_factor=scale_factor,
                                                   eps=eps)
        

        # upper block 
        self.up_block_layers = nn.ModuleList([])
        reversed_channels = list(reversed(channels))
        output_channels = reversed_channels[0]  # 512
        for i in range(decoder_num_layers):
            input_channels = output_channels
            output_channels = reversed_channels[i]

            # [512] -> [512]
            # [512] -> [512]
            # [512] -> [256]
            # [256] -> [128]
            up_block = CausalUpperBlock(in_channels=input_channels,
                                        out_channels=output_channels,
                                        num_layers=num_layers,
                                        norm_num_groups=norm_num_groups,
                                        add_height_width_2x=add_height_width_2x[i],
                                        add_frame_2x=add_frame_2x[i],
                                        dropout=dropout,
                                        eps=eps,
                                        scale_factor=scale_factor)
            self.up_block_layers.append(up_block)


        # output 
        self.conv_norm_out = CausalGroupNorm(in_channels=channels[0],
                                             num_groups=norm_num_groups,
                                             eps=eps)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(in_channels=channels[0],
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     stride=1)
        

    def forward(self, 
                sample: torch.FloatTensor):
        
        sample = self.conv_in(sample)
        sample = self.mid_block_layer(sample)

        for up_block in self.up_block_layers:
            sample = up_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    

if __name__ == "__main__":

    # causal_encoder = CausalEncoder(in_channels=3,
    #                                out_channels=4,
    #                                channels=[128, 256, 512, 512],
    #                                num_layers=2,
    #                                encoder_num_layers=4,
    #                                dropout=0.0,
    #                                eps=1e-6,
    #                                scale_factor=1.0,
    #                                norm_num_groups=2
    #                                )
    # print(causal_encoder)

    # x = torch.randn(2, 3, 8, 256, 256)
    # # (2, 3, 8, 256, 256) -> (2, 8, 1, 32, 32)
    # output = causal_encoder(x)
    # print(output.shape)
    # -------------------------------------------------------
    causal_decoder = CausalDecoder(in_channels=4, 
                                   out_channels=3,
                                   num_layers=3,
                                   eps=1e-6,
                                   norm_num_groups=2,
                                   )
    
    print(causal_decoder)
    z = torch.randn(2, 4, 1, 32, 32)

    # (2, 8, 1, 32, 32) -> (2, 3, 1, 256, 256)
    output = causal_decoder(z)
    print(output.shape)