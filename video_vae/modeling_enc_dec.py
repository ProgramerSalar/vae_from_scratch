import torch 
from dataclasses import dataclass
from diffusers.utils import BaseOutput, is_torch_version
import torch.utils.checkpoint
from .utils.utils import BaseOutput, is_torch_version

import torch.nn as nn 

from typing import Tuple
from .modeling_causal_conv import CausalConv3d, CausalGroupNorm
from .modeling_block import get_down_block, CausalUNetMidBlock2D, get_up_block

@dataclass
class DecoderOutput(BaseOutput):

    r""" 
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class CausalVaeEncoder(nn.Module):

    r""" 
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels 
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels 
        down_block_types (`Tuple[str, ...]` *optional*, defaults to `("DownEncoderBlock2D", "DownEncoderBlockCausal3D")`):
            The types of down blocks
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)` ):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function
        double_z (`bool`, *optional*, defaults to `True`):
            Wether to double the number of output channels for the last block.
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str, ...] = ("DownEncoderBlockCausal3D", ),
            spatial_down_sample: Tuple[bool, ...] = (True,),
            temporal_down_sample: Tuple[bool, ...] = (False,),
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: Tuple[int, ...] = (2,),
            norm_num_groups: int = 32,
            act_fn: str = "silu",
            double_z: bool = True,
            block_dropout: Tuple[int, ...] = (0.0,),
            mid_block_add_attention=True
    ):
        
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=3,
            stride=1
        )

        self.mid_block = None 
        self.down_blocks = nn.ModuleList([])

        # Down block
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type=down_block_type,
                num_layers=self.layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                add_spatial_downsample=spatial_down_sample[i],
                add_temporal_downsample=temporal_down_sample[i],
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
                dropout=block_dropout[i]
            )
            self.down_blocks.append(down_block)

        # mid block 
        self.mid_block = CausalUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            dropout=block_dropout[-1]
        )

        # output block 
        self.conv_norm_out = CausalGroupNorm(num_channels=block_out_channels[-1],
                                             num_groups=norm_num_groups,
                                             eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels 
        self.conv_out = CausalConv3d(in_channels=block_out_channels[-1],
                                     out_channels=conv_out_channels,
                                     kernel_size=3,
                                     stride=1)
        
        self.gradient_checkpointing = False

    
    def forward(self, 
                sample: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                
                return custom_forward
            
            # down block 
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, is_init_image,
                        temporal_chunk, use_reentrant=False
                    )

            # middle block 
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), sample, is_init_image, temporal_chunk, use_reentrant=False
            )

        else:

            # down 
            for down_block in self.down_blocks:
                sample = down_block(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

            # middle 
            sample = self.mid_block(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        
        # pos-process 
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        return sample 
    


class CausalVaeDecoder(nn.Module):

    r""" 
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.
    
    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, default to `("UpDecoderBlock2D", "UpDecoderBlockCausal3D",)`):
            The types of up blocks
        block_out_channels (`Tuple[int, ...]`, *optional*, default to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, default to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            up_block_types: Tuple[str, ...] = ("UpDecoderBlockCausal3D",),
            spatial_up_sample: Tuple[bool, ...] = (True,),
            temporal_up_sample: Tuple[bool, ...] = (False,),
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: Tuple[int, ...] = (2,),
            norm_num_groups: int = 32,
            act_fn: str = "silu",
            mid_block_add_attention=True,
            interpolate: bool = True,
            block_dropout: Tuple[int, ...] = (0.0,)
    ):
        
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=block_out_channels[-1],
            kernel_size=3,
            stride=1
        )

        self.mid_block = None 
        self.up_blocks = nn.ModuleList([])

        # mid block 
        self.mid_block = CausalUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            dropout=block_dropout[-1]
        )

        # up block 
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1 

            up_block = get_up_block(up_block_type=up_block_type,
                                    num_layers=self.layers_per_block[i],
                                    in_channels=prev_output_channel,
                                    out_channels=output_channel,
                                    prev_output_channel=None,
                                    add_spatial_upsample=spatial_up_sample[i],
                                    add_temporal_upsample=temporal_up_sample[i],
                                    resnet_eps=1e-6,
                                    resnet_act_fn=act_fn,
                                    resnet_groups=norm_num_groups,
                                    attention_head_dim=output_channel,
                                    temb_channels=None,
                                    resnet_time_scale_shift="default",
                                    interpolate=interpolate,
                                    dropout=block_dropout[i])
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out 
        self.conv_norm_out = CausalGroupNorm(num_channels=block_out_channels[0],
                                             num_groups=norm_num_groups,
                                             eps=1e-6)
        
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[0],
                                     out_channels, 
                                     kernel_size=3,
                                     stride=1)
        
        self.gradient_checkpointing = False 


    def forward(self,
                sample: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                
                return custom_forward
            
            if is_torch_version(">=", "1.11.0"):

                # middle block 
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    is_init_image=is_init_image,
                    temporal_chunk=temporal_chunk,
                    use_reentrant=False
                )
                sample = sample.to(upscale_dtype)

                # up block 
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        is_init_image=is_init_image,
                        temporal_chunk=temporal_chunk,
                        use_reentrant=False
                    )

            else:

                # middle 
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    is_init_image=is_init_image,
                    temporal_chunk=temporal_chunk
                )
                sample = sample.to(upscale_dtype)

                # up 
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block),
                                                               sample,
                                                               is_init_image=is_init_image,
                                                               temporal_chunk=temporal_chunk)
                    
        else:

            # middle 
            sample = self.mid_block(sample,
                                    is_init_image=is_init_image,
                                    temporal_chunk=temporal_chunk)
            sample = sample.to(upscale_dtype)

            # up 
            for up_block in self.up_blocks:
                sample = up_block(sample,
                                  is_init_image=is_init_image,
                                  temporal_chunk=temporal_chunk)
                
        
        # post-process 
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample,
                               is_init_image=is_init_image,
                               temporal_chunk=temporal_chunk)
        
        return sample
    

            



        

if __name__ == "__main__":

    causal_vae_encoder = CausalVaeEncoder().to("cuda:0")
    x = torch.randn(2, 3, 8, 512, 512).to("cuda:0")
    output = causal_vae_encoder(x)
    print(output)