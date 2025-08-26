import torch 
from torch import nn 
from typing import Tuple, Literal, Optional
import numpy as np 

from .conv import vae_Conv3d, vae_GroupNorm
from .blocks import (
    vae_DownEncoder,
    vae_MidBlock,
    vae_UpDecoder
)

class VaeEncoder(nn.Module):

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: str = ("DownEncoderBlockCausal3D",),
            spatial_down_sample: Tuple[bool, ...] = (True,),
            temporal_down_sample: Tuple[bool, ...] = (False,),
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: Tuple[int, ...] = (2,),
            norm_num_groups: int = 32,
            act_fn: Literal["swish", "silu", "mish", "gelu", "relu"] = "silu",
            double_z: bool = True,
            block_dropout: Tuple[int, ...] = (0.0,),
            mid_block_add_attention=True
    ):
        

        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = vae_Conv3d(in_channels=in_channels,
                                  out_channel=block_out_channels[0],
                                  kernel_size=3,
                                  stride=1)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # Down 
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            
            
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = vae_DownEncoder(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block[i],
                eps=1e-6,
                act_fn=act_fn,
                groups=norm_num_groups,
                add_spatial_downsample=spatial_down_sample[i],
                add_temporal_downsample=temporal_down_sample[i],
                dropout=block_dropout[i]
            )

            self.down_blocks.append(down_block)


        # mid block 
        self.mid_block = vae_MidBlock(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=block_dropout[-1],
            eps=1e-6,
            act_fn=act_fn,
            groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            output_scale_factor=1.,
            time_scale_shift="default",
        )
        
            
        # output 
        self.conv_norm_out = vae_GroupNorm(num_groups=norm_num_groups,
                                           num_channels=block_out_channels[-1],
                                           eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = vae_Conv3d(in_channels=block_out_channels[-1],
                                   out_channel=conv_out_channels,
                                   kernel_size=3,
                                   stride=1)
        
        self.gradient_checkpointing = False


    def forward(self,
                sample: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        

    
        sample = self.conv_in(sample,
                              is_init_image=is_init_image,
                              temporal_chunk=temporal_chunk)
        
        if self.training and self.gradient_checkpointing:

            pass 

        else:
            # down block 
            for down_block in self.down_blocks:
                sample = down_block(sample, 
                                    is_init_image=is_init_image,
                                    temporal_chunk=temporal_chunk)
                
            # mid block 
            sample = self.mid_block(sample,
                                    is_init_image=is_init_image,
                                    temporal_chunk=temporal_chunk)
            

            # post process 
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample, 
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)
            

            return sample
        



class VaeDecoder(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 up_block_types: Tuple[str, ...] = ("UpDecoderBlockCausal3D",),
                 spatial_up_sample: Tuple[bool, ...] = (True,),
                 temporal_up_sample: Tuple[bool, ...] = (False,),
                 block_out_channels: Tuple[int, ...] = (64,),
                 layers_per_block: Tuple[int, ...] = (2,),
                 norm_num_groups: int = 32,
                 act_fn: str = "silu",
                 mid_block_add_attention = True,
                 interpolate: bool = False,
                 block_dropout: Tuple[int, ...] = (0.0,)
                 ):
        
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = vae_Conv3d(in_channels=in_channels,
                                  out_channel=block_out_channels[-1],
                                  kernel_size=3,
                                  stride=1)
        
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid block 
        self.mid_block = vae_MidBlock(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=block_dropout[-1],
            eps=1e-6,
            time_scale_shift="default",
            act_fn=act_fn,
            groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            output_scale_factor=1,
        )

        # up block 
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_types in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1 
            

            up_block = vae_UpDecoder(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                dropout=block_dropout[i],
                num_layers=self.layers_per_block[i],
                eps=1e-6,
                time_scale_shift="default",
                act_fn=act_fn,
                groups=norm_num_groups,
                output_scale_factor=1.,
                add_spatial_upsample=spatial_up_sample[i],
                add_temporal_upsample=temporal_up_sample[i],
                temb_channels=None,
                interpolate=interpolate
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel


        # output block 
        self.conv_norm_out = vae_GroupNorm(num_groups=norm_num_groups,
                                           num_channels=block_out_channels[0],
                                           eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = vae_Conv3d(in_channels=block_out_channels[0],
                                   out_channel=out_channels,
                                   kernel_size=3,
                                   stride=1)
        
        self.gradient_checkpointing = False 


    def forward(self, 
                sample: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False,
                ) -> torch.FloatTensor:
        

        sample = self.conv_in(sample,
                              is_init_image=is_init_image,
                              temporal_chunk=temporal_chunk)
        
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        if self.training and self.gradient_checkpointing:
            pass 

        else:
            # middle block 
            sample = self.mid_block(sample, 
                                    is_init_image=is_init_image,
                                    temporal_chunk=temporal_chunk)
            sample = sample.to(upscale_dtype)

            # up block 
            for up_block in self.up_blocks:
                # print(f"what is the shape of input: <> <> <> <> {sample.shape}")
                sample = up_block(sample,
                                  is_init_image=is_init_image,
                                  temporal_chunk=temporal_chunk)
                # print(f"what is the output shape: <> <> <> <> {sample.shape}")
                
            # post-process 
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample,
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)
            
            return sample
        

class DiagonalGaussianDistribution(object):

    def __init__(self,
                 parameters: torch.Tensor,
                 determinstic: bool = False,
                 ):
        
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, chunks=2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

        self.deterministic = determinstic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean,
                device=self.parameters.device,
                dtype=self.parameters.dtype
            )


    def sample(self,
               generator: torch.Generator = None) -> torch.FloatTensor:
        
        sample = torch.randn(size=self.mean.shape,
                             generator=generator,
                             device=self.parameters.device,
                             dtype=self.parameters.dtype)
        
        x = self.mean + self.std * sample
        return x 
    

    def kl(self, 
           other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        
        if self.deterministic:
            return torch.Tensor([0.0])
        
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[2, 3, 4]
                )
            
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var 
                    + self.var / other.var 
                    - 1.0
                    - self.logvar 
                    + other.logvar,
                    dim=[2, 3, 4]
                )
            
    
    def nll(self,
            sample: torch.Tensor,
            dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        
        print(f"what is the dims: {dims}")
        
        if self.deterministic:
            return torch.Tensor([0.0])
        
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims
        )
    

    def mode(self) -> torch.Tensor:
        return self.mean
    





        



if __name__ == "__main__":

    vae_encoder = VaeEncoder(norm_num_groups=2,
                            block_out_channels=(128, 256, 512, 512)
                             )
    print(vae_encoder)

    x = torch.randn(2, 3, 8, 256, 246)

    output = vae_encoder(x)
    print(output.shape)
# -----------------------------------------------------------------

    # x = torch.randn(2, 8, 8, 16, 16)
    # vae_decoder = VaeDecoder(norm_num_groups=2,
    #                          in_channels=8)
    # output = vae_decoder(x)
    # print(output.shape)

# --------------------------------------------

    # tensor = torch.randn(2, 3, 8, 256, 256)
    

    # diagonal_gaussian_distribution = DiagonalGaussianDistribution(parameters=tensor,
    #                                                               determinstic=False)
    # # print(diagonal_gaussian_distribution)

    # sample = diagonal_gaussian_distribution.sample(
    #     generator=torch.Generator("cpu")
    # )
    # print(sample.shape)
    # kl = diagonal_gaussian_distribution.kl()
    # # print(kl.shape)

    # nll = diagonal_gaussian_distribution.nll(sample=sample,
    #                                          )
    # print(nll.shape)
