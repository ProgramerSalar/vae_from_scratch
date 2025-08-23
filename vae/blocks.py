import torch 
from torch import nn 
import warnings
from .resnet import (
    vae_Block3D, 
    vae_Downsample, 
    vae_TemporalDownsample, 
    vae_Upsample, 
    vae_TemporalUpsample
)
from typing import Union, Optional, Literal
from einops import rearrange

from diffusers.models.attention_processor import Attention

def vae_input_layer(
        in_channels: int,
        out_channels: int,
        layer_type: Literal["conv", "pixel_shuffle"],
):
    
    """ 
        so this is input layer function.

        Args:
            in_channels (`int`): input channels 
            out_channels (`int`): output channels 
            layer_type (`str`): Types of the conv layer (options: [`"conv"`, `"pixel_shuffle"`])
    """
    
    if layer_type == 'conv':
        input_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    elif layer_type == 'pixel_shuffle':
        input_layer = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)
        )

    else:
        raise NotImplementedError(f"Invalid layer_type: {layer_type}")
    
    return input_layer


def vae_output_layer(
        in_channels: int,
        out_channels: int,
        layer_type: Literal["norm_act_conv", "pixel_shuffle"] = "norm_act_conv",
        norm_num_groups: int = 32,
        affine: bool = True
):
    
    """ 
    so this is output layer function.
    
    Args:
        in_channels (`int`): input channels
        out_channels (`int`): output channels 
        layer_type (`str`): type of the layer (default: `"norm_act_conv"`)
        norm_num_groups (`int`): normalization groups (default: `32`)
        affine: (`bool`): affine parameters (default: `True`)
    """
    
    if layer_type == "norm_act_conv":
        output_layer = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, 
                         num_channels=in_channels,
                         eps=1e-6,
                         affine=affine),
            nn.SiLU(),
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        )

    elif layer_type == "pixel_shuffle":
        output_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels * 4,
                      kernel_size=3),
            nn.PixelShuffle(2)
        )

    else:
        raise NotImplementedError(f"Not support output layer {layer_type}")
    
    return output_layer



class vae_DownEncoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            eps: float = 1e-6,
            time_scale_shift: Literal["ada_group", "default"] = "default",
            act_fn: Literal["swish", "silu", "mish", "gelu", "relu"] = "swish",
            groups: int = 32,
            pre_norm: bool = True,
            output_scale_factor: float = 1.0,
            add_spatial_downsample: bool = True,
            add_temporal_downsample: bool = False,
            
    ):
        
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                vae_Block3D(in_channels=in_channels,
                            out_channels=out_channels,
                            dropout=dropout,
                            temb_channels=None,
                            groups=groups,
                            pre_norm=pre_norm,
                            eps=eps,
                            non_linearity=act_fn,
                            time_embedding_norm=time_scale_shift,
                            output_scale_factor=output_scale_factor
                            )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_spatial_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    vae_Downsample(channels=out_channels,
                                   use_conv=True,
                                   out_channels=out_channels)
                ]
            )
        else:
            self.downsamplers = None
            
        if add_temporal_downsample:
            self.temporal_downsamplers = nn.ModuleList(
                [
                    vae_TemporalDownsample(
                        channels=out_channels,
                        use_conv=True,
                        out_channels=out_channels
                    )
                ]
            )
        else:
            self.temporal_downsamplers = None


        


    def forward(self, 
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False,
                ) -> torch.FloatTensor:
        
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states,
                                   temb=None,
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)
            
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states,
                                            is_init_image=is_init_image,
                                            temporal_chunk=temporal_chunk)
                
            
        if self.temporal_downsamplers is not None:
            for temporal_downsampler in self.temporal_downsamplers:
                hidden_states = temporal_downsampler(hidden_states,
                                                     is_init_image=is_init_image,
                                                     temporal_chunk=temporal_chunk)


        return hidden_states
    



    

class vae_UpDecoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            eps: float = 1e-6,
            time_scale_shift: Literal["default", "ada_group"] = "default",
            act_fn: Literal["swish", "silu", "mish", "gelu", "relu"] = "swish",
            groups: int = 32,
            pre_norm: bool = True,
            output_scale_factor: float = 1.0,
            add_spatial_upsample: bool = True,
            add_temporal_upsample: bool = False,
            temb_channels: int = None,
            interpolate: bool = False
    ):
        
        super().__init__()

        resnets = []
        
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                vae_Block3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=groups,
                    pre_norm=pre_norm,
                    eps=eps,
                    non_linearity=act_fn,
                    time_embedding_norm=time_scale_shift,
                    output_scale_factor=output_scale_factor
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_spatial_upsample:
            self.upsamplers = nn.ModuleList([
                vae_Upsample(channels=out_channels,
                             use_conv=True,
                             out_channels=out_channels,
                             interpolate=interpolate)
            ])

        else:
            self.upsamplers = None

        if add_temporal_upsample:
            self.temporal_upsampler = nn.ModuleList([
                vae_TemporalUpsample(channels=out_channels,
                                     use_conv=True,
                                     out_channels=out_channels,
                                     interpolate=interpolate)
            ])

        else:
            self.temporal_upsampler = None

        


    def forward(self, 
                hidden_states: torch.FloatTensor,
                temb: torch.FloatTensor = None,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states,
                                   temb=temb,
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)
            
            if self.upsamplers is not None:
                for upsample in self.upsamplers:
                    hidden_states = upsample(hidden_states, 
                                             is_init_image=is_init_image,
                                             temporal_chunk=temporal_chunk)
                    
            if self.temporal_upsampler is not None:
                for temporal_upsampler in self.temporal_upsampler:
                    hidden_states = temporal_upsampler(hidden_states,
                                                       is_init_image=is_init_image,
                                                       temporal__chunk=temporal_chunk)
            

        return hidden_states
    



class vae_MidBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            temb_channels: int = None, # this option to get new 
            dropout: float = 0.0,
            num_layers: int = 1,
            eps: float = 1e-6,
            time_scale_shift:  Literal["ada_group", "default"] = "default",
            act_fn: Literal["swish", "silu", "mish", "gelu", "relu"] = "swish",
            groups: int = 32,
            attention_groups: Optional[int] = None,
            pre_norm: bool = True,
            add_attention: bool = True,  # this option is new 
            attention_head_dim: int = 1,    # this option is new 
            output_scale_factor: float = 1.0
    ):
        
        super().__init__()
        groups = groups if groups is not None else 32 
        self.add_attention = add_attention

        if attention_groups is None:
            attn_groups = groups if time_scale_shift == "default" else None


        resnets = [
            vae_Block3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=eps,
                groups=groups,
                dropout=dropout,
                time_embedding_norm=time_scale_shift,
                non_linearity=act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=pre_norm
            )
        ]

        attentions = []

        if attention_head_dim is None:
            warnings("make sure attention_head_dim is not None.")
        attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        query_dim=in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        dropout=dropout,
                        bias=True,
                        upcast_softmax=True,
                        rescale_output_factor=output_scale_factor,
                        residual_connection=True,
                        norm_num_groups=attention_groups,
                        eps=eps,
                        _from_deprecated_attn_block=True # if attention is load diprecated state dict
                    )
                )

            else:
                attentions.append(None)


            resnets.append(
                vae_Block3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=eps,
                    groups=groups,
                    dropout=dropout,
                    time_embedding_norm=time_scale_shift,
                    non_linearity=act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=pre_norm
                )
            )

        self.attention = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


    def forward(self,
                hidden_states: torch.FloatTensor,
                temb: Optional[torch.FloatTensor] = None,
                is_init_image=True,
                temporal_chunk=False,
                ) -> torch.FloatTensor:
        
        hidden_states = self.resnets[0](hidden_states, 
                                        temb, 
                                        is_init_image=is_init_image, 
                                        temporal_chunk=temporal_chunk)
        
        t = hidden_states.shape[2]

        for attn, resnet in zip(self.attention, 
                                self.resnets[1:]):
            if attn is not None:
                b, c, t, h, w = hidden_states.shape
                hidden_states = rearrange(hidden_states, 'b c t h w -> b t c h w')
                # convert video to image 
                hidden_states = rearrange(hidden_states, 'b t c h w -> (b t) c h w', b=b, t=t)
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(hidden_states, '(b t) c h w -> b t c h w', b=b, t=t)
                hidden_states = rearrange(hidden_states, 'b t c h w -> b c t h w')

            hidden_states = resnet(hidden_states,
                                   temb,
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)
            

        return hidden_states






if __name__ == "__main__":

            ### Down block ###
    # in_channels = 64
    # out_channels = 64
    # num_layers = 5
    # groups = 2 


    # vae_down_encoder = vae_DownEncoder(in_channels=in_channels,
    #                                    out_channels=out_channels,
    #                                    num_layers=num_layers,
    #                                    groups=groups)
    
    # print(vae_down_encoder)
    
    # x = torch.randn(2, 64, 8, 64, 64)
    # vae_down_encoder = vae_down_encoder(x)
    # print(vae_down_encoder.shape)
# ---------------------------------------------------------------------------

    # in_channels = 8
    # out_channels = 8 
    # groups = 2


    # vae_up_decoder = vae_UpDecoder(in_channels=in_channels,
    #                                out_channels=out_channels,
    #                                groups=groups)
    
    # x = torch.randn(3, 8, 8, 16, 16)
    # vae_up_decoder = vae_up_decoder(x)
    # print(vae_up_decoder.shape)

# -----------------------------------------------------------------------------

    ## Mid block ## 

    in_channels = 8
    temb_channels = 8
    groups = 2 
    attention_head_dim = 8

    vae_mid_block = vae_MidBlock(in_channels=in_channels,
                                 temb_channels=temb_channels,
                                 groups=groups,
                                 attention_head_dim=attention_head_dim
                                 )
    
    print(vae_mid_block)

    x = torch.randn(2, 8, 8, 16, 16)
    temb = torch.randn(2, temb_channels)
    vae_mid_block = vae_mid_block(x, temb)

    print(vae_mid_block.shape)
    
