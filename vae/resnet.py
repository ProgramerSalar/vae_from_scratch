import torch 
from torch import nn 
from typing import Optional, Union, Literal
from einops import rearrange, repeat, reduce

from .utils import SpatialNorm, get_activation, AdaGroupNorm
from .conv import vae_Conv3d




class vae_Block3D(nn.Module):


    """
    The conv3d block 

    Args:
        in_channel (`int`): input channels 
        out_channel (`int`): output channels 
        dropout (`float`): Dropout (default: `0.0`)
        temb_channels (`int`): time embedding channels (default: `512`)
        groups  (`int`): Batch size (default: `32`)
        groups_out  (`int`): same as groups (default: `None`)
        pre_norm  (`bool`): Initial normalization to conv layer (default: `True`)
        eps  (`float`): Exceptional Point (default: `1e-6`)
        non_linerity  (`str`): this is activation function (default: `"swish"` -> silu)
            Option [`"swish"`, `"silu"`, `"mish"`, `"gelu"`, `"relu"`]
        time_embedding_norm: this is the normalization (default: `"default"` -> GroupNorm)
            Option [`"ada_group"`, `"default"`]
        output_scale_factor (`1.0`): this is the scale factor to divide the output tensor (default: `1.0`)
    """


    def __init__(self,
                *,
                in_channels: int,
                out_channels: Optional[int] = None,
                dropout: float = 0.0,
                temb_channels: int = 512,
                groups: int = 32,
                groups_out: Optional[int] = None,
                pre_norm: bool = True,
                eps: float = 1e-6,
                non_linearity: Literal["swish", "silu", "mish", "gelu", "relu"] = "swish",   # activation function 
                time_embedding_norm: Literal["ada_group", "default"] = "default",   
                output_scale_factor: float = 1.0,
                ):
        
        
        super().__init__()
        

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.pre_norm = pre_norm
        if self.pre_norm:
            if self.time_embedding_norm == "ada_group":
                self.norm1 = AdaGroupNorm(embedding_dim=temb_channels,
                                        out_dim=in_channels,
                                        num_groups=groups,
                                        eps=eps)
                
            else:
                self.norm1 = torch.nn.GroupNorm(num_groups=groups,
                                            num_channels=in_channels,
                                            eps=eps,
                                            affine=True)
            

        self.conv1 = vae_Conv3d(in_channels=in_channels,
                                  out_channel=out_channels,
                                  kernel_size=3,
                                  stride=1)
        

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(embedding_dim=temb_channels,
                                      out_dim=out_channels,
                                      num_groups=groups_out,
                                      eps=eps)
            
            
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out,
                                            num_channels=out_channels,
                                            eps=eps,
                                            affine=True)
            
        
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = vae_Conv3d(in_channels=out_channels,
                                  out_channel=out_channels,     # out_channels
                                  kernel_size=3,
                                  stride=1
                                  )
        

        self.nonlinearity = get_activation(act_fn=non_linearity)
        self.upsample = self.downsample = None
        
            
        
    def forward(self,
                input_tensor: torch.FloatTensor,
                temb: torch.FloatTensor = None,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        

        hidden_states = input_tensor
        
        batch_size, channels, frame, height, width = hidden_states.shape
        
        # passing the data in normalization [1st-step]
        if self.time_embedding_norm == "ada_group":
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w')
            temb = repeat(temb, 'b c -> (b f) c', f=frame)
            hidden_states = self.norm1(hidden_states, temb)
            hidden_states = rearrange(hidden_states, '(b f) c h w -> b c f h w',  f=frame)
            temb = reduce(temb, '(b f) c -> b c', 'max', f=frame)

        else:
            hidden_states = self.norm1(hidden_states)

        # passing the data in activation function [1st-step]
        hidden_states = self.nonlinearity(hidden_states)
        # pass the data in conv [1st step]
        hidden_states = self.conv1(hidden_states,
                                   is_init_image,
                                   temporal_chunk)
        
        # passing the data in normalization [2nd-step]
        if temb is not None and self.time_embedding_norm == "default":
            temb = temb[:, :, None, None, None]
            hidden_states = hidden_states + temb

        elif self.time_embedding_norm == "ada_group":
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w')
            temb = repeat(temb, 'b c -> (b f) c', f=frame)
            hidden_states = self.norm2(hidden_states, temb)
            hidden_states = rearrange(hidden_states, '(b f) c h w -> b c f h w',  f=frame)
            temb = reduce(temb, '(b f) c -> b c', 'max', f=frame)

        else:
            hidden_states = self.norm2(hidden_states)

        # passing the data in activation function [2nd-step]
        hidden_states = self.nonlinearity(hidden_states)

        # do the dropout in the data 
        hidden_states = self.dropout(hidden_states)

        # now i apply the output using conv3d
        hidden_states = self.conv2(hidden_states, 
                                   is_init_image,
                                   temporal_chunk)
        
        
        # so this the residual connection 
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor
    




class vae_Downsample(nn.Module):

    """ 
    The downsample conv.

    Args:
        channels (`int`): input channels 
        use_conv (`bool`): use to conv (default: `True`)
        out_channels (`int`): output channels (default: `None`)
    """

    def __init__(
            self,
            channels: int,
            use_conv: bool = True,
            out_channels: Optional[int] = None,
            kernel_size = 3, 
            bias=True
    ):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or self.channels
        self.use_conv = use_conv
        stride = (1, 2, 2)

        if use_conv:
            conv = vae_Conv3d(
                in_channels=self.channels,
                out_channel=self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias
            )

        else:
            assert self.channels == self.out_channels, "make sure `channels` and `out_channels` is equal!"
            conv = nn.AvgPool3d(kernel_size=stride,  # (1, 1, 1)
                                stride=stride)      # (1, 1, 1)
            

        self.conv = conv 


    def forward(self, 
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False,
            ) -> torch.FloatTensor:
        
        assert hidden_states.shape[1] == self.channels, "make sure channels are equal to data channels!"

        hidden_states = self.conv(hidden_states,
                                  is_init_image,
                                  temporal_chunk)
        
        return hidden_states
    

class vae_TemporalDownsample(nn.Module):

    """ 
    The temporal Downsample conv. 

    Args:
        channels (`int`): input channels 
        use_conv (`bool`): use to convolution (default: `False`)
        out_channels (`int`): output channels (default: `None`)
        padding (`int`): padding (default: `0`)
    """


    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            out_channels: Optional[int] = None,
            padding: int = 0,
            kernel_size = 3,
            bias = True
    ):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        
        stride = (2, 1, 1)

        if use_conv:
            self.conv = vae_Conv3d(in_channels=self.channels,
                                     out_channel=self.out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     bias=bias)

        else:
            raise NotImplementedError("make sure conv is need to Temporal Downsample!")
        

    def forward(self, 
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        
        assert hidden_states.shape[1] == self.channels, "make sure channels are equal to data channels !"

        if self.use_conv and self.training == 0:
            if hidden_states.shape[2] == 1:
                # image 
                pad = (1,) * 6

            else:
                # video 
                pad = (1, 1, 1, 1, 0, 1)

            hidden_states = nn.functional.pad(input=hidden_states,
                                              pad=pad,
                                              mode='constant',
                                              value=0)
            

        hidden_states = self.conv(hidden_states,
                                  is_init_image,
                                  temporal_chunk)
        return hidden_states
    


class vae_Upsample(nn.Module):

    """ 
    This is Upsample conv.

    Args:
        channels (`int`): input channels 
        use_conv (`bool`): conv to use (default: `False`)
        out_channels (`int`): output channels (default: `None`)
        
    """

    def __init__(self,
                 channels: int,
                 use_conv: bool = False,
                 out_channels: Optional[int] = None,
                 kernel_size = 3,
                 bias = True,
                 interpolate=False):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interploate = interpolate
        
        if interpolate:
            raise NotImplementedError("please does not do `interploate=True`")
        else:
            self.conv = vae_Conv3d(in_channels=channels,
                                out_channel=self.out_channels * 4,
                                kernel_size=kernel_size,
                                stride=1,
                                bias=bias)
            

    def forward(self,
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels, "make sure channels are equal to data channels !"

        hidden_states = self.conv(hidden_states,
                                  is_init_image,
                                  temporal_chunk)
        
        # (batch_size, channels, frames, height, width)
        hidden_states = rearrange(hidden_states,
                                'b (c p1 p2) t h w -> b c t (h p1) (w p2)',
                                p1=2, p2=2)
        
        return hidden_states
    

class vae_TemporalUpsample(nn.Module):

    """ 
    The temporal Upsample 

    Args:
        channels (`int`): input channels 
        use_conv: (`bool`): use to conv 
        out_channels: (`int`): output channels (default: `None`)

    """

    def __init__(self,
                 channels: int,
                 use_conv: bool = True,
                 out_channels: Optional[int] = None,
                 kernel_size = 3,
                 bias = True,
                 interpolate=False):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interploate = interpolate
        
        if interpolate:
            raise NotImplementedError("please does not do `interploate=True`")
        else:
            self.conv = vae_Conv3d(in_channels=channels,
                                out_channel=self.out_channels * 4,
                                kernel_size=kernel_size,
                                stride=1,
                                bias=bias)
            

    def forward(self,
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels, "make sure channels are equal to data channels !"

        hidden_states = self.conv(hidden_states,
                                  is_init_image,
                                  temporal_chunk)
        
        # (batch_size, channels, frames, height, width)
        hidden_states = rearrange(hidden_states,
                                'b (c p) t h w -> b c (t p) h w',
                                p=2)
        
        if is_init_image:
            hidden_states = hidden_states[:, :, 1:]
        
        return hidden_states
    





if __name__ == "__main__":

    in_channels = 64
    out_channels = 64
    batch_size = 2 
    time_embedding_norm = "ada_group"
    temb_channels = 64  # 128
    frame = 64
    height, width = 64, 64

    causal_resnet_block_3d = vae_Block3D(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 groups=batch_size,
                                                 time_embedding_norm=time_embedding_norm,
                                                 temb_channels=temb_channels
                                                 )
    
    
    print(causal_resnet_block_3d)

    x = torch.randn(batch_size, in_channels, frame, height, width)  
    # make sure temb_channels is multiple of in_channels 
    temb = torch.randn(batch_size, in_channels) 

    output = causal_resnet_block_3d(x, temb)
    print(output.shape)
    # --------------------------------------------------------------------------------
   
    







        








        
        