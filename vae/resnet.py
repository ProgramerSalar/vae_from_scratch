import torch 
from torch import nn 
from typing import Optional
from einops import rearrange


from .utils import SpatialNorm, get_activation, AdaGroupNorm
from .conv import CausalGroupNorm, CausalConv3d




class CausalResnetBlock3D(nn.Module):


    def __init__(self,
                *,
                in_channels: int,
                out_channels: Optional[int] = None,
                conv_shortcut: bool = False,
                dropout: float = 0.0,
                temb_channels: int = 512,
                groups: int = 32,
                groups_out: Optional[int] = None,
                pre_norm: bool = True,
                eps: float = 1e-6,
                non_linearity: str = "swish",   # activation function 
                time_embedding_norm: str = "default",   
                output_scale_factor: float = 1.0,
                use_in_shortcut: Optional[bool] = None,
                conv_shortcut_bias: bool = True,
                conv_2d_out_channels: Optional[int] = None   
                ):
        
        
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        print(f"what is embedding_dim: {temb_channels}, out_dim: {in_channels}, num_groups: {groups}")

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(embedding_dim=temb_channels,
                                      out_dim=in_channels,
                                      num_groups=groups,
                                      eps=eps)
            
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(f_channels=in_channels,
                                     zq_channels=temb_channels)
            
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups,
                                            num_channels=in_channels,
                                            eps=eps,
                                            affine=True)
            

        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channel=out_channels,
                                  kernel_size=3,
                                  stride=1)
        

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(embedding_dim=temb_channels,
                                      out_dim=out_channels,
                                      num_groups=groups_out,
                                      eps=eps)
            
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(f_channels=out_channels,
                                     zq_channels=temb_channels)
            
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out,
                                            num_channels=out_channels,
                                            eps=eps,
                                            affine=True)
            
        
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = CausalConv3d(in_channels=out_channels,
                                  out_channel=conv_2d_out_channels,     # out_channels
                                  kernel_size=3,
                                  stride=1
                                  )
        

        self.nonlinearity = get_activation(act_fn=non_linearity)
        self.upsample = self.downsample = None
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(in_channels=in_channels,
                                              out_channel=conv_2d_out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              bias=conv_shortcut_bias)
            
        
    def forward(self,
                input_tensor: torch.FloatTensor,
                temb: torch.FloatTensor = None,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        

        hidden_states = input_tensor
        print(f"what is the hidden_states: {hidden_states.shape} and temb: {temb.shape}")

        # passing the data in normalization [1st-step]
        if self.time_embedding_norm == "ada_group" or  self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)

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
            hidden_states = hidden_states + temb

        elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)

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
        
        # the `conv_shortcut` is True
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor,
                                              is_init_image,
                                              temporal_chunk)
            
        # so this the residual connection 
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor
    




class CausalDownsample2x(nn.Module):

    def __init__(
            self,
            channels: int,
            use_conv: bool = True,
            out_channels: Optional[int] = None,
            name: str = "conv",
            kernel_size = 3, 
            bias=True
    ):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = (1, 2, 2)
        self.name = name 

        if use_conv:
            conv = CausalConv3d(
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
    

class CausalTemporalDownsample2x(nn.Module):


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
            self.conv = CausalConv3d(in_channels=self.channels,
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
    


class CausalUpsample2x(nn.Module):

    def __init__(self,
                 channels: int,
                 use_conv: bool = False,
                 out_channels: Optional[int] = None,
                 name: str = "conv",
                 kernel_size: Optional[int] = 3,
                 bias = True,
                 interpolate=False):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name 
        self.interploate = interpolate
        
        if interpolate:
            raise NotImplementedError("please does not do `interploate=True`")
        else:
            self.conv = CausalConv3d(in_channels=channels,
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
    

class CausalTemporalUpsample2x(nn.Module):

    def __init__(self,
                 channels: int,
                 use_conv: bool = True,
                 out_channels: Optional[int] = None,
                 name: str = "conv",
                 kernel_size: Optional[int] = 3,
                 bias = True,
                 interpolate=False):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name 
        self.interploate = interpolate
        
        if interpolate:
            raise NotImplementedError("please does not do `interploate=True`")
        else:
            self.conv = CausalConv3d(in_channels=channels,
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

    causal_resnet_block_3d = CausalResnetBlock3D(in_channels=8,
                                                 out_channels=8,
                                                 groups=2,
                                                #  time_embedding_norm="spatial"
                                                 )
    


    x = torch.randn(2, 8, 8, 64, 64)
    temb = torch.randn(2, 8, 1, 1, 1)

    output = causal_resnet_block_3d(x, temb)
    print(output.shape)
    







        








        
        