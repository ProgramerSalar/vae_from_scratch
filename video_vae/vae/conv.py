import torch 
from torch import nn 
from typing import Union, Tuple
from collections import deque
from timm.layers.weight_init import trunc_normal_
from torch import Tensor
from einops import rearrange

from middleware.gpu_processes import (
    is_context_parallel_initialized,
    get_context_parallel_rank
)

from middleware.single_gpu_cp_ops import context_parallel_pass_from_previous_rank




class CausalConv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 stride: Union[int, Tuple[int, int, int]] = 1, 
                 padding: Union[int, Tuple[int, int, int]] = 1,
                 **kwargs):
        
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_size = (self.kernel_size,)*3 if isinstance(self.kernel_size, int) else self.kernel_size
        self.stride = stride
        self.stride = (self.stride, 1, 1) if isinstance(self.stride, int) else self.stride
        self.padding = padding
        self.padding = (self.padding,)*3 if isinstance(self.padding, int) else self.padding
        self.padding_mode = 'constant'

        self.time_kernel_size, self.height_kernel_size, self.width_kernel_size = self.kernel_size
        self.dilation = 1 

        ## <-- padding --> ##
        width_pad = self.width_kernel_size // 2
        height_pad = self.height_kernel_size // 2
        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        self.time_pad = self.dilation * (self.time_kernel_size -1)
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, self.time_pad, 0)

        self.temporal_stride = self.stride[0]


        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=0,
                              dilation=self.dilation,
                              **kwargs)
        self.cache_first_feat = deque()
        
    def context_parallel_forward(self, x):
        
        # context parallel ranks 
        cp_rank = get_context_parallel_rank()

        if self.time_kernel_size == 3 and \
            ((cp_rank == 0 and x.shape[2] <= 2) or (cp_rank != 0 and x.shape[2] <= 1)):

            print('work in progress...')

        else:
            # [2, 3, 8, 256, 256] -> [2, 3, 10, 256, 256]
            x = context_parallel_pass_from_previous_rank(input_=x,
                                                         dim=2,
                                                         kernel_size=self.time_kernel_size)
            
        # [2, 3, 10, 256, 256] -> [2, 3, 10, 258, 258]
        x = torch.nn.functional.pad(input=x,
                                    pad=self.time_uncausal_padding, 
                                    mode=self.padding_mode)
        
        
        # [2, 3, 10, 258, 258] -> [2, 3, 8, 256, 256]
        x = self.conv(x)
        return x 
    

    def _init_weights(self, m):

        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(tensor=m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(tensor=m.bias, val=0)
            nn.init.constant_(tensor=m.weight, val=1.0)


        
        

    def forward(self, 
                x: torch.FloatTensor,
                ) -> torch.FloatTensor:
        
        
        # [2, 3, 8, 256, 256] -> [2, 3, 8, 256, 256]
        if is_context_parallel_initialized():
            return self.context_parallel_forward(x)
        
        padding_mode = self.padding_mode if self.time_pad < x.shape[2] else 'constant'

        # [2, 3, 8, 256, 256] -> [2, 3, 8, 256, 256]
        x = torch.nn.functional.pad(input=x,
                                    pad=self.time_causal_padding, # (1, 1, 1, 1, 2, 0)
                                    mode=padding_mode)
        
        # [2, 3, 8, 256, 256] -> [2, 3, 8, 256, 256]
        x = self.conv(x)

        
        
        return x 
    


class CausalGroupNorm(nn.Module):

    def __init__(self,
                 in_channels:int,
                 num_groups:int = 32, 
                 eps: float= 1e-5):
        
        """ 
            This is custom normalization.
            
            in_channels (`int`): number of channels expected in input
            num_groups (`int`): number of groups to separate the channels into 
            eps (`float`): a value added to the denominator for numerical stability. Default: 1e-5
        """
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=num_groups,
                                       num_channels=in_channels,
                                       eps=eps)

    def forward(self, 
                x: torch.FloatTensor) -> torch.FloatTensor:
        

        b, c, t, h, w = x.shape
        # x = x.view(b*t, c, h, w)
        x = rearrange(x, 
                      'b c t h w -> (b t) c h w', t=t)
        x = self.group_norm(x)

        # x = x.view(b, c, t, h, w)
        x = rearrange(x,
                      '(b t) c h w -> b c t h w', t=t)

        return x 