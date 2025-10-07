import torch 
from torch import nn 
from typing import Union, Tuple
from collections import deque

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

        time_pad = self.dilation * (self.time_kernel_size -1)
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.temporal_stride = stride[0]


        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
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
            x = context_parallel_pass_from_previous_rank(input_=x,
                                                         dim=2,
                                                         kernel_size=self.time_kernel_size)
            
        
        x = torch.nn.functional.pad(input=x,
                                    pad=self.time_uncausal_padding, 
                                    mode=self.padding_mode)
        
        if cp_rank != 0:
            if self.temporal_stride == 2 and self.time_kernel_size == 3:
                x = x[:, :, 1:]

        x = self.conv(x)
        return x 

        
        

    def forward(self, 
                x: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False
                ) -> torch.FloatTensor:
        
        # x = [32, 3, 8, 256, 256]

        if is_context_parallel_initialized():
            return self.context_parallel_forward(x)
        
        padding_mode = self.padding_mode if self.time_pad < x.shape[2] else 'constant'

        if not temporal_chunk:
            x = torch.nn.functional.pad(input=x,
                                        pad=self.time_causal_padding,
                                        mode=padding_mode)
            
        else:
            assert not self.training, "The feature cache should not be used in training."
            if is_init_image:
                # encode first chunk 
                x = torch.nn.functional.pad(input=x,
                                            pad=self.time_causal_padding,
                                            mode=padding_mode)
                self._clear_context_parallel_cache()
                self.cache_first_feat.append(x[:, :, -2:].clone().detach())

            else:
                x = torch.nn.functional.pad(input=x,
                                            pad=self.time_uncausal_padding,
                                            mode=padding_mode)
                
        
        
        




