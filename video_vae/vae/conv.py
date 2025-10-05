import torch 
from torch import nn 
from typing import Union, Tuple

from middleware.gpu_processes import (
    is_context_parallel_initialized,
    get_context_parallel_rank
)




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
        self.stride = (self.stride,)*3 if isinstance(self.stride, int) else self.stride
        self.padding = padding
        self.padding = (self.padding,)*3 if isinstance(self.padding, int) else self.padding
        self.padding_mode = 'constant'

        self.time_kernel_size, self.height_kernel_size, self.width_kernel_size = self.kernel_size
        self.dilation = 1 


        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              **kwargs)
        
    def context_parallel_forward(self, x):
        
        # context parallel ranks 
        cp_rank = get_context_parallel_rank()

        if self.time_kernel_size == 3 and \
            ((cp_rank == 0 and x.shape[2] <= 2) or (cp_rank != 0 and x.shape[2] <= 1)):

            print('work in progress...')

        else:
            x = 

        
        

    def forward(self, 
                x: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False
                ) -> torch.FloatTensor:
        
        # x = [32, 3, 8, 256, 256]

        if is_context_parallel_initialized():
            return self.context_parallel_forward(x)
        
        
        




