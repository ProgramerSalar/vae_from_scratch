import torch 
from torch import nn 
from typing import Union, Tuple
from collections import deque
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from .utils.utils import trunc_normal_
from utils import (
    get_context_parallel_rank, 
    is_context_parallel_intialized
    )
from .context_parallel_ops import cp_pass_from_previous_rank


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0 

def is_odd(n):
    return not divisible_by(n, 2)

class CausalConv3d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = 1,
            pad_mode: str = 'constant',
            **kwargs
    ):
        
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = cast_tuple(kernel_size, 3)
        print(f"what is the kernel_size : {kernel_size}")

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop('dilation', 1)
        self.pad_mode = pad_mode

        if isinstance(stride, int):
            stride = (stride, 1, 1)

        time_pad = dilation * (time_kernel_size - 1)
        height_pad = height_kernel_size // 2 
        width_pad = width_kernel_size // 2 

        self.temporal_stride = stride[0]
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              dilation=dilation,
                              **kwargs)
        
        self.cache_front_feat = deque()


    def __clear_context_parallel_cache(self):
        del self.cache_front_feat
        self.cache_front_feat = deque()


    def __init_weights(self, m):

        """
            m: (tpically a layer in a neural network) = nn.Linear, nn.Conv2d, nn.Conv3d, nn.LayerNorm or nn.GroupNorm
        """

        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(tensor=m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def context_parallel_forward(self, x):

        """ 
            context parallelism is technique used in distributed training of deep learning models
            where different parts of the input context (in this case, likely the temporal dimension of a video or sequence)
            are processed on different devices or ranks.
        """

        cp_rank = get_context_parallel_rank()

        if self.time_kernel_size == 3 and ((cp_rank == 0 and x.shape[2] <= 2) or cp_rank != 0 and x.shape[2] <= 1):

            # This code is only for training 8 frames per GPU (except for cp_rank=0, 9 frames) with context parallel 
            # if you do not have enought GPU memory, you can set the total frames = 8 * CONTEXT_SIZE + 1, enable each GPU 
            # only forward 8 frames during training

            x = cp_pass_from_previous_rank(input_=x,
                                           dim=2,
                                           kernel_size=2)
            
            
            
            trans_x = cp_pass_from_previous_rank(input_=x[:, :, :-1],
                                                 dim=2,
                                                 kernel_size=2)
            x = torch.cat([trans_x, x[:, :, -1:]], dim=2)


        else:
            x = cp_pass_from_previous_rank(input_=x,
                                           dim=2,
                                           kernel_size=self.time_kernel_size)
            
            
            

        x = F.pad(x, self.time_uncausal_padding, mode='constant')

        if cp_rank != 0:
            if self.temporal_stride == 2 and self.time_kernel_size == 3:
                x = x[:, :, 1:]

            
        x = self.conv(x)
        return x 
    


    def forward(self, 
                x, 
                is_init_image=True,
                temporal_chunk=False):
        

        if is_context_parallel_intialized():
            return self.context_parallel_forward(x)
        
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        if not temporal_chunk:
            x = F.pad(x, self.time_causal_padding, mode=pad_mode)

        else:
            assert not self.training, "The feature cache should not be used in training"
            if is_init_image:
                # Encode the first chunk 
                x = F.pad(x, self.time_causal_padding, mode=pad_mode)
                self.__clear_context_parallel_cache()
                self.cache_front_feat.append(x[:, :, -2:].clone().detach())

            else:
                x = F.pad(x, self.time_uncausal_padding, mode=pad_mode)
                video_format_context = self.cache_front_feat.pop()
                self.__clear_context_parallel_cache()


                if self.temporal_stride == 1 and self.time_kernel_size == 3:
                    x = torch.cat([video_format_context, x], dim=2)
                elif self.temporal_stride == 2 and self.time_kernel_size == 3:
                    x = torch.cat([video_format_context[:, :, -1:], x], dim=2)


                self.cache_front_feat.append(x[:, :, -2:].clone().detach())

        print(f"what is the input shape {x.shape} and what is the dtype: {x.dtype} \n what is the weight-dtype of conv: {self.conv.} and bias-dtype: {self.conv.bias.dtype}")
        x = self.conv(x)
        return x 
    



class CausalGroupNorm(nn.GroupNorm):

    def forward(self, 
                x: Tensor) -> Tensor:
        
        t = x.shape[2]
        x = rearrange(tensor=x, 
                      pattern='b c t h w -> (b t) c h w')
        x = super().forward(x)
        x = rearrange(tensor=x, 
                      pattern='(b t) c h w -> b c t h w', 
                      t=t)
        
        return x 
        

    




    



if __name__ == "__main__":

    # causasl_conv_3d = CausalConv3d(in_channels=3,
    #                                out_channels=3,
    #                                kernel_size=3,
    #                                stride=1)
    # # print(causasl_conv_3d)

    # x = torch.randn(2, 3, 8, 64, 64)
    # output = causasl_conv_3d(x)
    # print(output.shape)

    pass 
    