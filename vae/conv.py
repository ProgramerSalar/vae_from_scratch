import torch 
from torch import nn 
from typing import Union, Tuple
from collections import deque

from vae_utils.context_parallel import (
    is_context_parallel_initialized,
    get_context_parallel_rank,
    cp_pass_from_previous_rank
)
# from torchrl.modules import TruncatedNormal
from timm.layers.weight_init import trunc_normal_


class CausalConv3d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channel,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = 1,
            pad_mode: str = 'constant',
            **kwargs
    ):
        
        """
            This conv is based on this paper: https://arxiv.org/pdf/1412.0767

            Args:
                in_channels (int): input channels of the data 
                out_channels (int): output channels of the data 
                kernel_size (int, tuple): size of the kernel like: (3, 3, 3)
                stride (int, tuple): size of the stride like: (1, 1, 1)
                dilation (int): what is the dilation of the kernel default is `1`
        """

        
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3

        assert not kernel_size[0] % 2 == 0, f"make sure kernel_size={kernel_size} is odd number"
        
            
        self.time_kernel_size, \
        self.height_kernel_size, \
        self.width_kernel_size = kernel_size
        self.dilation = kwargs.pop('dilation', 1)
        self.pad_mode = pad_mode

        if isinstance(stride, int):
            stride = stride if isinstance(stride, tuple) else (stride,) * 3
        
        # padding of conv
        self.time_pad = self.dilation * (self.time_kernel_size - 1)  # 1 * (3 - 1) => 2 
        self.height_pad = self.height_kernel_size // 2               # 3 / 2 = 1..
        self.width_pad = self.width_kernel_size // 2 


        self.temporal_stride = stride[0]
        #  5D padding in 3d conv arch. (1, 1, 1, 1, 2, 0)
        self.time_causal_padding = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        self.time_uncausal_padding = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, 0, 0)


        
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              dilation=self.dilation,
                              **kwargs)
        
        self.cache_front_feat = deque()


    def context_parallel_initialized(self, x):

        # how many gpu work in pair wise
        cp_rank = get_context_parallel_rank()

        if self.time_kernel_size == 3 and \
            ((cp_rank == 0 and x.shape[2] <= 2) or (cp_rank != 0 and x.shape[2] <= 1)):

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
            
        x = torch.nn.functional.pad(input=x,
                                    pad=self.time_uncausal_padding,
                                    mode='constant')
        
        if cp_rank != 0:
            if self.temporal_stride == 2 and self.time_kernel_size == 3:
                x = x[:, :, 1:]

        x = self.conv(x)
        return x 



    def _clear_context_parallel_cache(self):
        del self.cache_front_feat
        self.cache_front_feat = deque()


    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


            



    def forward(self, 
                x,
                is_init_image=True,
                temporal_chunk=False):
        

        if is_context_parallel_initialized():
            return self.context_parallel_initialized(x)
        
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        if not temporal_chunk:
            x = torch.nn.functional.pad(input=x,
                                        pad=self.time_causal_padding,
                                        mode=pad_mode)
            
        else:
            assert not self.training, "The feature cache should not be used in training."
            
            if is_init_image:
                # Encode the first chunk 
                x = torch.nn.functional.pad(input=x,
                                            pad=self.time_causal_padding,
                                            mode=pad_mode)
                self._clear_context_parallel_cache()
                self.cache_front_feat.append(x[:, :, -2].clone().detach())

            else:
                x = torch.nn.functional.pad(input=x,
                                            pad=self.time_uncausal_padding,
                                            mode=pad_mode)
                video_front_context = self.cache_front_feat.pop()
                self._clear_context_parallel_cache()


                if self.temporal_stride == 1 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context, x], dim=2)

                elif self.temporal_stride == 2 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context[:, :, -1:], x], dim=2)

                
                self.cache_front_feat.append(x[:, :, -2:].clone().detach())



        

        ### calculation of input -> output  ### 
        ## frame input(8) -> output(6)
        # (input_length - kernel_size) + 1 => (8 - 3) + 1 => 6
        ## height,width (225, 225) -> (223, 223)
        # (input_length - kernel_size) + 1 => (225 - 3) + 1 => 223
        x = self.conv(x)
        return x 
    


    



if __name__ == "__main__":
    causal_conv3d = CausalConv3d(in_channels=3,
                                 out_channel=3,
                                 kernel_size=3,
                                 stride=1,
                                 )
    
    print(causal_conv3d)
    x = torch.randn(32, 3, 8, 225, 225)
    output = causal_conv3d(x)
    print(output.shape) # torch.Size([32, 3, 6, 223, 223])





