import torch 
from torch import nn 
from typing import Union, Tuple
from collections import deque




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



    def forward(self, 
                x):
        

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





