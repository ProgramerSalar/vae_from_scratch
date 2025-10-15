import torch 
from torch import nn 
from einops import rearrange

from .conv import CausalConv3d, CausalGroupNorm





class CausalResnetBlock3D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 eps: float = 1e-5,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32):
        
        super().__init__()
        self.scale_factor = scale_factor
        
        self.norm1 = CausalGroupNorm(in_channels=in_channels,
                                     num_groups=norm_num_groups,
                                     eps=eps)
        
        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channels=out_channels)
        
        self.norm2 = CausalGroupNorm(in_channels=out_channels,
                                     num_groups=norm_num_groups,
                                     eps=eps)
        
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(in_channels=out_channels,
                                  out_channels=out_channels)
        
        self.act_fn = nn.SiLU()

        output_channels = out_channels
        self.in_out_not_equal = None
        # [128] != [256]
        if in_channels != out_channels:
            self.in_out_not_equal = CausalConv3d(in_channels=in_channels,
                                                 out_channels=output_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 bias=True)
            
            print(f"i want to know that arch: {self.in_out_not_equal}")
            


    def forward(self, x):

        # feed-forward tensor layer 
        fd_tensor = x 

        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.conv2(x)

        # [128] != [256]
        # [256] != [512]
        if self.in_out_not_equal is not None:

            fd_tensor = self.in_out_not_equal(fd_tensor)

        x = (fd_tensor + x) / self.scale_factor

        return x 


class CausalHeightWidth2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        
        super().__init__()

        # frame, height/2, width/2
        stride = (1, 2, 2)
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride = stride,
                                 bias=True)
        
    def forward(self, x):

        x = self.conv(x)
        return x 
        

        
class CausalFrame2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        
        super().__init__()

        # frame/2, height, width
        stride = (2, 1, 1)

        

        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 bias=True)
        
    def forward(self, x):

        x = self.conv(x)
        return x 
    

class CausalUpsampleHeigthWidth(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()

        # [512] -> [4*256]
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels*4,
                                 kernel_size=3,
                                 stride=1,
                                 bias=True)
        

    def forward(self,
                x):
        
        x = self.conv(x)
        # [2, (256*2*2), 8, 256, 256] -> [2, 256, 8, (2*256), (2*256)]
        x = rearrange(x, 'b (c p1 p2) t h w -> b c t (h p1) (w p2)', 
                      p1=2, p2=2)
        
        return x 
    

class CausalTemporalUpsample2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()

        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels * 2,
                                 kernel_size=3,
                                 stride=1,
                                 bias=True)
        
    def forward(self,
                x):
        
        b, c, t, h, w = x.shape 
        x = self.conv(x)
        x = rearrange(x,
                      'b (c p) t h w -> b c (t p) h w', t=t, p=2)
        # [2, (256*2), 8, 256, 256] -> [2, 512, (8*2-1), 256, 256]
        x = x[:, :, 1:]
        return x 

