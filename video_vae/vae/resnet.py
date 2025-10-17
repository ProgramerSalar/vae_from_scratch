import torch 
from torch import nn 
from einops import rearrange

from conv import CausalConv3d, CausalGroupNorm

class CausalResnet3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int,
                 device):
        
        super().__init__()
        

        self.norm1 = CausalGroupNorm(in_channels=in_channels,
                                     num_groups=num_groups,
                                     device=device,
                                     )

        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channel=out_channels,
                                  device=device)
        
        self.norm2 = CausalGroupNorm(in_channels=out_channels,
                                     num_groups=num_groups,
                                     device=device)
        
        # 128 -> 256
        output_channels = out_channels
        self.increment_conv = None
        if in_channels != out_channels:
            self.increment_conv = CausalConv3d(in_channels=in_channels,
                                               out_channel=output_channels,
                                               device=device,
                                               kernel_size=3)
        
    def forward(self, x):

        sample = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.norm2(x)

        # [2, 128, 8, 256, 256]
        if self.increment_conv is not None:
            x = self.increment_conv(sample)

        return x 
    

class DecreaseFeature(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 device,
                 ):
        super().__init__()
        

        stride = (1, 2, 2)
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channel=out_channels,
                                 device=device,
                                 stride=stride)
        
    def forward(self, x):

        x = self.conv(x)
        return x 
        

        
class DecreaseFrame(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 device,
                 ):
        super().__init__()
        

        stride = (2, 1, 1)
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channel=out_channels,
                                 device=device,
                                 stride=stride)
        
    def forward(self, x):

        x = self.conv(x)
        return x 



class IncreaseFeature(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 device):
        
        super().__init__()

        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channel=out_channels * 4,
                                 kernel_size=3,
                                 stride=1,
                                 device=device)

        
    def forward(self, x):

        # [2, 512, 1, 16, 16] -> [2, 512*4, 1, 16, 16]
        x = self.conv(x)

        x = rearrange(x, 
                      'b (c p1 p2) t h w -> b c t (h p1) (w p2)', p1=2, p2=2)
        
        return x 
    

class IncreaseFrame(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 device):
        
        super().__init__()
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channel=out_channels * 2,
                                 kernel_size=3,
                                 device=device)
        

    def forward(self, x):

        t = x.shape[2]
        x = self.conv(x)
        x = rearrange(x,
                      'b (c p) t h w -> b c (t p) h w', t=t, p=2)
        
        return x 



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CausalResnet3d(in_channels=128,
    #                        out_channels=256,
    #                        num_groups=2,
    #                        device=device)
    
    # learnable_parameters = sum(parm.numel() for parm in model.parameters())
    # print(f"learnable_parameters -> {learnable_parameters / 100000} Lakh")
    
    # print(model)
    # print(model.norm1.norm.weight)
    # print(model.norm1.norm.bias)
    # ---
    # model = DecreaseFrame(in_channels=128,
    #                         out_channels=128,
    #                         device=device)
    
    # x = torch.randn(2, 128, 8, 256, 256)
    # out = model(x)
    # print(out.shape)
    # -----

    # model = IncreaseFeature(in_channels=512, 
    #                         out_channels=512,
    #                         device=device)
    # x = torch.randn(2, 512, 1, 16, 16)
    # out = model(x)
    # print(out.shape)
    # ---

    model = IncreaseFrame(in_channels=512, 
                          out_channels=512,
                          device=device)
    
    x = torch.randn(2, 512, 1, 16, 16)
    out = model(x)
    print(out.shape)
        
