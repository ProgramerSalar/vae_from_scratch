import torch
from torch import nn 
from timm.models.layers import trunc_normal_



class CausalConv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channel: int,
                 kernel_size: int = 3,
                 stride=1,
                 padding=1,
                 **kwargs):
        
        super().__init__()

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              **kwargs)
        
        self.apply(self._init_weights)
        
    # custom init weights
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def forward(self, x):

        x = self.conv(x)
        return x 
    

class CausalGroupNorm(nn.Module):

    def __init__(self,
                 in_channels,
                 num_groups,
                 eps=1e-5,
                 affine=True
                 ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups,
                                 num_channels=in_channels,
                                 eps=eps,
                                 affine=affine
                                 )
        

    def forward(self, x):

        x = self.norm(x)
        return x 
    





if __name__ == "__main__":

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = CausalConv3d(in_channels=128,
                         out_channel=3,
                        )
    
    
    x = torch.randn(2, 128, 8, 256, 256)

    out = model(x)
    print(out.shape)
    # print(model.conv.weight.shape)
    # print(model.conv.bias.shape)

    # norm_data = torch.randn(2, 128, 8, 256, 256)
    # normalization = CausalGroupNorm(in_channels=128, 
    #                                 num_groups=2,
    #                                 device=device)
    
    # norm_out = normalization(norm_data)
    # print(norm_out)

    # print(normalization.norm.weight)
    # print(normalization.norm.bias)