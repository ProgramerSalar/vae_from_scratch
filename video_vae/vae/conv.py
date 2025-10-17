import torch
from torch import nn 




class CausalConv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channel: int,
                 device,
                 kernel_size: int = 3,
                 stride=1):
        
        super().__init__()

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=1,
                              device=device)
        

    def forward(self, x):

        x = self.conv(x)
        return x 
    

class CausalGroupNorm(nn.Module):

    def __init__(self,
                 in_channels,
                 num_groups,
                 device,
                 eps=1e-5,
                 ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups,
                                 num_channels=in_channels,
                                 eps=eps,
                                 device=device)
        

    def forward(self, x):

        x = self.norm(x)
        return x 
    





if __name__ == "__main__":

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = CausalConv3d(in_channels=128,
                         out_channel=3,
                         device=device)
    
    
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