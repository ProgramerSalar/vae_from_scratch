import torch 
from torch import nn 
from einops import rearrange

from resnet import CausalResnet3d
from attention import AttentionLayer


class MiddleLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 num_groups,
                 device):
        super().__init__()

        

        resnets = [
            CausalResnet3d(
                in_channels=in_channels,
                out_channels=in_channels,
                num_groups=num_groups,
                device=device
            )
        ]

        
        attention = AttentionLayer(in_channels=in_channels,
                                        num_groups=num_groups)
    
    
        resnets.append(
                CausalResnet3d(
                in_channels=in_channels,
                out_channels=in_channels,
                num_groups=num_groups,
                device=device
            )
        )

        self.attentions = nn.ModuleList([attention])
        self.resnets = nn.ModuleList(resnets)



    def forward(self, x):
        
        b, c, t, h, w = x.shape

        x = self.resnets[0](x)
        x = rearrange(x, 
                      'b c t h w -> (b t) c h w')
        
        # attention layer 
        for layer in self.attentions:
            x = layer(x)

        x = rearrange(x,
                      '(b t) c h w -> b c t h w', b=b, t=t)
        
        x = self.resnets[1](x)

        return x             

        
    



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiddleLayer(in_channels=512,
                        num_groups=2,
                        device=device)
    # print(model)

    x = torch.randn(2, 512, 1, 32, 32)
    out = model(x)
    print(out.shape)