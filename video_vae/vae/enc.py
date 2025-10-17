import torch 
from torch import nn 

from block import CausalDeownBlock

class Encoder(nn.Module):

    def __init__(self,
                 device,
                 channels=[128, 256, 512, 512],
                 num_groups=2,
                 enc_feature=(True, True, True, False),
                 enc_frame=(True, True, True, False)):
        
        super().__init__()


        output_channels = channels[0]
        self.down_encoder_blocks = nn.ModuleList([])
        for i in range(4):

            in_channels = output_channels
            output_channels = channels[i]

            self.down_encoder_blocks.append(
                CausalDeownBlock(
                    in_channels=in_channels,
                    out_channels=output_channels,
                    num_groups=num_groups,
                    device=device,
                    enc_feature=enc_feature[i],
                    enc_frame=enc_frame[i]
                )
            )

    def forward(self, x):

        for layer in self.down_encoder_blocks:
            x = layer(x)
            
        print(f"Encoder shape of data: {x.shape}")
        return x 
    



    


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(device=device)
    print(model)

    learnable_parameters = sum(param.numel() for param in model.parameters())
    print(f"learnable_parameters >>>>> {learnable_parameters / 1e6} Million")

    x = torch.randn(2, 128, 8, 256, 256)
    out = model(x)
    print(out.shape)
        