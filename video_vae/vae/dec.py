import torch 
from torch import nn 

from block import causalUpperBlock

class Decoder(nn.Module):

    def __init__(self,
                 device,
                 num_groups=2,
                 channels = [128, 256, 512, 512],
                 dec_feature = (True, True, True, False),
                 dec_frame = (True, True, True, False)):
        
        super().__init__()

        self.upper_decoder_blocks = nn.ModuleList([])

        reversed_channels = list(reversed(channels))
        
        output_channels = reversed_channels[0]
        for i in range(4):
            input_channels = output_channels
            output_channels = reversed_channels[i]

            self.upper_decoder_blocks.append(
                causalUpperBlock(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_groups=num_groups,
                    device=device,
                    dec_feature=dec_feature[i],
                    dec_frame=dec_frame[i]
                )
            )

    def forward(self, x):


        for layer in self.upper_decoder_blocks:
            x = layer(x)
            print(f"shape of data: {x.shape}")
            
        return x 
    



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Decoder(device=device)
    print(model)

    x = torch.randn(2, 512, 1, 16, 16)
    out = model(x)
    print(out.shape)


        
        