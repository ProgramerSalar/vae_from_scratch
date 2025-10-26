import torch 
from torch import nn 

from .block import CausalDeownBlock
from .middle import MiddleLayer
from .conv import CausalConv3d, CausalGroupNorm

class Encoder(nn.Module):

    def __init__(self,
                 num_groups,
                 channels=[128, 256, 512, 512],
                 in_channels=3,
                 conv_out_channels=3,
                 enc_feature=(True, True, True, False),
                 enc_frame=(True, True, True, False)):
        
        super().__init__()

        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channel=channels[0],
                                    kernel_size=3,
                                    stride=1)


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
                    enc_feature=enc_feature[i],
                    enc_frame=enc_frame[i]
                )
            )

        self.middle_layer = nn.ModuleList([
            MiddleLayer(in_channels=channels[-1],
                        num_groups=num_groups)
        ])

        self.norm_layer = CausalGroupNorm(in_channels=channels[-1],
                                          num_groups=num_groups,
                                          eps=1e-6)
        self.conv_act = nn.SiLU()

        self.conv_out = CausalConv3d(in_channels=channels[-1],
                                     out_channel=conv_out_channels*2,
                                     kernel_size=3,
                                     stride=1)
        



        

    def forward(self, x):

        x = self.conv_in(x)

        def create_module_list(module):
            def module_list(*inputs):
                return module(*inputs)
            return module_list

        for layer in self.down_encoder_blocks:
            x = torch.utils.checkpoint.checkpoint(
                create_module_list(layer),
                x,
                use_reentrant=False    
            )

        for layer in self.middle_layer:
            x = torch.utils.checkpoint.checkpoint(
                create_module_list(layer),
                x,
                use_reentrant=False
            )

        x = self.norm_layer(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x 
    



    


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder().to(device)
    print(model)

    learnable_parameters = sum(param.numel() for param in model.parameters())
    print(f"learnable_parameters >>>>> {learnable_parameters / 1e6} Million")

    x = torch.randn(2, 3, 8, 256, 256)
    out = model(x)
    print(out.shape)
        