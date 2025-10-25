import torch 
from torch import nn 

from block import causalUpperBlock
from conv import CausalConv3d, CausalGroupNorm
from middle import MiddleLayer


class Decoder(nn.Module):

    def __init__(self,
                 num_groups,
                 in_channels=3,
                 out_channels=3,
                 channels = [128, 256, 512, 512],
                 dec_feature = (True, True, True, False),
                 dec_frame = (True, True, True, False)):
        
        super().__init__()

        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channel=channels[-1],
                                    kernel_size=3,
                                    stride=1)

        # [2, 512, 1, 32, 32] -> [2, 512, 1, 32, 32]
        self.middle_layer = nn.ModuleList([
            MiddleLayer(in_channels=channels[-1],
                        num_groups=num_groups)
        ])



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
                    dec_feature=dec_feature[i],
                    dec_frame=dec_frame[i]
                )
            )

        self.conv_norm_out = CausalGroupNorm(in_channels=channels[0],
                                             num_groups=num_groups,
                                             eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(in_channels=channels[0],
                                     out_channel=out_channels,
                                     kernel_size=3,
                                     stride=1)
        


        

        
        

    def forward(self, x):

        x = self.conv_in(x)
    

        def create_module_list(module):
            def module_list(*inputs):
                return module(*inputs)
            return module_list

        for layer in self.middle_layer:
            # x = layer(x)
            x = torch.utils.checkpoint.checkpoint(
                create_module_list(layer),
                x,
                use_reentrant=False
            )

        for layer in self.upper_decoder_blocks:
            # x = layer(x)
            x = torch.utils.checkpoint.checkpoint(
                create_module_list(layer),
                x,
                use_reentrant=False
            )

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
    
        return x 
    



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Decoder().to(device)
    print(model)

    x = torch.randn(2, 3, 1, 32, 32)
    out = model(x)
    print(out.shape)


        
        