import torch 
from torch import nn 
import sys 
sys.path.append('/home/manish/Desktop/projects/vae_from_scratch/vae_from_scratch/video_vae/vae')


from enc import Encoder
from conv import CausalConv3d
from middle import MiddleLayer
from dec import Decoder
from gaussian import DiagonalGaussianDistribution


class CausalVAE(nn.Module):

    def __init__(self,
                 encoder_out_channels=3,):

        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder(num_groups=2)

        self.quant_conv = CausalConv3d(in_channels=2*encoder_out_channels,
                                       out_channel=2*encoder_out_channels,
                                       kernel_size=1,
                                       stride=1)
        
        self.post_quant_conv = CausalConv3d(in_channels=encoder_out_channels,
                                            out_channel=encoder_out_channels,
                                            kernel_size=1,
                                            stride=1)
        
        
        
        

    def get_last_layer(self):

        # [2, 3*2, 8, 256, 256]
        out = self.decoder.conv_out.conv.weight
        return out
    
        


    def forward(self, x):

        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)

        # generator = torch.Generator(device=torch.device('cpu'))
        # # [2, 6, 1, 32, 32] -> torch.Size([2, 3, 1, 32, 32])
        z = posterior.mode()
        dec = self.decoder(z)

        return posterior, dec





if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalVAE().to(device)
    print(model)

    learnable_param =  sum(param.numel() for param in model.parameters())
    print(f"learnable_parameters: {learnable_param / 1e6} Million")


    x = torch.randn(2, 3, 8, 256, 256).to(device)
    posterior, dec = model(x)
    print(posterior, dec.shape)

    # print('-'*40)
    # print(model.get_last_layer().shape)

