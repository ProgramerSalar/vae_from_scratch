import torch 
from torch import nn 

from enc import Encoder
from conv import CausalConv3d
from middle import MiddleLayer
from dec import Decoder

class CausalVAE(nn.Module):

    def __init__(self,
                 device,
                 ):

        super().__init__()
        self.conv1 = CausalConv3d(in_channels=3,
                                 out_channel=128,
                                 device=device,
                                 kernel_size=3)
        self.encoder = Encoder(device=device)
        self.middle = MiddleLayer(in_channels=512,
                                  num_groups=2,
                                  device=device)
        
        self.conv2 = CausalConv3d(in_channels=512,
                                 out_channel=512,
                                 device=device,
                                 kernel_size=3)
        
        self.decoder = Decoder(device=device,
                               num_groups=2,
                               )
        
        self.conv3 = CausalConv3d(in_channels=128,
                                 out_channel=3,
                                 device=device,
                                 kernel_size=3)
        


    def forward(self, x):

        x = self.conv1(x)

        def create_module_function(module):
            def module_function(*inputs):
                return module(*inputs)
            return module_function
     
        x = torch.utils.checkpoint.checkpoint(
            create_module_function(self.encoder),
            x
        )
        
        x = torch.utils.checkpoint.checkpoint(
            create_module_function(self.middle),
            x
        )

        x = torch.utils.checkpoint.checkpoint(
            create_module_function(self.decoder),
            x
        )
     
        x = self.conv3(x)

        return x 




if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalVAE(device=device)
    print(model)

    learnable_param =  sum(param.numel() for param in model.parameters())
    print(f"learnable_parameters: {learnable_param / 1e6} Million")


    x = torch.randn(2, 3, 8, 256, 256)
    out = model(x)
    print(out.shape)
