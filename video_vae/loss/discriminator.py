import torch 
from torch import nn 



class NumberLayerDiscriminator(nn.Module):

    """ 
        To learn to distinguish between real and fake image patches to provide a rich, 
        spatially-aware gradient signal to train the generator.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=64):
        
        super().__init__()

        sequence = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1
                      ),
            nn.LeakyReLU(0.2, inplace=False)
        ]

        self.num_layers = 4
        self.channels = [64, 128, 256, 512]

        out_channels = self.channels[0]
        for i in range(1, self.num_layers):
            
            input_channels = out_channels
            out_channels = self.channels[i]

            # print(f"index: {i} input_channels: {input_channels}, out_channels: {out_channels} ")
            
            sequence += (
                nn.Conv2d(in_channels=input_channels,
                          out_channels=out_channels,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          ),
                nn.InstanceNorm2d(num_features=out_channels),
                nn.LeakyReLU(0.2, inplace=False)
            )
        sequence += [
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=4,
                      stride=1,
                      padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=False)
        ]

        sequence += [
            nn.Conv2d(in_channels=512,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=1)
        ]

        self.sequence = nn.ModuleList(sequence)


    def forward(self, x):

        for layer in self.sequence:
            x = layer(x)

        return x 



class NumberLayerDiscriminator3d(nn.Module):

    """ 
        To learn to distinguish between real and fake image patches to provide a rich, 
        spatially-aware gradient signal to train the generator.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=64):
        
        super().__init__()

        sequence = [
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1
                      ),
            nn.LeakyReLU(0.2, inplace=False)
        ]

        self.num_layers = 4
        self.channels = [64, 128, 256, 512]

        out_channels = self.channels[0]
        for i in range(1, self.num_layers):
            
            input_channels = out_channels
            out_channels = self.channels[i]

            # print(f"index: {i} input_channels: {input_channels}, out_channels: {out_channels} ")
            
            sequence += (
                nn.Conv3d(in_channels=input_channels,
                          out_channels=out_channels,
                          kernel_size=(4, 4, 4),
                          stride=(1, 2, 2),
                          padding=1,
                          ),
                nn.InstanceNorm3d(num_features=out_channels),
                nn.LeakyReLU(0.2, inplace=False)
            )
        sequence += [
            nn.Conv3d(in_channels=512,
                      out_channels=512,
                      kernel_size=4,
                      stride=1,
                      padding=1),
            nn.InstanceNorm3d(num_features=512),
            nn.LeakyReLU(0.2, inplace=False)
        ]

        sequence += [
            nn.Conv3d(in_channels=512,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=1)
        ]

        self.sequence = nn.ModuleList(sequence)


    def forward(self, x):

        for layer in self.sequence:
            # print(f" layer: {layer}")
            x = layer(x)
            if isinstance(layer, nn.Conv3d):
                print(f"what is the shape of data: >>>> {x.shape}")

        return x 


if __name__ == "__main__":

    # model = NumberLayerDiscriminator(in_channels=3,
    #                                  out_channels=64)
    
    # print(model)
    # x = torch.randn(2, 3, 256, 256)
    # # x = torch.randn(2, 1, 1, 1)
    # out = model(x)
    # print(out.shape)    # [2, 1, 14, 14]
    # ---------------------------------------------------------

    model = NumberLayerDiscriminator3d(in_channels=3,
                                       out_channels=64)
    print(model)
    x = torch.randn(8, 3, 16, 64, 64)
    out = model(x)
    print(out.shape)