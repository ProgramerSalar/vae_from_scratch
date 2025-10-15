import torch 
from torch import nn 
import functools


def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class NLayerDiscriminator(nn.Module):

    def __init__(self,
                 input_nc=3,
                 ndf=64,
                 n_layers=4):
        
        super().__init__()
        
        norm_layer = nn.InstanceNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d


        sequence = [nn.Conv2d(in_channels=input_nc,
                             out_channels=ndf,
                             kernel_size=4,
                             stride=2,
                             padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                    ]
        
        
        output_channels = 1
        for n in range(1, n_layers):
            input_channels = output_channels
            output_channels = min(2 **n, 8) # 2**1->2, 2**2->4, 2**3->8

            # [64] -> [128], [1*64] -> [2*64]
            # [128] -> [256], [2*64] -> [4*64]
            # [256] -> [512], [4*64] -> [8*64]

            sequence += [
                nn.Conv2d(in_channels=input_channels * ndf,     
                          out_channels=output_channels * ndf,   
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=use_bias),
                norm_layer(ndf * output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
            

        # [512] -> [512], [8*64] -> [2**4*64]
        input_channels_prev = output_channels
        output_channels = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(in_channels=ndf * input_channels_prev,
                      out_channels=ndf * output_channels,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      bias=use_bias
                      ),
            norm_layer(ndf * output_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]

        sequence += [
            nn.Conv2d(in_channels=ndf * output_channels,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=1)
        ]
       
        self.main = nn.Sequential(*sequence)

        

    def forward(self, x):
        return self.main(x)
    

class NLayerDiscriminator3D(nn.Module):

    def __init__(self,
                 input_nc=3,
                 ndf=64,
                 n_layers=3):
        
        super().__init__()
        
        norm_layer = nn.InstanceNorm3d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d


        sequence = [nn.Conv3d(in_channels=input_nc,
                             out_channels=ndf,
                             kernel_size=4,
                             stride=2,
                             padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                    ]
        
        
        output_channels = 1
        for n in range(1, n_layers):
            input_channels = output_channels
            output_channels = min(2 **n, 8) # 2**1->2, 2**2->4, 2**3->8

            # [64] -> [128], [1*64] -> [2*64]
            # [128] -> [256], [2*64] -> [4*64]
            # [256] -> [512], [4*64] -> [8*64]

            sequence += [
                nn.Conv3d(in_channels=input_channels * ndf,     
                          out_channels=output_channels * ndf,   
                          kernel_size=(4, 4, 4),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=use_bias),
                norm_layer(ndf * output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
            

        # [512] -> [512], [8*64] -> [2**4*64]
        input_channels_prev = output_channels
        output_channels = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(in_channels=ndf * input_channels_prev,
                      out_channels=ndf * output_channels,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      bias=use_bias
                      ),
            norm_layer(ndf * output_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]

        sequence += [
            nn.Conv3d(in_channels=ndf * output_channels,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=1)
        ]
       
        self.main = nn.Sequential(*sequence)

        

    def forward(self, x):
        return self.main(x)
    


if __name__ == "__main__":

    out = NLayerDiscriminator3D()
    x = torch.randn(8, 3, 16, 64, 64)
    out = out(x)
    print(out.shape)

        


        



