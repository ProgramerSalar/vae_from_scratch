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




class NumberLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=4):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        

        # norm_layer = nn.BatchNorm2d
        norm_layer = nn.InstanceNorm2d

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    
    

class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        
        # if not use_actnorm:
        #     norm_layer = nn.BatchNorm3d
        # else:
        #     raise NotImplementedError("Not implemented.")
        
        norm_layer = nn.InstanceNorm3d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw), stride=(1,2,2), padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw), stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    

# class NumberLayerDiscriminator(nn.Module):

#     """ 
#         To learn to distinguish between real and fake image patches to provide a rich, 
#         spatially-aware gradient signal to train the generator.
#     """

#     def __init__(self,
#                  in_channels=3,
#                  out_channels=64):
        
#         super().__init__()

#         sequence = [
#             nn.Conv2d(in_channels=in_channels,
#                       out_channels=out_channels,
#                       kernel_size=4,
#                       stride=2,
#                       padding=1
#                       ),
#             nn.LeakyReLU(0.2, inplace=False)
#         ]

#         self.num_layers = 4
#         self.channels = [64, 128, 256, 512]

#         out_channels = self.channels[0]
#         for i in range(1, self.num_layers):
            
#             input_channels = out_channels
#             out_channels = self.channels[i]

#             # print(f"index: {i} input_channels: {input_channels}, out_channels: {out_channels} ")
            
#             sequence += (
#                 nn.Conv2d(in_channels=input_channels,
#                           out_channels=out_channels,
#                           kernel_size=4,
#                           stride=2,
#                           padding=1,
#                           ),
#                 nn.InstanceNorm2d(num_features=out_channels),
#                 nn.LeakyReLU(0.2, inplace=False)
#             )
#         sequence += [
#             nn.Conv2d(in_channels=512,
#                       out_channels=512,
#                       kernel_size=4,
#                       stride=1,
#                       padding=1),
#             nn.InstanceNorm2d(num_features=512),
#             nn.LeakyReLU(0.2, inplace=False)
#         ]

#         sequence += [
#             nn.Conv2d(in_channels=512,
#                       out_channels=1,
#                       kernel_size=4,
#                       stride=1,
#                       padding=1)
#         ]

#         self.sequence = nn.ModuleList(sequence)


#     def forward(self, x):

#         for layer in self.sequence:
#             x = layer(x)

#         return x 





# class NumberLayerDiscriminator3d(nn.Module):

#     """ 
#         To learn to distinguish between real and fake image patches to provide a rich, 
#         spatially-aware gradient signal to train the generator.
#     """

#     def __init__(self,
#                  in_channels=3,
#                  out_channels=64):
        
#         super().__init__()

#         sequence = [
#             nn.Conv3d(in_channels=in_channels,
#                       out_channels=out_channels,
#                       kernel_size=4,
#                       stride=2,
#                       padding=1
#                       ),
#             nn.LeakyReLU(0.2, inplace=False)
#         ]

#         self.num_layers = 4
#         self.channels = [64, 128, 256, 512]

#         out_channels = self.channels[0]
#         for i in range(1, self.num_layers):
            
#             input_channels = out_channels
#             out_channels = self.channels[i]

#             # print(f"index: {i} input_channels: {input_channels}, out_channels: {out_channels} ")
            
#             sequence += (
#                 nn.Conv3d(in_channels=input_channels,
#                           out_channels=out_channels,
#                           kernel_size=(4, 4, 4),
#                           stride=(1, 2, 2),
#                           padding=1,
#                           ),
#                 nn.InstanceNorm3d(num_features=out_channels),
#                 nn.LeakyReLU(0.2, inplace=False)
#             )
#         sequence += [
#             nn.Conv3d(in_channels=512,
#                       out_channels=512,
#                       kernel_size=4,
#                       stride=1,
#                       padding=1),
#             nn.InstanceNorm3d(num_features=512),
#             nn.LeakyReLU(0.2, inplace=False)
#         ]

#         sequence += [
#             nn.Conv3d(in_channels=512,
#                       out_channels=1,
#                       kernel_size=4,
#                       stride=1,
#                       padding=1)
#         ]

#         self.sequence = nn.ModuleList(sequence)


#     def forward(self, x):

#         for layer in self.sequence:
#             # print(f" layer: {layer}")
#             x = layer(x)
#             if isinstance(layer, nn.Conv3d):
#                 print(f"what is the shape of data: >>>> {x.shape}")

#         return x 


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