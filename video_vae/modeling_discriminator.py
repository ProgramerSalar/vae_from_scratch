import torch 
import torch.nn as nn 
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
        
        """
        This is PatchGAN discriminator
        Arg:
            input_nc (`int`): the number of channels in input images 
            ndf (`int`): the number of filters in the last conv layer 
            n_layers (`int`): the number of conv layers in the discriminator
            
        """

        super(NLayerDiscriminator, self).__init__()

        norm_layer = nn.InstanceNorm2d

        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d

        else:
            use_bias = norm_layer != nn.BatchNorm2d


        kw = 4  # kernel width 
        padw = 1    # padding width 
        sequence = [nn.Conv2d(in_channels=input_nc,
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    nn.LeakyReLU(0.2, True)]
        

        nf_mult = 1  # number of filter multiplier 
        nf_mult_prev = 1 

        # generally increase the number of filters
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n,  8)

            sequence += [
                nn.Conv2d(in_channels=ndf * nf_mult_prev,
                          out_channels= ndf * nf_mult,
                          kernel_size=kw,
                          stride=2,
                          padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]


        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv2d(in_channels=ndf * nf_mult_prev,
                      out_channels= ndf * nf_mult,
                      kernel_size=kw,
                      stride=1,
                      padding=padw,
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(in_channels=ndf * nf_mult,
                      out_channels=1,
                      kernel_size=kw,
                      stride=1,
                      padding=padw)
        ]

        self.main = nn.Sequential(*sequence)



    def forward(self, input):
        return self.min(input)
    



class NLayerDiscriminator3D(nn.Module):

    def __init__(self,
                 input_nc=3,
                 ndf=64,
                 n_layers=3,
                 use_actnorm=False):
        


        super(NLayerDiscriminator3D, self).__init__()

        norm_layer = nn.InstanceNorm3d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d


        kw = 4 
        padw = 1 
        sequence = [nn.Conv3d(in_channels=input_nc,
                              out_channels=ndf,
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    nn.LeakyReLU(0.2, True)]
        

        nf_mult = 1 
        nf_mult_prev = 1 
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(in_channels=ndf * nf_mult_prev,
                          out_channels=ndf * nf_mult,
                          kernel_size=(kw, kw, kw),
                          stride=(1, 2, 2),
                          padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]


        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(in_channels=ndf * nf_mult_prev,
                      out_channels=ndf * nf_mult,
                      kernel_size=(kw, kw, kw),
                      stride=1,
                      padding=padw,
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(in_channels=ndf * nf_mult,
                               out_channels=1,
                               kernel_size=kw,
                               stride=1,
                               padding=padw)]
        self.main = nn.Sequential(*sequence)


    def forward(self, input):
        return self.main(input)
    



        