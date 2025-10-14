import torch 
from torch import nn 
from torchvision import models
from collections import namedtuple


class LPIPS(nn.Module):

    def __init__(self,
                 use_dropout=True,
                 lpips_ckpt_path=None):
        super().__init__()
        
        # replace with your lpips path
        self.lpips_ckpt_path = lpips_ckpt_path
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]    
        self.net = Vgg16(requires_grad=False)

        self.lin0 = NetLinLayer(in_channels=self.channels[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(in_channels=self.channels[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(in_channels=self.channels[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(in_channels=self.channels[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(in_channels=self.channels[4], use_dropout=use_dropout)

    
    def forward(self, 
                input,
                target):
        
        in0_input, in1_input =  (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)

        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        for kk in range(len(self.channels)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(x=lins[kk].model(diffs[kk]),
                            keepdim=True) for kk in range(len(self.channels))
            ]
        
        val = res[0]
        for l in range(1, len(self.channels)):
            val += res[l]

        return val
        







def normalize_tensor(x,
                     eps=1e-10):
    
    norm_factor = torch.sqrt(torch.sum(x**2, 
                                       dim=1,
                                       keepdim=True))
    
    return x / (norm_factor+eps)


def spatial_average(x, keepdim=True):

    return x.mean([2, 3],
                  keepdim=keepdim)



class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])     # mean
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])        # variance 

    def forward(self, inp):
        return (inp - self.shift) / self.scale 
    

class Vgg16(nn.Module):
    def __init__(self,
                 requires_grad=False,
                 ):
        super().__init__()
        
        vgg_pretrained_features = models.vgg16().features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        
        for i in range(4): 
            self.slice1.add_module(name=str(i), module=vgg_pretrained_features[i]) 
        for i in range(4, 9): 
            self.slice2.add_module(name=str(i), module=vgg_pretrained_features[i])
        for i in range(9, 16):  
            self.slice3.add_module(name=str(i), module=vgg_pretrained_features[i])
        for i in range(16, 23):  
            self.slice4.add_module(name=str(i), module=vgg_pretrained_features[i])
        for i in range(23, 30):  
            self.slice5.add_module(name=str(i), module=vgg_pretrained_features[i])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)

        vgg_outputs = namedtuple(typename="VggOutputs", field_names=['h1', 'h2', 'h3', 'h4', 'h5'])
        out = vgg_outputs(h1, h2, h3, h4, h5)

        return out 



class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self,
                 in_channels,
                 out_channels: int = 1,
                 use_dropout: bool = False):
        super().__init__()
        
        layers = []
        if use_dropout:
            layers.append(nn.Dropout())

        layers.append(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False)
        )

        self.model = nn.Sequential(*layers)


if __name__ == "__main__":
    lpips = LPIPS()

    input = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)

    out = lpips(input, target)
    print(out)