import torch 
from torch import nn 
from torchvision import models
from collections import namedtuple





class Lpips(nn.Module):

    """ 
    Output:
        * A score of 0 means the images are perceptually identical.
        * A higher score (e.g., 0.4, 0.7) means the images are perceptually different.
        * A tiny negative value is often a numerical artifact and should be interpreted as being effectively zero
    """

    def __init__(self):
        
        super().__init__()
        self.vgg = Vgg16()
        self.scaling_layer = ScalingLayer()

        self.channels = [64, 128, 256, 512, 512]
        self.linear0 = NetLinearLayer(in_chanels=self.channels[0])
        self.linear1 = NetLinearLayer(in_chanels=self.channels[1])
        self.linear2 = NetLinearLayer(in_chanels=self.channels[2])
        self.linear3 = NetLinearLayer(in_chanels=self.channels[3])
        self.linear4 = NetLinearLayer(in_chanels=self.channels[4])

        self.linears = [self.linear0, self.linear1, self.linear2, self.linear3, self.linear4]
        

    
    def forward(self, x, target):
        
        scale_input, scale_target = (self.scaling_layer(x), self.scaling_layer(target))
        vgg_input, vgg_target = self.vgg(scale_input), self.vgg(scale_target)
        # print(vgg_input[4].shape)

        output = 0
        for channel in range(len(self.channels)):

            # normalize the tensor
            norm_input, norm_target = normalize_tensor(vgg_input[channel]), normalize_tensor(vgg_target[channel])

            # calculate the difference
            diff_out = (norm_input - norm_target) **2

            # calculate the perceptual weight
            lin = self.linears[channel].model(diff_out)
        
            # calculate the avg of perceptual weight 
            avg = spatial_avg(lin)
            output += avg 

        return output

        
        
        
            
    
class NetLinearLayer(nn.Module):

    """ 
        When you compare two images, you get a difference for each of these features.
        The SliceLinearLayer then takes these differences and performs a `weighted sum`. 
        It has been trained on a dataset of human perceptual judgments to learn 
        which feature differences are important and which are not.

        Args:
            in_channels: It takes a multi-channel feature map
            out_channels=1: it computes a weighted sum of the values across all input channels and outputs a single value.
            kernel_size=1 : It looks at each spatial location (pixel) independently.

        Output: 
            The result is a single-channel map where each pixel's value is the "perceptually weighted" difference score for that location.
    """

    def __init__(self,
                 in_chanels,
                 out_channels=1, # default 
                 ):
        
        super().__init__()

        layers = []
        layers.append(nn.Dropout())
        layers.append(
            nn.Conv2d(in_channels=in_chanels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False)
        )
        self.model = nn.Sequential(*layers)
        # print(self.model)

    def forward(self, x):
        x = self.model(x)
        return x 
    
   




class Vgg16(nn.Module):

    def __init__(self):
        super().__init__()

        vgg_pretriend_feature = models.vgg16().features
        for module in  vgg_pretriend_feature:
            if isinstance(module, nn.ReLU):
                module.inplace = False
            
      
        self.slice1 = nn.Sequential()
        for i in range(4):
            self.slice1.add_module(name=str(i) , module=vgg_pretriend_feature[i])


        self.slice2 = nn.Sequential()
        for i in range(4, 9):
            self.slice2.add_module(name=str(i), module=vgg_pretriend_feature[i])

        self.slice3 = nn.Sequential()
        for i in range(8, 16):
            self.slice3.add_module(name=str(i), module=vgg_pretriend_feature[i])

        self.slice4 = nn.Sequential()
        for i in range(16, 23):
            self.slice4.add_module(name=str(i), module=vgg_pretriend_feature[i])

        self.slice5 = nn.Sequential()
        for i in range(23, 30):
            self.slice5.add_module(name=str(i), module=vgg_pretriend_feature[i])


        # print(self.slice1)
        # print(self.slice2)
        # print(self.slice3)
        # print(self.slice4)
        # print(self.slice5)

        for param in vgg_pretriend_feature.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x = self.vgg_pretriend_feature(x)
        s1 = self.slice1(x)
        s2 = self.slice2(s1)
        s3 = self.slice3(s2)
        s4 = self.slice4(s3)
        s5 = self.slice5(s4)

        vgg_outputs = namedtuple(typename="VggOutputs", 
                                 field_names=['s1', 's2', 's3', 's4', 's5'])
        # print(vgg_outputs.__doc__)

        output = vgg_outputs(s1, s2, s3, s4, s5)
        # print(output[4].shape)

        return output


class ScalingLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([458, .448, .450])[None, :, None, None])

    def forward(self, input):

        result = (input - self.shift) / self.scale
        return result


def normalize_tensor(x, eps=1e-10):

    norm_factor = torch.sqrt(torch.sum(x**2,
                                       dim=1,
                                       keepdim=True))
    
    norm_factor = x / (norm_factor+eps)
    return norm_factor


def spatial_avg(x, keepdim=True):

    result = x.mean([2, 3],
                    keepdim)
    
    return result




if __name__ == "__main__":

    # model = Vgg16()
    # print(model)
    # x = torch.randn(2, 3, 256, 256)
    # out = model(x)
    # print(out[4].shape)
    # --------------------

    # model = ScalingLayer()
    # x = torch.randn(2, 3, 256, 256)
    # out = model(x)
    # print(out.shape)
    # -------------------------

    model = Lpips()
    print(model)
    input = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)

    for epoch in range(20):
      print(f"<--------------- epoch: {epoch}")
      out = model(input, target)
      print(out.dtype) # [2, 1, 1, 1]
    # loss = out.mean()
    # print(loss.item())

    # print(normalize_tensor(input))
    # print(spatial_avg(input).shape)
    # -----------------------------------

    # model = NetLinearLayer(in_chanels=3)
    
    # print(models)
    # x = torch.randn(2, 3, 256, 256)
    # out = model(x)
    # print(out.shape)