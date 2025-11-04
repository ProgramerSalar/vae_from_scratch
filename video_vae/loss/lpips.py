import torch 
from torch import nn 
from torchvision import models
from collections import namedtuple




class Lpips(nn.Module):
    # Learned perceptual metric
    def __init__(self, ):
        super().__init__()
        use_dropout = True
        self.lpips_ckpt_path =  "../../vae_from_scratch/video_vae/loss/vgg_lpips.pth" # replace with your lpips path
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=False, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        ckpt = self.lpips_ckpt_path
        assert ckpt is not None, "Please replace with your lpips path"
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=True)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)


if __name__ == "__main__":
    model = LPIPS().eval()
    _ = torch.manual_seed(123)
    img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
    img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
    print(model(img1, img2).shape)
    # embed()
#####################################################################################

# class Lpips(nn.Module):

#     """ 
#     Output:
#         * A score of 0 means the images are perceptually identical.
#         * A higher score (e.g., 0.4, 0.7) means the images are perceptually different.
#         * A tiny negative value is often a numerical artifact and should be interpreted as being effectively zero
#     """

#     def __init__(self):
        
#         super().__init__()
#         self.vgg = Vgg16()
#         self.scaling_layer = ScalingLayer()

#         self.channels = [64, 128, 256, 512, 512]
#         self.linear0 = NetLinearLayer(in_chanels=self.channels[0])
#         self.linear1 = NetLinearLayer(in_chanels=self.channels[1])
#         self.linear2 = NetLinearLayer(in_chanels=self.channels[2])
#         self.linear3 = NetLinearLayer(in_chanels=self.channels[3])
#         self.linear4 = NetLinearLayer(in_chanels=self.channels[4])

#         self.linears = [self.linear0, self.linear1, self.linear2, self.linear3, self.linear4]
        

    
#     def forward(self, x, target):
        
#         scale_input, scale_target = (self.scaling_layer(x), self.scaling_layer(target))
#         vgg_input, vgg_target = self.vgg(scale_input), self.vgg(scale_target)
#         # print(vgg_input[4].shape)

#         output = 0
#         for channel in range(len(self.channels)):

#             # normalize the tensor
#             norm_input, norm_target = normalize_tensor(vgg_input[channel]), normalize_tensor(vgg_target[channel])

#             # calculate the difference
#             diff_out = (norm_input - norm_target) **2

#             # calculate the perceptual weight
#             lin = self.linears[channel].model(diff_out)
        
#             # calculate the avg of perceptual weight 
#             avg = spatial_avg(lin)
#             output += avg 

#         return output

        
        
        
            
    
# class NetLinearLayer(nn.Module):

#     """ 
#         When you compare two images, you get a difference for each of these features.
#         The SliceLinearLayer then takes these differences and performs a `weighted sum`. 
#         It has been trained on a dataset of human perceptual judgments to learn 
#         which feature differences are important and which are not.

#         Args:
#             in_channels: It takes a multi-channel feature map
#             out_channels=1: it computes a weighted sum of the values across all input channels and outputs a single value.
#             kernel_size=1 : It looks at each spatial location (pixel) independently.

#         Output: 
#             The result is a single-channel map where each pixel's value is the "perceptually weighted" difference score for that location.
#     """

#     def __init__(self,
#                  in_chanels,
#                  out_channels=1, # default 
#                  ):
        
#         super().__init__()

#         layers = []
#         layers.append(nn.Dropout())
#         layers.append(
#             nn.Conv2d(in_channels=in_chanels,
#                       out_channels=out_channels,
#                       kernel_size=1,
#                       stride=1,
#                       padding=0,
#                       bias=False)
#         )
#         self.model = nn.Sequential(*layers)
#         # print(self.model)

#     def forward(self, x):
#         x = self.model(x)
#         return x 
    
   




# class Vgg16(nn.Module):

#     def __init__(self):
#         super().__init__()

#         vgg_pretriend_feature = models.vgg16().features
#         for module in  vgg_pretriend_feature:
#             if isinstance(module, nn.ReLU):
#                 module.inplace = False
            
      
#         self.slice1 = nn.Sequential()
#         for i in range(4):
#             self.slice1.add_module(name=str(i) , module=vgg_pretriend_feature[i])


#         self.slice2 = nn.Sequential()
#         for i in range(4, 9):
#             self.slice2.add_module(name=str(i), module=vgg_pretriend_feature[i])

#         self.slice3 = nn.Sequential()
#         for i in range(8, 16):
#             self.slice3.add_module(name=str(i), module=vgg_pretriend_feature[i])

#         self.slice4 = nn.Sequential()
#         for i in range(16, 23):
#             self.slice4.add_module(name=str(i), module=vgg_pretriend_feature[i])

#         self.slice5 = nn.Sequential()
#         for i in range(23, 30):
#             self.slice5.add_module(name=str(i), module=vgg_pretriend_feature[i])


#         # print(self.slice1)
#         # print(self.slice2)
#         # print(self.slice3)
#         # print(self.slice4)
#         # print(self.slice5)

#         for param in vgg_pretriend_feature.parameters():
#             param.requires_grad = False
            
#     def forward(self, x):
#         # x = self.vgg_pretriend_feature(x)
#         s1 = self.slice1(x)
#         s2 = self.slice2(s1)
#         s3 = self.slice3(s2)
#         s4 = self.slice4(s3)
#         s5 = self.slice5(s4)

#         vgg_outputs = namedtuple(typename="VggOutputs", 
#                                  field_names=['s1', 's2', 's3', 's4', 's5'])
#         # print(vgg_outputs.__doc__)

#         output = vgg_outputs(s1, s2, s3, s4, s5)
#         # print(output[4].shape)

#         return output


# class ScalingLayer(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
#         self.register_buffer('scale', torch.Tensor([458, .448, .450])[None, :, None, None])

#     def forward(self, input):

#         result = (input - self.shift) / self.scale
#         return result


# def normalize_tensor(x, eps=1e-10):

#     norm_factor = torch.sqrt(torch.sum(x**2,
#                                        dim=1,
#                                        keepdim=True))
    
#     norm_factor = x / (norm_factor+eps)
#     return norm_factor


# def spatial_avg(x, keepdim=True):

#     result = x.mean([2, 3],
#                     keepdim)
    
#     return result




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

    out = model(input, target)
    print(out) # [2, 1, 1, 1]
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