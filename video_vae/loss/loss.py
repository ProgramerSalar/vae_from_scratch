import torch 
from torch import nn 
from einops import rearrange
import sys 
sys.path.append("../../vae_from_scratch/video_vae")

# from loss.discriminator import NumberLayerDiscriminator, NLayerDiscriminator3D, weights_init
from loss.discriminator import NumberLayerDiscriminator, NumberLayerDiscriminator3d, weights_init
from loss.lpips import Lpips
# from lpips import Lpips

from vae.gaussian import DiagonalGaussianDistribution
from vae.causal_vae import CausalVAE
from vae.enc import Encoder

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.video_dataset import VideoDataset
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)



class LossFunction(nn.Module):

    """This is Loss function where found the Lpips and Discriminator."""

    def __init__(self,
                 disc_in_channels=3,
                 perceptual_weight=1.0,
                 pixelloss_weight=1.0,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 # <-- Discriminator --> ## 
                 disc_factor=1.0,
                 disc_start=0,
                 disc_weight=0.5,
                useing_3d_discriminator=True
                 ):
        
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.pixel_weight = pixelloss_weight
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.kl_weight = kl_weight
        self.using_3d_discriminator = useing_3d_discriminator
        self.discriminator_iter_start = disc_start

        """
            perceptual_weight: you can turn to fine-tune how much your model cares about making images that look right to a human eye.
            pixelloss_weight: calculate the losses to image pixel wise.
            disc_factor: filter the value to be range of 1.0 that's real when value to lower than 0 the more chance that has fake value.
            disc_weight: Balancing value of discriminator fake and real data.
        """

        self.discriminator = NumberLayerDiscriminator3d(in_channels=disc_in_channels).apply(weights_init) if self.using_3d_discriminator \
                                else NumberLayerDiscriminator(in_channels=disc_in_channels).apply(weights_init)
        self.lpips = Lpips().eval()
        self.logvar = nn.Parameter(data=torch.ones(()) * logvar_init)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):

        if last_layer is not None:
            nll_grads = torch.autograd.grad(outputs=nll_loss,
                                                inputs=last_layer,
                                                retain_graph=True
                                                )[0]
            
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        
        return d_weight

 
    def forward(self, 
                input, 
                reconstruct, 
                posteriors, 
                global_step, 
                last_layer,
                optimizer_idx,
                split="train",
                kl_weight=None):
        
        t = reconstruct.shape[2]
        input = rearrange(input, 'b c t h w -> (b t) c h w').contiguous()
        reconstruct = rearrange(reconstruct, 'b c t h w -> (b t) c h w').contiguous()

        

        if optimizer_idx == 0:

            

            ##  calculate the reconstruction loss 
            rec_loss = torch.mean(nn.functional.mse_loss(input, reconstruct, reduction='none'),
                                  dim=(1, 2, 3),
                                  keepdim=True)

           
            
            if self.perceptual_weight > 0:
                
                # [16, 3, 256, 256] -> [16, 1, 1, 1]
                perceputual_loss = self.lpips(input, reconstruct)
                # print(perceputual_loss.shape)
                nll_loss = self.pixel_weight * rec_loss + self.perceptual_weight * perceputual_loss
               

            nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar  
            weighted_nll_loss = nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            

            kl_loss = posteriors.kl()
            kl_loss = torch.mean(kl_loss)
            
            
            disc_factor = adopt_weight(
                weight=self.disc_factor,
                global_step=global_step,
                threshold=self.discriminator_iter_start
            )

            if disc_factor > 0.0:
                if self.using_3d_discriminator:
                    reconstruct = rearrange(reconstruct, "(b t) c h w -> b c t h w", t=t)
                
                
                logits_fake = self.discriminator(reconstruct.contiguous())    
                print(f"what is the shape of logits_fake: >>>>>>>>>>> {logits_fake.shape}")
                g_loss = -torch.mean(logits_fake)

                
                d_weight = self.calculate_adaptive_weight(nll_loss=nll_loss,
                                                          g_loss=g_loss,
                                                          last_layer=last_layer)
                
                

                
            loss = (
            weighted_nll_loss 
            + kl_weight * kl_loss
            + d_weight * disc_factor * g_loss
            )

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/perception_loss".format(split): perceputual_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            
            return loss, log
        
        if optimizer_idx == 1:
            if self.using_3d_discriminator:
                input = rearrange(input, "(b t) c h w -> b c t h w", t=t)
                reconstruct = rearrange(reconstruct, "(b t) c h w -> b c t h w", t=t)

            
            # [16, 3, 256, 256] -> [16, 1, 14, 14]
            real_logits = self.discriminator(input.contiguous().detach())
            fake_logits = self.discriminator(reconstruct.contiguous().detach())
            
            disc_factor = adopt_weight(weight=self.disc_factor,
                                       global_step=global_step,
                                       threshold=self.discriminator_iter_start)
            
            d_loss = disc_factor * hinge_disc_loss(logits_real=real_logits,
                                     logits_fake=fake_logits)
            
            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): real_logits.detach().mean(),
                "{}/logits_fake".format(split): fake_logits.detach().mean(),
            }
            
            
            return d_loss, log

            
            
                
def hinge_disc_loss(logits_real, logits_fake):

    loss_real = torch.mean(nn.functional.relu(1.0 - logits_real))
    loss_fake = torch.mean(nn.functional.relu(1.0 + logits_fake))

    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
        


def adopt_weight(weight,
                 global_step,
                 threshold=0,
                 value=0.0):

    # if global_step < threshold:
    #     weight = value

    # else:
    #     assert ValueError, "make sure global_step is minimum of threshold"

    return weight








