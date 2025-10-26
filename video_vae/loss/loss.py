import torch 
from torch import nn 
from einops import rearrange
import sys 
sys.path.append("../../vae_from_scratch/video_vae")

from loss.discriminator import NumberLayerDiscriminator
# from discriminator import NumberLayerDiscriminator
from loss.lpips import Lpips
# from lpips import Lpips

from vae.gaussian import DiagonalGaussianDistribution
from vae.causal_vae import CausalVAE
from vae.enc import Encoder

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.video_dataset import VideoDataset

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
                 disc_weight=0.5
                
                 ):
        
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.pixel_weight = pixelloss_weight
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.kl_weight = kl_weight

        """
            perceptual_weight: you can turn to fine-tune how much your model cares about making images that look right to a human eye.
            pixelloss_weight: calculate the losses to image pixel wise.
            disc_factor: filter the value to be range of 1.0 that's real when value to lower than 0 the more chance that has fake value.
            disc_weight: Balancing value of discriminator fake and real data.
        """

        self.discriminator = NumberLayerDiscriminator(in_channels=disc_in_channels)
        self.lpips = Lpips().eval()
        self.logvar = nn.Parameter(data=torch.ones(()) * logvar_init)

 
    def forward(self, 
                input, 
                reconstruct, 
                posteriors, 
                global_step, 
                last_layer,
                optimizer_idx):

        input = rearrange(input, 'b c t h w -> (b t) c h w').contiguous()
        target = rearrange(reconstruct, 'b c t h w -> (b t) c h w').contiguous()

        if optimizer_idx == 0:

            ##  calculate the reconstruction loss 
            rec_loss = torch.mean(nn.functional.mse_loss(input, target, reduction='none'),
                                  dim=(1, 2, 3),
                                  keepdim=True)
            
            if self.perceptual_weight > 0:
                
                # [16, 3, 256, 256] -> [16, 1, 1, 1]
                perceputual_loss = self.lpips(input, target)
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
                threshold=0
            )

            if disc_factor > 0.0:
                # [16, 3, 256, 256] -> [16, 1, 14, 14]
                logits_fake = self.discriminator(target)    
                g_loss = -torch.mean(logits_fake)

                nll_grads = torch.autograd.grad(outputs=nll_loss,
                                                inputs=last_layer,
                                                retain_graph=True
                                                )[0]
            
                g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
                d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
                d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
                d_weight = d_weight * self.disc_weight


            loss = (
            weighted_nll_loss 
            + self.kl_weight * kl_loss
            + d_weight * disc_factor * g_loss
            )

            log = {
                f"total_loss: {loss.mean()},",
                f"kl_loss: {kl_loss.mean()}",
                f"nll_loss: {nll_loss.mean()}",
                f"rec_loss: {rec_loss.mean()}",
                f"perceptual_loss: {perceputual_loss.mean()}",
                f"g_loss: {g_loss.mean()}"
            }
            
            return log, loss
        
        if optimizer_idx == 1:

            # [16, 3, 256, 256] -> [16, 1, 14, 14]
            real_logits = self.discriminator(input)
            fake_logits = self.discriminator(target)
            
            disc_factor = adopt_weight(weight=self.disc_factor,
                                       global_step=global_step,
                                       threshold=0)
            
            d_loss = hinge_disc_loss(logits_real=real_logits,
                                     logits_fake=fake_logits)
            
            log = {
                f"disc_loss: {d_loss.mean()}",
                f"logits_real: {real_logits.mean()}",
                f"logits_fake: {fake_logits.mean()}"
            }
            
            
            return log, d_loss

            
            
                
def hinge_disc_loss(logits_real, logits_fake):

    loss_real = torch.mean(nn.functional.relu(1.0 - logits_real))
    loss_fake = torch.mean(nn.functional.relu(1.0 + logits_fake))

    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
        


def adopt_weight(weight,
                 global_step,
                 threshold=0,
                 value=0.0):

    if global_step < threshold:
        weight = value

    else:
        assert ValueError, "make sure global_step is minimum of threshold"

    return weight










if __name__ == "__main__":

    # original data 
    ## Dataset 
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    

    # Instantiate the Dataset
    video_dataset = VideoDataset(video_dir='/home/manish/Desktop/projects/vae_from_scratch/vae_from_scratch/Data/train_dataset', num_frames=16, transform=data_transform)
    print(f"Dataset created with {len(video_dataset)} videos.")
    data_loader = DataLoader(video_dataset, batch_size=2, shuffle=True, num_workers=2)
    x = next(iter(data_loader))
    x = rearrange(x, 'b t c h w -> b c t h w').contiguous()
    

    # Test dataset 
    test_dataset = VideoDataset(video_dir='/home/manish/Desktop/projects/vae_from_scratch/vae_from_scratch/Data/test_dataset', num_frames=16, transform=data_transform)
    print(f"Dataset created with {len(test_dataset)} videos.")
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=2)
    target = next(iter(test_loader))
    target = rearrange(target, 'b t c h w -> b c t h w').contiguous()
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LossFunction().to(device)
    print(model)

    # x = torch.randn(2, 3, 8, 256, 256).to(device)
    # target = torch.randn(2, 3, 8, 256, 256).to(device)

    

    
    vae = CausalVAE().to(device)
    print(vae)
    posterior, reconstruct = vae(x)
    out = model(x, reconstruct, posterior, 100, vae.get_last_layer())
    print(out)
    # ----------------------------------------------------------------------------

    # fun = adopt_weight(weight=1.0,
    #                    global_step=100,
    #                    threshold=0,
    #                    value=0.0)
    
    # print(fun)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CausalVAE(device=device)
    # print(model.get_last_layer())