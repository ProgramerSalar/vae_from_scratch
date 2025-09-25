import torch 
from torch import nn 
from einops import rearrange

from discriminator import NLayerDiscriminator, NLayerDiscriminator3D, weights_init
from lpips import LPIPS
from enc_dec import DiagonalGaussianDistribution




def hinge_d_loss(logits_real,
                 logits_fake):
    
    loss_real = torch.mean(nn.functional.relu(input=1.0 - logits_real))
    loss_fake = torch.mean(nn.functional.relu(input=1.0 + logits_fake))

    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real,
                   logits_fake):
    
    d_loss = 0.5 * (
        torch.mean(nn.functional.softplus(-logits_real))
        + torch.mean(nn.functional.softplus(logits_fake))
    )

    return d_loss


def adopt_weight(weight,
                 global_step,
                 threshold=0,
                 value=0.0):
    
    if global_step < threshold:
        weight = value
    return weight


class LPIPSWithDiscriminator(nn.Module):
    
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 perceptual_weight=1.0,
                 lpips_ckpt="vae/vgg_lpips.pth",
                 # <-- Discriminator loss --> 
                 disc_num_layers=4,
                 disc_in_channels=3,
                 disc_factor=1.0,
                 disc_weight=0.5,
                 disc_loss="hinge",
                 add_discriminator=True,
                 using_3d_discriminator=False
                 ):
        
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "make sure choose ['hinge', 'vanilla'] discriminator loss"
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

        if add_discriminator:
            # so i am adding the discriminator 
            self.discriminator = NLayerDiscriminator3D if using_3d_discriminator else NLayerDiscriminator
            self.discriminator = self.discriminator(input_nc=disc_in_channels,
                                                    ndf=64,
                                                    n_layers=disc_num_layers).apply(weights_init)
        else:
            self.discriminator = None



        self.perceptual_loss = LPIPS(use_dropout=True,
                                     lpips_ckpt_path=lpips_ckpt).eval()
        self.logvar = nn.Parameter(data=torch.ones(())* logvar_init)


        self.disc_start = disc_start
        self.kl_weight = kl_weight
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.using_3d_discriminator = using_3d_discriminator

    def calculate_adaptive_weight(self,
                                  nll_loss,
                                  g_loss,
                                  last_layer=None):
        
        if last_layer is not None:
            
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            print(f"working in Progress....")

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

        


    
    def forward(self, 
                inputs,
                reconstructions,
                posteriors,
                optimizer_idx,
                global_step,
                split="train",
                last_layer=None):
        

        t = reconstructions.shape[2]
        inputs = rearrange(tensor=inputs,
                           pattern="b c t h w -> (b t) c h w").contiguous()
        reconstructions = rearrange(reconstructions,
                                    pattern="b c t h w -> (b t) c h w").contiguous()
        
        # reconstruction loss 
        if optimizer_idx == 0:
            
            rec_loss = torch.mean(input=nn.functional.mse_loss(input=inputs,
                                                               target=reconstructions,
                                                               reduction='none'),
                                    dim=(1, 2, 3),
                                    keepdim=True)
            # print(rec_loss.shape)   # torch.Size([8, 1, 1, 1])
            
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)  # torch.Size([8, 1, 1, 1])
                nll_loss = self.pixelloss_weight * rec_loss + self.perceptual_weight * p_loss   # torch.Size([8, 1, 1, 1])

            nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar  # torch.Size([8, 1, 1, 1])
            weighted_nll_loss = nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]   # tensor(1.9975, grad_fn=<DivBackward0>)
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]  # tensor(1.9975, grad_fn=<DivBackward0>)
            
            kl_loss = posteriors.kl()
            kl_loss = torch.mean(kl_loss)

            disc_factor = adopt_weight(
                weight=self.disc_factor,
                global_step=global_step,
                threshold=self.disc_start
            )   # 0.0
            
            if disc_factor > 0.0:
                # print("work in progress....")
                if self.using_3d_discriminator:
                    reconstructions = rearrange(reconstructions,
                                                '(b t) c h w -> b c t h w', t=t)
                    
                logits_fake = self.discriminator(x=reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)

                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss=nll_loss,
                                                              g_loss=g_loss,
                                                              last_layer=last_layer)

                except RuntimeError:
                    assert not self.training, "please come and see the {loss} file code. and solve the error."
                    d_weight = torch.tensor(0.0)

            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0)

            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
            )
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                "{}/d_weight".format(split): torch.tensor(disc_factor),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean()
            }
            
            return loss, log 
                



        # discriminator loss 
        if optimizer_idx == 1:
            if self.using_3d_discriminator:
                inputs = rearrange(inputs, 
                                   "(b t) c h w -> b c t h w", t=t)
                reconstructions = rearrange(reconstructions,
                                            "(b t) c h w -> b c t h w", t=t)
                
            logits_real = self.discriminator(x=inputs.contiguous().detach())    # torch.Size([8, 1, 14, 14])
            logits_fake = self.discriminator(x=reconstructions.contiguous().detach())  # torch.Size([8, 1, 14, 14])

            # print(logits_real.shape, logits_fake.shape)
            disc_factor = adopt_weight(weight=self.disc_factor,     # 1.0 default
                                       global_step=global_step,     # 10
                                       threshold=self.disc_start,   # 100
                                       )
            # print(disc_factor)  # 0.0

            d_loss = disc_factor * self.disc_loss(logits_real=logits_real,
                                                  logits_fake=logits_fake)
            
            # print(d_loss)   # tensor(0., grad_fn=<MulBackward0>)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }

            return d_loss, log 


        



if __name__ == "__main__":
    out = LPIPSWithDiscriminator(disc_start=1000)
    # print(out)

    x = torch.randn(2, 3, 4, 256, 256)
    rec = torch.randn(2, 3, 4, 256, 256)
    posterior = DiagonalGaussianDistribution(parameters=x)
    optimizer_idx = 0
    global_step = 10


    out = out(x, rec, posterior, optimizer_idx, global_step)
    print(out)
