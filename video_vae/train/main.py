
from logging import NOTSET
from math import isnan
import torch 

from args import get_args
import sys 
sys.path.append("../../vae_from_scratch/video_vae")
from vae.wrapper import CausalVideoLossWrapper
from dataset.video_dataloader import Video_dataloader
from middleware.optimizer import create_optimizer
from middleware.native_scaler import NativeScalerWithGradNormCount
from middleware.scheduler import cosine_scheduler
from vae.causal_vae import CausalVAE
from loss.loss import LossFunction



def main(args):
  device = torch.device("cuda:0")
  video_dataloaders = Video_dataloader(args=args)
  # video_dataloaders = next(iter(train_video_dataloaders)).to(device)

  vae = CausalVAE(num_groups=args.batch_size).to(device)
  
  
  # optimizer_g = create_optimizer(args=args,
  #                                model=vae)
  # optimizer_d = create_optimizer(args, model=loss)

  loss = LossFunction(perceptual_weight=args.perceptual_weight,
                            pixelloss_weight=args.pixelloss_weight,
                            logvar_init=args.logvar_init,
                            kl_weight=None,
                            disc_factor=args.disc_factor,
                            disc_start=args.disc_start,
                            disc_weight=args.disc_weight
                            ).to(device)

  optimizer_d = torch.optim.AdamW(params=loss.discriminator.parameters())

  optimizer_g = torch.optim.AdamW(params=vae.parameters())
  # optimizer_d = torch.optim.AdamW(params=loss.discriminator.parameters())

  kl_weight_start = 0.0 
  kl_weight_end = 1e-6 
  kl_anneal_steps = 10000
  
  
  scaler = torch.amp.GradScaler()

  global_step = 0
  for epoch in range(100):
    print(f'------------------------------------------- Epoch: [{epoch}]')

    for train_video_dataloaders in video_dataloaders:
      train_video_dataloaders = train_video_dataloaders.to(device)

      # calculate current kl_weight 
      if global_step < kl_anneal_steps:
        current_kl_weight = kl_weight_start + (kl_weight_end - kl_weight_start) * (global_step / kl_anneal_steps)
      else:
        current_kl_weight = kl_weight_end

      

      for p in loss.discriminator.parameters():
        if p.requires_grad:
          p.requires_grad = False 

      optimizer_g.zero_grad()
      with torch.autocast(device_type="cuda", dtype=torch.float32):

        
        posterior, reconstruct = vae(train_video_dataloaders)

        if torch.isnan(reconstruct).any() or torch.isinf(reconstruct).any():
          print("!!!!!!!!!!!!!!!!!!! NaN & and Inf value are found !!!!!!!!!!!!!!!!!!!!")
          break

        # the reconstruct loss 
        reconstruct_loss, rec_log = loss(train_video_dataloaders,
                                              reconstruct.detach(),
                                              posterior,
                                              global_step=global_step,
                                              last_layer=vae.get_last_layer(),
                                              optimizer_idx=0,
                                              kl_weight=current_kl_weight)  

        print(reconstruct_loss, rec_log)
      
        scaler.scale(reconstruct_loss).backward()
        scaler.unscale_(optimizer_g)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters=vae.parameters(),
                                                    max_norm=1.0)
        scaler.step(optimizer_g)
        # scaler.update()
      ################################################################################################
      for p in loss.discriminator.parameters():
        if not p.requires_grad:
          p.requires_grad = True

      optimizer_d.zero_grad()
      with torch.autocast(device_type="cuda", dtype=torch.float32):

        
        gan_loss, gan_log  = loss(train_video_dataloaders,
                                        reconstruct.detach(),
                                        posterior,
                                        global_step=global_step,
                                        last_layer=vae.get_last_layer(),
                                        optimizer_idx=1)
        print(gan_loss, gan_log)

        scaler.scale(gan_loss).backward()
        scaler.unscale_(optimizer_d)
        disc_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=loss.discriminator.parameters(), max_norm=1.0)
        scaler.step(optimizer_d)
        scaler.update()


        global_step += 1 

    


    

    






    
      
      
    



if __name__ == "__main__":
    args = get_args()
    main(args)
