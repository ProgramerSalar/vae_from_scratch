
import torch 

from args import get_args
import sys 
sys.path.append("../../vae_from_scratch/video_vae")
from vae.wrapper import CausalVideoLossWrapper
from dataset.video_dataloader import Video_dataloader
from middleware.optimizer import create_optimizer
from middleware.native_scaler import NativeScalerWithGradNormCount
from middleware.scheduler import cosine_scheduler




def main(args):
  device = torch.device("cuda:0")
  train_video_dataloaders = Video_dataloader(args=args)
  train_video_dataloaders = next(iter(train_video_dataloaders)).to(device)
  model = CausalVideoLossWrapper(num_groups=args.batch_size, args=args).to(device)
  optimizer = create_optimizer(args=args,
                                 model=model)
  
  scaler = torch.amp.GradScaler(device="cuda")


  for epoch in range(100):
    optimizer.zero_grad()

    # the reconstruct loss 
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
      rec_loss, gan_loss, log_loss = model(train_video_dataloaders, epoch)
      print(rec_loss, gan_loss, log_loss)

    scaler.scale(rec_loss).backward()
    scaler.unscale_(optimizer)
    total_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                      max_norm=1.0)
    print(f"what is the value have total_grad_norm: >>>>>>>>>>>>>> {total_grad_norm}")
    scaler.step(optimizer)
    scaler.update()





    
      
      
    



if __name__ == "__main__":
    args = get_args()
    main(args)
