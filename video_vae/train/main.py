
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
  train_video_dataloaders = next(iter(train_video_dataloaders))


  for epoch in range(100):
    model = CausalVideoLossWrapper(num_groups=args.batch_size, args=args).to(device)
    # optimizer = create_optimizer(args=args,
    #                              model=model.vae)
      
    rec_loss, gan_loss, log_loss = model(train_video_dataloaders, args.global_step)

    



if __name__ == "__main__":
    args = get_args()
    main(args)
