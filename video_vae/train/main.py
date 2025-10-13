import torch 
import sys
from torch.distributed import is_available 
sys.path.append('/content/vae_from_scratch/video_vae')

from train.args import get_args

from vae.wrapper import CausalVideoVAELossWrapper
from middleware.start_distributed_mode import init_distributed_mode
from middleware.gpu_processes import initialized_context_parallel
from train_controllers.optimizer_handlers import Optimizer_handler
from train_controllers.loss_scaler import LossScaler
from train_controllers.scheduler import cosine_scheduler
from train_controllers.train_epoch import train_epoch


def main(args):

    # start the distributed mode
    init_distributed_mode(args=args)

    if args.use_context_parallel:
        initialized_context_parallel(context_parallel_size=args.context_size)

    model_dtype = args.model_dtype
    model = CausalVideoVAELossWrapper().to("cuda:0").half()
    # print(">>>>>>>>>", model)

    # x = torch.randn(2, 3, 8, 256, 256).to("cuda:0").half()
    # out = model(x, step=10)
    # print(out)
   

    number_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable paramters: {number_learnable_parameters / 1e6} Million") # 236M

    number_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Total number of fixed parameters: {number_fix_parameters / 1e6} Million")         # 124M

    optimizer = Optimizer_handler(model=model.vae)
    optimizer_disc = Optimizer_handler(model=model.loss.discriminator)
    # print(f"optimizer: >>>> {optimizer}")

    # print(model.loss.discriminator)

    loss_scaler = LossScaler()
    loss_scaler_disc = LossScaler()

    lr_schedule_values = cosine_scheduler()
    lr_schedule_values_disc = cosine_scheduler()

    # print(f"lr_schedule_values: >>>>>>>>>>>>> {len(lr_schedule_values)}")

    device = torch.device("cuda" if torch.cuda.is_available() else None)

    torch.distributed.barrier()

    for epoch in range(0, 10):
      print(f"--------------------------> epoch: {epoch}")
      train_epoch(model=model,
                optimizer=optimizer,
                optimizer_disc=optimizer_disc,
                epoch=epoch,
                lr_scheduler_values=lr_schedule_values,
                lr_scheduler_values_disc= lr_schedule_values_disc)

    

    











if __name__ == "__main__":
    args = get_args()
    out = main(args=args)


