import torch 

from args import get_args
import sys 
sys.path.append("../../vae_from_scratch/video_vae")
from vae.wrapper import CausalVideoLossWrapper
from dataset.video_dataloader import Video_dataloader
from middleware.optimizer import create_optimizer
from middleware.native_scaler import NativeScalerWithGradNormCount
from middleware.scheduler import cosine_scheduler
from train.one_epoch import train_one_epoch

def main(args):
    
    device = torch.device("cuda:0")
    model = CausalVideoLossWrapper(num_groups=args.batch_size, args=args).to(device)
    # Get the device of the first parameter of the model
    model_device = next(model.parameters()).device

    print(f"The model is currently on device: {model_device}")

    # You can also check specifically if it is on a CUDA device
    if model_device.type == 'cuda':
        print("The model is on a CUDA (GPU) device.")
    elif model_device.type == 'cpu':
        print("The model is on a CPU device.")
    
    num_training_steps_per_epoch = args.iters_per_epoch
    train_video_dataloaders = Video_dataloader(args=args)
    
    model_without_ddp = model
    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {n_learnable_parameters / 1e6} Million")

    print(f"LR: {args.lr:.8f}")
    print(f"Min Lr: {args.min_lr:.8f}")
    print(f"Weight Decay: {args.weight_decay:.8f}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of training steps {(num_training_steps_per_epoch * args.epochs)}")
    print(f"Number of Training example per epoch {(args.batch_size * num_training_steps_per_epoch)}")

    optimizer = create_optimizer(args=args,
                                 model=model_without_ddp.vae)

    
    optimizer_disc = create_optimizer(args=args,
                                      model=model_without_ddp.loss.discriminator) if args.add_discriminator else None
    
    # print(f"is_second_order: >>>>>>>>> {optimizer_disc.is_second_order}")
    
    loss_scaler = NativeScalerWithGradNormCount(enable=True if args.model_dtype == "fp16" else False)
    loss_scaler_disc = NativeScalerWithGradNormCount(enable=True if args.model_dtype == "fp16" else False) if args.add_discriminator else None 
    # print(loss_scaler_disc)

    lr_schedule_values = cosine_scheduler(base_value=args.lr,
                                          final_value=args.min_lr,
                                          epochs=args.epochs,
                                          niter_per_ep=num_training_steps_per_epoch,
                                          warmup_epochs=args.warmup_epoch,
                                          warmup_steps=args.warmup_steps)
    # print(len(lr_schedule_values))

    lr_schedule_values_disc = cosine_scheduler(base_value=args.lr,
                                          final_value=args.min_lr,
                                          epochs=args.epochs,
                                          niter_per_ep=num_training_steps_per_epoch,
                                          warmup_epochs=args.warmup_epoch,
                                          warmup_steps=args.warmup_steps) if args.add_discriminator else None
    # print(lr_schedule_values_disc)

    print(f"Start training for {args.epochs} the global iters is {args.global_step}")
    log_writer = None

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(args,
            model=model,
            optimizer=optimizer,
            optimizer_disc=optimizer_disc,
            epoch=epoch,
            lr_schedule_values=lr_schedule_values,
            lr_schedule_values_disc=lr_schedule_values_disc,
            data_loader = train_video_dataloaders,
            loss_scaler=loss_scaler,
            loss_scaler_disc=loss_scaler_disc,
            log_writer=log_writer
        )
    
    





if __name__ == "__main__":
    args = get_args()
    main(args)