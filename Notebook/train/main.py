import torch, random, time, os, json
import numpy as np 
import torch.backends.cudnn as cudnn
from datetime import timedelta
from pathlib import Path 


from middleware.init_dist_mode import init_distributed_mode
from flow.utils import initialize_context_parallel, get_rank, get_world_size
from build_model import build_model
from flow.dataset.dataset_cls import VideoDataset, ImageDataset
from flow.dataset.dataloaders import create_mixed_dataloaders
from middleware.create_optimizer import create_optimizer
from middleware.nativescaler_with_gradnormcount import NativeScalerWithGradNormCount
from middleware.scheduler import cosine_scheduler
from middleware.autoload_model import auto_load_model
from middleware.vae_ddp_trainer import train_one_epoch
from middleware.save_model import save_model, is_main_process
from get_args import get_args

def main(args):

    init_distributed_mode(args)

    # if enabled, distributed multiple video clips to different divices.
    if args.use_context_parallel:
        initialize_context_parallel(args.context_size)

    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility 
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # The direct the cudnn library to run a few benchmark tests to find the fastest conv algo
    # for the specific input size your model is using.
    cudnn.benchmark = True
    model = build_model(args)

    world_size = get_world_size()
    global_rank = 0 

    num_training_steps_per_epoch = args.iters_per_epoch
    log_writer = None 

    # building dataset and dataloaders 
    image_gpus = max(1, int(world_size * args.image_mix_ratio))
    if args.use_image_video_mixed_training:
        video_gpus = world_size - image_gpus
    else:
        # only use video data 
        video_gpus = world_size
        image_gpus = 0 

    
    if global_rank < video_gpus:
        training_dataset = VideoDataset(anno_file=args.video_anno,
                                        resolution=args.resolution,
                                        max_frames=args.max_frames,
                                        add_normalize=not args.not_add_normalize)
        
    else:
        training_dataset = ImageDataset(anno_file=args.anno_file,
                                        resolution=args.resolution,
                                        max_frames=args.max_frames // 4,
                                        add_normalize=not args.not_add_normalize)

    print(f"What is the global rank: {global_rank}")
    data_loader_train = create_mixed_dataloaders(
        dataset=training_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=global_rank,
        epoch=args.seed,
        image_mix_ratio=args.image_mix_ratio,
        use_image_video_mixed_training=args.use_image_video_mixed_training   
    )
    # Synchronize all processes.
    torch.distributed.barrier()

    model.to(device)
    model_without_ddp = model

    n_learnable_parameters = sum(
        p.numel()
        if p.requires_grad else print("requires_grad are not found.")
        for p in model.parameters()
    )
    n_fix_parameters = sum(
        p.numel()
        if not p.requires_grad else print("requires_grad are found.")
        for p in model.parameters()
    )

    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(name)
    print(f"Total number of learnable params: {n_learnable_parameters / 1e6} M")
    print(f"Total number of fixed params in : {n_fix_parameters / 1e6} M")


    total_batch_size = args.batch_size * get_world_size()
    print(f"LR = {args.lr:.8f}")
    print(f"Min LR = {args.min_lr:.8f}")
    print(f"Weight Decay = {args.weight_decay:.8f}")
    print(f"Batch size = {total_batch_size}")
    print(f"Number of training steps = {num_training_steps_per_epoch * args.epochs}")
    print(f"Number of training examples per epoch = {total_batch_size * num_training_steps_per_epoch}")

    optimizer = create_optimizer(args=args,
                                 model=model_without_ddp.vae)
    optimizer_disc = create_optimizer(args, model_without_ddp.loss.discriminator) if args.add_discriminator else None

    loss_scaler = NativeScalerWithGradNormCount(enabled=True if args.model_dtype == "fp16" else False)
    loss_scaler_disc = NativeScalerWithGradNormCount(enabled=True if args.model_dtype == "fp16" else False) if args.add_discriminator else None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    print("Use step level LR (learning_rate) & WD (weight_decay) scheduler")
    lr_schedule_values = cosine_scheduler(
        base_value=args.lr,
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_ep=num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps
    )
    lr_schedule_values_disc = cosine_scheduler(
        base_value=args.lr,
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_ep=num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps
    ) if args.add_discriminator else None


    auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        optimizer_disc=optimizer_disc
    )

    print(f"Start training for {args.epochs} epochs, the global iteration is {args.global_step}")
    start_time = time.time()
    torch.distributed.barrier()

    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(model=model,
                                      model_dtype=args.model_dtype,
                                      data_loader=data_loader_train,
                                      optimizer=optimizer,
                                      optimizer_disc=optimizer_disc,
                                      loss_scaler=loss_scaler,
                                      loss_scaler_disc=loss_scaler_disc,
                                      clip_grad=args.clip_grad,
                                      log_writer=log_writer,
                                      lr_schedule_values=lr_schedule_values,
                                      lr_schedule_values_disc=lr_schedule_values_disc,
                                      args=args,
                                      print_freq=args.print_freq,
                                      iters_per_epoch=num_training_steps_per_epoch)
        
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 \
                or epoch + 1 == args.epochs:
                
                save_model(
                    args=args,
                    epoch=epoch,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    optimizer_disc=optimizer_disc,
                    save_ckpt_freq=args.save_ckpt_freq
                )

        log_stats = {

            **{   f'train_{k}': v
                for k, v in train_stats.items()
            },
            'epoch': epoch,
            'n_parameters': n_learnable_parameters
        }

        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            
            with open(os.path.join(args.output_dir,
                                   "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")








if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(opts)

