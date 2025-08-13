import torch 
import argparse
import numpy as np
import random
import torch.backends.cudnn as cudnn






from video_vae.causal_video_vae_wrapper import CausalVideoVAELossWrapper
from dataset.dataset_cls import VideoDataset, ImageDataset
from dataset.dataloaders import create_mixed_datalaoders
from trainer_misc.utils import (
    init_distributed_mode
)

from utils import (
    initialize_context_parallel,
    get_rank,
    get_world_size
)



def get_args():

    parser = argparse.ArgumentParser('Pytorch Multi-process Training script for video VAE',
                                     add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--iters_per_epoch', default=2000, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)


    # Model parameters 
    parser.add_argument('--ema_update', action='store_true')
    parser.add_argument('--ema_decay', action=0.99, type=float, metavar='MODEL', help='ema decay for quantizer')

    parser.add_argument('--model_path', default='', type=str, help='The vae weight path')
    parser.add_argument('--model_dtype', default='bf16', help='The Model Dtype: bf16 or df16')

    # Using the context parallel to distribute multipler video clips to different devices 
    parser.add_argument('--use_context_parallel', action='store_true')
    parser.add_argument('--context_size', default=2, type=int, help='The context length size')
    parser.add_argument('--resolution', default=256, type=int, help='The input resolution for VAE training')
    parser.add_argument('--max_frames', default=24, type=int, help='number of max video frames')
    parser.add_argument('--use_image_video_mixed_training', action='store_true', help="whether to use the mixed image and video training")

    # The loss weights 
    parser.add_argument('--lpips_ckpt', default="flow/PATH/pyramid-flow-miniflux/lpips/vgg_lpips.pth", type=str, help="The LPIPS checkpoint path")
    parser.add_argument('--disc_start', default=0, type=int, help="The start iteration for adding GAN Loss")
    parser.add_argument('--logvar_init', default=0.0, type=float, help="The log var init" )
    parser.add_argument('--kl_weight', default=1e-6, type=float, help="The KL loss weight")
    parser.add_argument('--pixelloss_weight', default=1.0, type=float, help="The pixel reconstruction loss weight")
    parser.add_argument('--perceptual_weight', default=1.0, type=float, help="The perception loss weight")
    parser.add_argument('--disc_weight', default=0.1, type=float,  help="The GAN loss weight")
    parser.add_argument('--pretrained_vae_weight', default='', type=str, help='The pretrained vae ckpt path')  
    parser.add_argument('--not_add_normalize', action='store_true')
    parser.add_argument('--add_discriminator', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')


    # optimizer parameters 
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='optimizer (defualt: "adamw")')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (default: 1e-4)')

    parser.add_argument('--weight_decay_end', type=float, default=None, help="""
        Final value of the weight decay. We use a cosine schedule for WD.
                        (Set the same value with args.weight_decay to keep weight decay no change)
    """)
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, and clipping)')


    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='leraning_rate (default: 5e-5)')
    parser.add_argument('--lr_disc', type=float, default=1e-5, metavar='LR', help='leraning rate (default: 1e-5) of the discriminator')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='warmup leraning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='epochs to warmup LR, if scheduler supports')


    # Dataset parameters 
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--image_anno', default='', type=str, help='The image data annotation file path')
    parser.add_argument('--video_anno', default='', type=str, help='The video data annotation file path')
    parser.add_argument('--image_mix_ratio', default=0.1, type=float, help='The image data proporation in the training batch')


    # Distributed Training parameters 
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)


    parser.add_argument('--dist_eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--global_step', default=0, type=int, metavar='N', help='the global optimization step')
    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for  more efficient (sometimes) transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    # distributed training parameters 
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()




def build_model(args):

    model_dtype = args.model_dtype 
    model_path = args.model_path

    print(f"Load the base videoVAE checkpoint from path: {model_path} using dtype: {model_dtype}")

    model = CausalVideoVAELossWrapper(
        model_path=model_path,
        model_dtype='fp32',
        disc_start=args.disc_start,
        logvar_init=args.logvar_init,
        kl_weight=args.kl_weight,
        pixelloss_weight=args.pixelloss_weight,
        perceptual_weight=args.perceptual_weight,
        disc_weight=args.disc_weight,
        interpolate=False,
        add_discriminator=args.add_discriminator,
        freeze_encoder=args.freeze_encoder,
        load_loss_module=True,
        lpips_ckpt=args.lpips_ckpt
    )

    if args.pretrained_vae_weight:
        pretrained_vae_weight = args.pretrained_vae_weight 
        print(f"Loading the vae checkpoint from {pretrained_vae_weight}")
        model.load_checkpoint(pretrained_vae_weight)

    return model




def main(args):

    init_distributed_mode(args)

    # if enbaled, distributed multiple video clips to different divices 
    if args.use_context_parallel:
        initialize_context_parallel(args.context_size)

    device = torch.device(args.device)

    # fix the seed for reproducibility 
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    model = build_model(args)

    world_size = get_world_size()
    global_rank = get_rank()

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
        training_dataset = ImageDataset(anno_file=args.image_anno,
                                        resolution=args.resolution,
                                        max_frames=args.max_frames // 4,
                                        add_normalize=not args.not_add_normalize)
        

    data_loader_train = create_mixed_datalaoders(
        dataset=training_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=global_rank,
        epoch=args.seed,
        image_mix_ratio=args.image_mix_ratio,
        use_image_video_mixed_training=args.use_image_video_mixed_training
    )

    torch.distributed.barrier()

    model.to(device)
    model_without_ddp = model

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    for name, p in model.numed_parameters():
        if not p.requires_grad:
            print(name)

    print(f'total number of learnable param: {n_learnable_parameters / 1e6} M')
    print(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

    total_batch_size = args.batch_size * get_world_size()


        
    








