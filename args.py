import argparse


def get_args():

    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size', default=32, type=int)
    parse.add_argument('--epochs', default=100, type=int)
    parse.add_argument('--print_freq', default=20, type=int, help="save the weight after the `print_freq` epochs.")
    parse.add_argument('--iters_per_epochs', default=2000, type=int)
    parse.add_argument('--save_weight_freq', default=20, type=int)

    ## <-- Model paramaters --> ##
    parse.add_argument('--ema_update', action='store_true')
    parse.add_argument('--ema_decay', default=0.99, type=float, metavar='MODEL', help='ema decay for quantizer')

    parse.add_argument('--model_path', default='', type=str, help='the vae weight path.')
    parse.add_argument('--model_dtype', default='bf16', help='the model Data type')
    
    ## <-- Using the context parallel to distribute multiple video clips to different devices --> ##
    parse.add_argument('--use_context_parallel', action='store_true')
    parse.add_argument('--context_size', default=2, type=int, help='The context length size')
    parse.add_argument('--resolution', default=256, type=int, help='The shape of the image')
    parse.add_argument('--max_frames', default=24, type=int, help='number of max video frames')
    parse.add_argument('--use_image_video_mixed_training', action='store_true', help='whether to use the mixed image and video training.')

    ## <-- loss weight --> ## 
    parse.add_argument('--lpips_weight', default='', type=str, help='The Lpips weight path')
    parse.add_argument('--disc_start', default=0, type=int, help='The start itration for for adding GAN loss.')
    parse.add_argument('--logvar_init', default=0.0, type=float, help='The log varaince init is used in the vae')
    parse.add_argument('--kl_weight', default=1e-6, type=float, help='The kl loss weight')
    parse.add_argument('--pixelloss_weight', default=1.0, type=float, help='The pixel reconstruction loss weight.')
    parse.add_argument('--perceptual_weight', default=1.0, type=float, help='The perceptual loss weight')
    parse.add_argument('--disc_weight', default=0.1, type=float, help='The discriminator Gan loss weight.')
    parse.add_argument('--pretrained_vae_weight', default='', type=str, help='The pretrained vae weight.')
    parse.add_argument('--not_add_normalize', action='store_true')
    parse.add_argument('--add_discriminator', action='store_true')
    parse.add_argument('--freeze_encoder', action='store_true')

    ## <-- Optimizer parameters --> ##
    parse.add_argument('--opt', default='adamw', type=str, metavar='OPTIMZER', help='the optimizer of the vae model.')
    parse.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='The optimizer epsilon value')
    parse.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA', help='The optimizer beta (default 0.9, 0.95)' )
    parse.add_argument('--clip_grad', type=float, default=1.0, metavar='NORMALIZATION', help='clip gradient norm (default 1.0)')
    parse.add_argument('--weight_decay', type=float, default=1e-4, help='that helps prevent overfitting')
    
    parse.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate')
    parse.add_argument('--lr_disc', type=float, default=1e-5, metavar='LR', help='discriminator learning rate.')
    parse.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='start_warmup_value in cosine scheduler.')
    parse.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='final_value in cosine scheduler.')
    parse.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='cosine scheduler epochs.')
    
    ## <-- Dataset parameters --> ##
    parse.add_argument('--output_dir', default='', type=str, help='path where to save your dataset.')
    parse.add_argument('--image_anno', default='', type=str, help='annotation of image dataset')
    parse.add_argument('--video_anno', default='', type=str, help='annotation of the video dataset')
    parse.add_argument('--image_mix_ratio', default=0.1, type=float, help='The image data poration in the training batch.')

    ## <-- Distributed Training parameters --> ##
    parse.add_argument('--device', default='cuda')
    parse.add_argument('--seed', default=0, type=int)
    
    parse.add_argument('--start_epoch', default=0, type=int, metavar="N")
    parse.add_argument('--num_workers', default=10, type=int)
    parse.add_argument('--pin_mem', action='store_true')
    parse.set_defaults(pin_mem=True)

    parse.add_argument('--world_size', default=1, type=int, help='number of distributed processes.')
    parse.add_argument('--local_rank', default=-1, type=int)
    parse.add_argument('--dist_url', default='env://', help="url used to set up distributed training.")


    return parse.parse_args()



if __name__ == "__main__":
    out = get_args()
    print(out)


