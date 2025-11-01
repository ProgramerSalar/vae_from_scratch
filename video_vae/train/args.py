import argparse 


def get_args():
    parser = argparse.ArgumentParser("Let's train a vae model", add_help=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--iters_per_epoch', default=2000, type=int)
    parser.add_argument('--model_dtype', default='bf16', type=str)
    parser.add_argument('--global_step', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    
    # optimizer parameters 
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+')
    parser.add_argument('--min_lr', default=1e-5, type=float, help="this lr is used in scheduler")
    parser.add_argument('--warmup_epoch', default=5, type=int)
    parser.add_argument('--warmup_steps', default=-1, type=int)
    parser.add_argument('--add_discriminator', default=True, type=bool)


    # loss parameters 
    parser.add_argument('--perceptual_weight', default=1.0, type=float)
    parser.add_argument('--pixelloss_weight', default=1.0, type=float)
    parser.add_argument('--logvar_init', default=0.0, type=float)
    parser.add_argument('--kl_weight', default=1e-6, type=float)
    parser.add_argument('--disc_factor', default=1.0, type=float)
    parser.add_argument('--disc_start', default=0, type=int)
    parser.add_argument('--disc_weight', default=0.1, type=float)

    return parser.parse_args()
    
