import torch 
from middleware.init_dist_mode import init_distributed_mode


def main(args):

    init_distributed_mode(args)

    # if enabled, distributed multiple video clips to different divices.
    if args.use_context_parallel:
        



