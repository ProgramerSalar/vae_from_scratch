import torch 

import sys 
sys.path.append('/content/vae_from_scratch/video_vae')

from args import get_args
from middleware.start_distributed_mode import init_distributed_mode
from middleware.multiple_gpus_cp_ops import context_parallel_pass_from_previous_rank
from middleware.gpu_processes import initialized_context_parallel


def test_train(args):

    # start the distributed mode
    init_distributed_mode(args=args)

    if args.use_context_parallel:
        initialized_context_parallel(context_parallel_size=args.context_size)

    x = torch.randn(2, 3, 8, 256, 256)
    out = context_parallel_pass_from_previous_rank(input_=x,
                                                   dim=2,
                                                   kernel_size=3)
    
    


if __name__ == "__main__":
    args = get_args()
    out = test_train(args=args)

