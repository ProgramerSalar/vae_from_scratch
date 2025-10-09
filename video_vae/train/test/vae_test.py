import torch 
import sys 
sys.path.append('/content/vae_from_scratch/video_vae')

from train.args import get_args
from middleware.start_distributed_mode import init_distributed_mode
from middleware.gpu_processes import initialized_context_parallel
# from vae.enc_dec import CausalDecoder
# from vae.blocks import CausalUpperBlock
from vae.causal_vae import CausalVAE


def test_train(args):

    # start the distributed mode
    init_distributed_mode(args=args)

    if args.use_context_parallel:
        initialized_context_parallel(context_parallel_size=args.context_size)

    x = torch.randn(2, 3, 8, 256, 256).to("cuda:0").half()
    model = CausalVAE(norm_num_groups=2).to("cuda:0").half()
    out = model.encode(x)
    
    
    model.gradient_checkpointing = True
    model.train(mode=True)
    
    # out = model(x)
    print(out)
    
    
    


if __name__ == "__main__":
    args = get_args()
    out = test_train(args=args)

