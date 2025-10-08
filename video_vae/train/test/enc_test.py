import torch 
import sys 
sys.path.append('/content/vae_from_scratch/video_vae')

from train.args import get_args
from middleware.start_distributed_mode import init_distributed_mode
from middleware.gpu_processes import initialized_context_parallel
from vae.enc_dec import CausalEncoder


def test_train(args):

    # start the distributed mode
    init_distributed_mode(args=args)

    if args.use_context_parallel:
        initialized_context_parallel(context_parallel_size=args.context_size)

    x = torch.randn(2, 3, 8, 256, 256).to("cuda:0").half()
    model = CausalEncoder(in_channels=3,
                          out_channels=3).to("cuda:0").half()
    
    model.gradient_checkpointing = True
    model.train(mode=True)
    
    
    out = model(x)
    print(out.shape)    # torch.Size([2, 6, 1, 16, 16])
    
    
    


if __name__ == "__main__":
    args = get_args()
    out = test_train(args=args)

