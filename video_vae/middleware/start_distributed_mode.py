import torch, os 
from datetime import timedelta


def init_distributed_mode(args,
                          init_pytorch_ddp=True):
    

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

        print(f"rank: {args.rank}, world_size: {args.world_size}, gpu: {args.gpu}")

    else:
        SystemError("Distributed mode if not initialized.")


    



