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

    
    # activate the distributed mode 
    args.distributed = True 
    args.dist_backend = 'nccl'
    args.dist_url = "env://"

    # initialized the distributed data parallel
    if init_pytorch_ddp:
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=args.rank,
                                             timeout=timedelta(days=365))
        torch.distributed.barrier()
        







    



