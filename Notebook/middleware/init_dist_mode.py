import torch, os 
from datetime import timedelta




def init_distributed_mode(args,
                          init_pytorch_ddp=True):
    
    if int(os.getenv('OMPI_COMM_WORLD_SIZE', '0')) > 0:

        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

        os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ["RANK"] = os.environ['OMPI_COMM_WORLD_RANK']
        os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])


    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])


    else:
        print('Not using distributed mode.')
        args.distributed = False
        return None
    
    args.distributed = True
    args.dist_backend = 'nccl'
    args.dist_url = "env://"
    print(f'| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}', 
          flush=True)
    
    # ddp -> distributed  data parallel
    if init_pytorch_ddp:
        # Init DDP Group, for script without using accelerate framework
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=args.rank,
                                             timeout=timedelta(days=365))
        torch.distributed.barrier()

        setup_for_distributed(args.rank == 0)


    
def setup_for_distributed(is_master):
    
    """This function disables printing when not in master process."""

    import builtins as __builtin__ 
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print









    
