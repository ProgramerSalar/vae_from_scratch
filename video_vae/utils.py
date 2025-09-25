import os, torch
from datetime import datetime, timedelta

def init_distributed_mode(args,
                          init_pytorch_ddp=True):
    
    if int(os.getenv('OMPI_COMM_WORLD_SIZE', '0')) > 0:
        print('work in progress...')
        
        
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print('else condition is working...')
        
    else:
        print('Not using distributed mode')
        args.distributed = False 


    args.distributed = True 
    args.dist_backend = 'nccl'
    args.dist_url = "env://"
    print(f"| distributed init (rank {args.local_rank}): {args.dist_url}, gpu: {args.device}",
          flush=True)
    

    if init_pytorch_ddp:
        # Init DDP Group, for script without using accelerate framework.
        torch.cuda.set_device(args.device)
        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world,
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





# if __name__ == "__main__":
    
#     out = init_distributed_mode(args=get_args)
