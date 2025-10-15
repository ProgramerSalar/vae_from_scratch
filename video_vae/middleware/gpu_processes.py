import torch 

# how many gpu work in parallel like
_CONTEXT_PARALLEL_SIZE = None




def is_context_parallel_initialized():

    """Filter the context_parallel_size of the gpu."""
    
    if _CONTEXT_PARALLEL_SIZE is None:
        return False
    
    else:
        return True
    




def get_rank():

    """Fetch the rank of the GPU."""

    if not is_distribute_avail_and_initialized():
        return 0
    
    # fetch the world_size in distribution
    return torch.distributed.get_world_size()



def is_distribute_avail_and_initialized():

    """Filter the Distributed available or intialized."""

    # check the gpu are found or not
    if not torch.distributed.is_available():
        return False
    
    # check the env gpu are found or not
    if not torch.distributed.is_initialized():
        return False
    
    return True


