import torch.distributed as dist 

_CONTEXT_PARALLEL_SIZE = None 
_CONTEXT_PARALLEL_GROUP = None 

# dist - distribution
def is_dist_avail_and_initialized():

    if not dist.is_available():
        return False 
    
    if not dist.is_initialized():
        return False
    
    return True



def is_context_parallel_intialized():

    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    else:
        return True
    
    
    


def get_rank():

    if not is_dist_avail_and_initialized():
        return 1 
    
    return dist.get_world_size()



def get_context_parallel_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    rank = get_rank()
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE

    return cp_rank



def get_context_parallel_group():

    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"

    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_group_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    rank = get_rank()
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank



def get_context_parallel_world_size():

    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    return _CONTEXT_PARALLEL_SIZE




