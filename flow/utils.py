import torch.distributed as dist 
import torch

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
        return 0
    
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




def initialize_context_parallel(context_parallel_size):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE

    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    _CONTEXT_PARALLEL_SIZE = context_parallel_size


    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for i in range(0, world_size, context_parallel_size):
        ranks = range(i, i + context_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            break


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1 
    
    return dist.get_world_size()

