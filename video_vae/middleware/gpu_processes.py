import torch 

# how many gpu work in parallel like
_CONTEXT_PARALLEL_SIZE = None
_CONTEXT_PARALLEL_GROUP = None



def is_context_parallel_initialized():

    """Filter the context_parallel_size of the gpu."""
    
    if _CONTEXT_PARALLEL_SIZE is None:
        return False
    
    else:
        return True
    


def get_context_parallel_rank():

    """
        Fetch the context rank of the gpu
        
        Imagine you have 8 GPUs in total for training, and you decide to create context parallel groups of size 4.
        get_rank(): This will return the global, unique ID for each GPU, from 0 to 7.
        _CONTEXT_PARALLEL_SIZE: This would be 4.

        The calculation cp_rank = rank % 4 would work as follows:

        Group 1
        GPU with global rank = 0: 0 % 4 = 0. Its cp_rank (local rank) is 0.
        GPU with global rank = 1: 1 % 4 = 1. Its cp_rank is 1.
        GPU with global rank = 2: 2 % 4 = 2. Its cp_rank is 2.
        GPU with global rank = 3: 3 % 4 = 3. Its cp_rank is 3.

        Group 2
        GPU with global rank = 4: 4 % 4 = 0. Its cp_rank is 0.
        GPU with global rank = 5: 5 % 4 = 1. Its cp_rank is 1.
        GPU with global rank = 6: 6 % 4 = 2. Its cp_rank is 2.
        GPU with global rank = 7: 7 % 4 = 3. Its cp_rank is 3.
    """

    assert _CONTEXT_PARALLEL_SIZE is not None, "make sure world_size of the gpu does not None."

    rank = get_rank()
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE     

    return cp_rank


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



def get_context_parallel_group():

    """filter the context parallel group"""

    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized."

    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_group_rank():

    """calculate the context parallel group rank."""

    assert _CONTEXT_PARALLEL_SIZE is not None, "context parllel size is not None."

    rank = get_rank()
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank


def get_context_parallel_world_size():

    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not None."

    return _CONTEXT_PARALLEL_SIZE