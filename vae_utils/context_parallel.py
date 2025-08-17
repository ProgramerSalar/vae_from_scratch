import torch 
import torch.distributed as dist

_CONTEXT_PARALLEL_GROUP = 2     # init `None`
# This variable stores a number (an integer) that dictates how many GPUs will work together to process a single long input sequence (or "context").
_CONTEXT_PARALLEL_SIZE = 2  # init `None`

# the context=2 (group of GPU) so means => (2 GPU per group)
# how many context are initialized to parallelly
def is_context_parallel_initialized():

    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    
    else:
        return True
    
# so this function are check distribution are available or not 
def is_dist_avail_and_initialized():
    
    if not dist.is_available():
        return False 
    
    if not dist.is_initialized():
        return False
    
    return True


def get_rank():

    if not is_dist_avail_and_initialized():
        return 0 
    
    # if available the distribution then return world_size = `1`
    return dist.get_world_size()


def get_context_parallel_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized."

    rank = get_rank()
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE

    return cp_rank


def get_context_parallel_group():

    assert _CONTEXT_PARALLEL_GROUP is not None, "make sure context_parallel_group is initialized."

    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_group_rank():

    assert _CONTEXT_PARALLEL_SIZE is not None, "make sure context parallel size is not None"

    rank = get_rank()
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank


def get_context_parallel_world_size():

    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized."

    return _CONTEXT_PARALLEL_SIZE




def _cp_pass_from_previous_rank(input_,
                                dim,
                                kernel_size):
    
    """ 
        input_ : The input tensor (a multi-dimensinal array of data, common in deep learning)
        dim: The dimension along which the operation (and data transfer) should occur.
        kernel_size: The likely refers the size of a kernel of context window, indicating 
            how much data needs to be received from the previous rank.

        Pupose:
            Retrives data from the previous rank in the context-parallel group and prepends it to the local `input_`
            tensor. The provides the necessary "context" for operations (e.g. convolutions) that require data from 
            neighboring partitions.

        Workflow:
            1. skip if `kernel_size = 1` (no context needed)
            2. Transpose the target `dim` to position `0` from easier manipulation
            3. Determine comunication ranks:
                - `send_rank`: Rank that recieves data from the current rank (next rank in the group)
                - `recv_rank`: Rank that sends data to the current rank (previous rank in the group)
                - Handle boundary cases (e.g. First/last rank in the group)

            4. Asynchronouns communication:
                - Non-last rank: send the last `kernel_size - 1` elements to `send_rank`.
                - Non-first rank: Recieve  `kernel_size - 1` elements from `recv_rank` into `recv_buffer`.

            5. combine data:
                - First rank: Prepend (kernel_size - 1) zeros.
                - other rank: Prepend recevied `recv_buffer` to `input_`.

            6. Restore original dimensions and return the tensor.

    """

    
     
     # Bypass the function if kernel size is 1 
    if kernel_size == 1:
        return input_
     
    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    print(f"In _pass_from_previous_rank, cp_rank: {cp_rank}, input_size: {input_.shape}")
     
    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    # print(f'what is the global_rank: {global_rank} and global_world_size: {global_world_size}')

    input_ = input_.transpose(0, dim)
    # print(f"what is the input: {input_}")

    # pass from last rank 
    send_rank = global_rank + 1 
    recv_rank = global_rank - 1 

    if send_rank % cp_world_size == 0:
        send_rank -= cp_group_rank

    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size

    recv_buffer = torch.empty_like(input=input_[-kernel_size + 1:]).contiguous()
    
    if cp_rank < cp_world_size -1:
        req_send = torch.distributed.isend(tensor=input_[-kernel_size + 1:].contiguous(),
                                           dst=send_rank,
                                           group=group)
        
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(tensor=recv_buffer,
                                           src=recv_rank,
                                           group=group)
        
    if cp_rank == 0:
        input_ = torch.cat([torch.zeros_like(input_[1:])] * (kernel_size - 1) + [input_], dim=0)

    else:
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)

    input_ = input_.transpose(0, dim).contiguous()

    return input_

    

    
def _drop_from_previous_rank(input_,
                             dim,
                             kernel_size):
    
    """ 
    Removes the extra `kernel_size-1` elements prepended during `_cp_pass_from_previous_rank`
    used in the backward pass to exclude gradients corresponding to the "borrowed" context.
    """

    # transpose the target dim to position 0, slice of the first kernel_size - 1 elements, restore original dim 
    input_ = input_.transpose(0, dim)[kernel_size - 1:].transpose(0, dim)
    return input_




class _CPConvolutionPassFromPreviousRank(torch.autograd.Function):

    """Integrates the context-passing logic with Pytorch autograd system for end-to-end training."""


    @staticmethod
    def forward(ctx,
                input_,
                dim,
                kernel_size):
        
        # print(f'what is the dim of context: {ctx.dim} and kernel_size: {ctx.kernel_size}')
        
        ctx.dim = dim 
        ctx.kernel_size = kernel_size
        return _cp_pass_from_previous_rank(input_=input_,
                                           dim=dim,
                                           kernel_size=kernel_size)
    


    @staticmethod
    def backward(ctx,
                 grad_outputs):
        
        return _drop_from_previous_rank(input_=grad_outputs,
                                        dim=ctx.dim,
                                        kernel_size=ctx.kernel_size), None, None







def cp_pass_from_previous_rank(input_,
                               dim,
                               kernel_size):
    
    """Public interface for the context-passing operation. Invokes the custom autograd function via `apply()`
    
    Usage Case:
        Imagine a conv with `kernel_size=3` applied to a tensor split along the sequence dimension
        (dim=1) across 2 ranks:

        - Rank 0: Holds [A1, A2, A3]
        - Rank 1: Holds [B1, B2, B3]

        After `cp_pass_from_previous_rank(dim=1, kernel_size=3)`
        - Rank 0: [0, 0, A1, A2, A3] (padding with zeros)
        - Rank 1: [A2, A3, B1, B2, B3] (padding with A2, A3 from Rank 0)

        Now both ranks have sufficient context to compute the conv locally
    
    """

    return _CPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size)



if __name__ == "__main__":

    
    # output = is_context_parallel_initialized()
    output = get_context_parallel_rank()
    print(output)



