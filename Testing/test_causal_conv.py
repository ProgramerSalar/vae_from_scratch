import torch 
from torch import nn 
from typing import Union, Tuple
from collections import deque
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist 



# will store the size of each context parallel group 
_CONTEXT_PARALLEL_SIZE = None 
# will store the process group for context parallelism
_CONTEXT_PARALLEL_GROUP = None 




# checks if distributed training is available and initialized 
def is_dist_avail_and_initialized():

    # check if PyTorch distribution package is available 
    if not torch.distributed.is_available():
        return False 
    
    # check if distributed environment is initialized
    if not torch.distributed.is_initialized():
        return False
    
    return True



# checks if context parallelism is set up
def is_context_parallel_intialized():

    # context group is not created
    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    
    # context group exists
    else:
        return True
    
    
    

# this has a logical error (return wold size, not rank)
def get_rank():

    # Non-distribution case 
    if not is_dist_avail_and_initialized():
        return 0 
    
    # return total process (e.g. 8 GPUs), not current rank 
    return torch.distributed.get_world_size()   



# Get rank within the context parallel group
def get_context_parallel_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    # Uses flawed get_rank() (returns world size)
    rank = get_rank()
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE     # Local rank in context group

    return cp_rank


# Returns the context parallel process group
def get_context_parallel_group():

    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"

    # returns the process group
    return _CONTEXT_PARALLEL_GROUP



# Get index of the context parallel group
def get_context_parallel_group_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    rank = get_rank()   # Uses flawed get_rank() (Returns world size)
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE  # Group index (e.g. 0, 1, 2...)

    return cp_group_rank



# Returns size of context parallel groups
def get_context_parallel_world_size():

    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    return _CONTEXT_PARALLEL_SIZE   # value set during initialization





# ---------------------------------------------------------------------------------

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
    

    # Get context parallel group information
    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    print(f"in _pass_from_previous_rank, cp_rank: {cp_rank}, input_size: {input_.shape}")

    # Get global distributed ranks 
    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    # Transpose target dimension to 0 for easier manipulation
    input_ = torch.transpose(input=input_,
                             dim0=0,
                             dim1=dim)

    # calculate naive neighbor ranks (will be adjusted)
    send_rank = global_rank + 1 
    recv_rank = global_rank - 1 

    # Adjust send rank at context group boundaries 
    # prevent sending to next context group 
    if send_rank % cp_world_size == 0:
        send_rank = send_rank - cp_group_rank

    # Adjust receive rank at context group boundaries
    # Prevents receiving from previous context group
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank = recv_rank + cp_world_size


    # create buffer for incomimg data (matches last k-1 elements)
    recv_buffer = torch.empty_like(input=input_[-kernel_size + 1 :]).contiguous()
    
    # Non-last ranks: Send last k-1 elements to next rank 
    # Uses non-blocking send 
    if cp_rank < cp_world_size -1:
        req_send = torch.distributed.isend(tensor=input_[-kernel_size + 1 :].contiguous(), 
                                           dst=send_rank, 
                                           group=group)

    # 
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(tensor=recv_buffer,
                                           src=recv_rank,
                                           group=group)
        
    

    if cp_rank == 0:
        input_ = torch.cat([torch.zeros_like(input_[:1])] * (kernel_size - 1) + [input_], dim=0)

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
    
    # transpose the target dim to position 0, slice of the first kernel_size -1 elements, restore original dim 
    input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim)
    return input_



class _CPConvolutionPassFromPreviousRank(torch.autograd.Function):

    """Integrates the context-passing logic with Pytorch autograd system for end-to-end training."""

    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        
        ctx.dim = dim 
        ctx.kernel_size = kernel_size
        return _cp_pass_from_previous_rank(input_=input_,
                                           dim=dim,
                                           kernel_size=kernel_size)
    

    @staticmethod
    def backward(ctx, grad_outputs):
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



# -----------------------------------------------------------------------

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0 

def is_odd(n):
    return not divisible_by(n, 2)

class CausalConv3d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = 1,
            pad_mode: str = 'constant',
            **kwargs
    ):
        
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = cast_tuple(kernel_size, 3)
        print(f"what is the kernel_size : {kernel_size}")

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop('dilation', 1)
        self.pad_mode = pad_mode

        if isinstance(stride, int):
            stride = (stride, 1, 1)

        time_pad = dilation * (time_kernel_size - 1)
        height_pad = height_kernel_size // 2 
        width_pad = width_kernel_size // 2 

        self.temporal_stride = stride[0]
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              dilation=dilation,
                              **kwargs)
        
        self.cache_front_feat = deque()


    def __clear_context_parallel_cache(self):
        del self.cache_front_feat
        self.cache_front_feat = deque()


    


    def context_parallel_forward(self, x):

        """ 
            context parallelism is technique used in distributed training of deep learning models
            where different parts of the input context (in this case, likely the temporal dimension of a video or sequence)
            are processed on different devices or ranks.
        """

        cp_rank = get_context_parallel_rank()

        if self.time_kernel_size == 3 and ((cp_rank == 0 and x.shape[2] <= 2) or cp_rank != 0 and x.shape[2] <= 1):

            # This code is only for training 8 frames per GPU (except for cp_rank=0, 9 frames) with context parallel 
            # if you do not have enought GPU memory, you can set the total frames = 8 * CONTEXT_SIZE + 1, enable each GPU 
            # only forward 8 frames during training

            x = cp_pass_from_previous_rank(input_=x,
                                           dim=2,
                                           kernel_size=2)
            
            
            
            trans_x = cp_pass_from_previous_rank(input_=x[:, :, :-1],
                                                 dim=2,
                                                 kernel_size=2)
            x = torch.cat([trans_x, x[:, :, -1:]], dim=2)


        else:
            x = cp_pass_from_previous_rank(input_=x,
                                           dim=2,
                                           kernel_size=self.time_kernel_size)
            
            
            

        x = F.pad(x, self.time_uncausal_padding, mode='constant')

        if cp_rank != 0:
            if self.temporal_stride == 2 and self.time_kernel_size == 3:
                x = x[:, :, 1:]

            
        x = self.conv(x)
        return x 
    


    def forward(self, 
                x, 
                is_init_image=True,
                temporal_chunk=False):
        

        if is_context_parallel_intialized():
            return self.context_parallel_forward(x)
        
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        if not temporal_chunk:
            x = F.pad(x, self.time_causal_padding, mode=pad_mode)

        else:
            assert not self.training, "The feature cache should not be used in training"
            if is_init_image:
                # Encode the first chunk 
                x = F.pad(x, self.time_causal_padding, mode=pad_mode)
                self.__clear_context_parallel_cache()
                self.cache_front_feat.append(x[:, :, -2:].clone().detach())

            else:
                x = F.pad(x, self.time_uncausal_padding, mode=pad_mode)
                video_format_context = self.cache_front_feat.pop()
                self.__clear_context_parallel_cache()


                if self.temporal_stride == 1 and self.time_kernel_size == 3:
                    x = torch.cat([video_format_context, x], dim=2)
                elif self.temporal_stride == 2 and self.time_kernel_size == 3:
                    x = torch.cat([video_format_context[:, :, -1:], x], dim=2)


                self.cache_front_feat.append(x[:, :, -2:].clone().detach())


        x = self.conv(x)
        return x 
    



    




    



if __name__ == "__main__":

    # causasl_conv_3d = CausalConv3d(in_channels=3,
    #                                out_channels=3,
    #                                kernel_size=3,
    #                                stride=1)
    # # print(causasl_conv_3d)

    # x = torch.randn(2, 3, 8, 64, 64)
    # output = causasl_conv_3d(x)
    # print(output.shape)

    pass