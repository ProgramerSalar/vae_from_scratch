import torch 


from middleware.gpu_processes import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_group_rank,
    get_context_parallel_world_size
)


def context_parallel_pass_from_previous_rank(input_,
                                             dim,
                                             kernel_size):
    


    return _context_parallel_conv_pass_from_prev_rank.apply(input_, dim, kernel_size)


class _context_parallel_conv_pass_from_prev_rank(torch.autograd.Function):

    

    @staticmethod
    def forward(ctx, input_, dim, kernel_size):

        """
            This method defines the forward pass of your custom operation. it takes a `ctx` object 
            (context) as its first argument. followed by the `input_tensors` and any `other` necessary 
            argrument. 
            it should return the output tensor of the operation. The `ctx` object can be used to store 
            tensors or `other data` that will be needed during backward pass using `ctx.save_for_backward()`

            ```
                @staticmethod
                def forward(ctx, input_tensor, other_arg):
                    # Perform the forward computation
                    output = input_tensor * other_arg
                    # Save tensors needed for backward pass
                    ctx.save_for_backward(input_tensor, other_arg)
                    return output
            ```
        """

        ctx.dim = dim 
        ctx.kernel_size = kernel_size

        return _cp_pass_from_previous_rank(input_=input_,
                                           dim=dim,
                                           kernel_size=kernel_size)
    

    

    @staticmethod
    def backward(ctx, *grad_outputs):

        """
            This method defines the backward pass (gradient calculation) of your custom 
            operation. It also takes a `ctx` object as its first argument followed by the gradient of 
            the outputs. It should return the gradient with respect to each input tensor, in the same 
            order as they were passed to the `forward` method. you can recieve saved tensor from using 
            `ctx.saved_tensors`

            ```
                @staticmethod
                def backward(ctx, grad_output):
                    # Retrieve saved tensors
                    input_tensor, other_arg = ctx.saved_tensors
                    # Calculate gradients for inputs
                    grad_input = grad_output * other_arg
                    grad_other_arg = grad_output * input_tensor
                    return grad_input, grad_other_arg
            ```
        """
        

        return _drop_from_previous_rank(input_=grad_outputs,
                                        dim=ctx.dim,
                                        kernel_size=ctx.kernel_size), None, None
    

        
    


    

def _cp_pass_from_previous_rank(input_, 
                                dim,
                                kernel_size):
    
    
    """ 
        Args:
            input_: The input tensor (a multi-dimensional array of data, common in deep learning) like: [32, 3, 8, 256, 256]
            dim: The dimension along with the operation (and data transfer)
            kernel_size: The likely refers the size of a kernel of context window, indicating 
                how much data needs to be received from the previous rank.
        
        PURPOSE:
            how to stitch data back together across multiple GPUs before performing a calculation.
            Imagine you've split a long video clip into chunks and given one chunk to each of your GPUs to process.

            Now, imagine you're using a convolution, which works like a sliding window. 
            When the sliding window reaches the very beginning of a chunk on GPU #2, 
            it needs to see the last few frames from the end of the chunk on GPU #1 to do its job correctly.

        Workflow:
            1. skip if `kernel_size=1` (no context needed)
            2. Transpose the target `dim` to position `0` from easier manipulation
            3. Determine communication ranks:
                - `send_rank`: Rank that recieves data from current rank (next rank in the group)
                - `recv_rank`: Rank that sends data to the current rank (previous rank in the group)
                - Handle boundary cases (e.g, First/last rank in the group)

            4. Asynchronouns communications
                - Non-last rank: send the last `kernel_size - 1` elements to `send_rank`.
                - Non-first rank: Receive `kernel_size - 1` elements to `recv_rank` into `recv_buffer`.

            5. combine data:
                - First rank: Prepend (kernel-size - 1) zeros.
                - other rank: Prepend recevied `recv_buffer` to `input_`

            6. Restore original dimensions and return the tensor.

        

    """
    
    

    # 1. bypass the function if kernel_size = 1 
    if kernel_size == 1:
        return input_
    
    group = get_context_parallel_group()    # 1
    cp_rank = get_context_parallel_rank()   # 0
    cp_group_rank = get_context_parallel_group_rank()   #  1
    cp_world_size = get_context_parallel_world_size()   #  1

    print(f"group: {group}, cp_rank: {cp_rank}, cp_group_rank: {cp_group_rank}, cp_world_size: {cp_world_size}")

    global_rank = torch.distributed.get_rank() # 0 
    global_world_size = torch.distributed.get_world_size()  # 1 

    # [8, 3, 2, 256, 256]
    input_ = input_.transpose(0, dim)

    send_rank = global_rank + 1  # 1
    recv_rank = global_rank - 1  # -1 
    
    # 1 % 1 == 0
    if send_rank % cp_world_size == 0:
      send_rank -= cp_group_rank  # 1 - 1 = 0

    # [-1 % 1 = 0] ==[ 1 - 1 = 0]
    if recv_rank % cp_world_size == cp_world_size - 1:
      recv_rank += cp_world_size #  -1 + 1 = 0

    # -3 + 1 = -2 [times, channels, batch, height, width]
    recv_buffer = torch.empty_like(input=input_[-kernel_size + 1:]).contiguous()

    # 0 < [1 -1 = 0] 
    # this conidtion has work when more than one gpu.
    if cp_rank < cp_world_size -1:
      req_send = torch.distributed.isend(tensor=input_[-kernel_size + 1:].contiguous(),
                                        dst=send_rank,
                                        group=group)

    # 0 > 0 
    # this conidtion has work when more than one gpu.
    if cp_rank > 0:
      req_recv = torch.distributed.irecv(tensor=recv_buffer,
                                          src=recv_rank,
                                          group=group)

    # 0 = 0 
    # torch.Size([8, 3, 2, 256, 256] -> torch.Size([10, 3, 2, 256, 256]
    if cp_rank == 0:
      input_ = torch.cat([torch.zeros_like(input_[:1])] * (kernel_size - 1) + [input_],
                        dim=0)  

    else:
      # this conidtion has work when more than one gpu.
      req_recv.wait()
      input_ = torch.cat([recv_buffer, input_], dim=0)

    # torch.Size([10, 3, 2, 256, 256] -> torch.Size([2, 3, 10, 256, 256])
    input_ = input_.transpose(0, dim).contiguous()

    return input_


    










def _drop_from_previous_rank(input_,
                             dim,
                             kernel_size):
    

    input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim)
    return input_




