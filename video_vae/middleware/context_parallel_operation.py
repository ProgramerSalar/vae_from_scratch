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
    


    return 


class _context_parallel_conv_pass_from_prev_rank(torch.autograd.Function):

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

    @staticmethod
    def forward(ctx, input_, dim, kernel_size):

        ctx.dim = dim 
        ctx.kernel_size = kernel_size

        return _cp_pass_from_previous_rank(input_=input_,
                                           dim=dim,
                                           kernel_size=kernel_size)
    


def _cp_pass_from_previous_rank(input_, 
                                dim,
                                kernel_size):
    
    

    # bypass the function if kernel_size = 1 
    if kernel_size == 1:
        return input_
    
    group = get_context_parallel_group()    # group of the gpu
    cp_rank = get_context_parallel_rank()   # rank of the gpu
    cp_group_rank = get_context_parallel_group_rank()   #  group rank of the gpu
    cp_world_size = get_context_parallel_world_size()   #  world size of the gpu

    


        


