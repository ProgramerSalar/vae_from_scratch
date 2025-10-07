import torch 


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
                    ooutput = input_tensor * ther_arg
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
    
    
    # 1. bypass the function if kernel_size = 1 
    if kernel_size == 1:
        return input_
    
    # [2, 3, 8, 256, 256] -> [8, 3, 2, 256, 256]
    input_ = input_.transpose(0, dim)

    # 0 = 0 [working...]
    # torch.Size([8, 3, 2, 256, 256] -> torch.Size([10, 3, 2, 256, 256]
    # only first row of matrix convert to zeros bcz add the padding first row of matrix tensor.
    input_ = torch.zeros_like(input_[:1])
    input_ = torch.cat([input_] * (kernel_size - 1) + [input_],
                       dim=0)
    
    
    # torch.Size([10, 3, 2, 256, 256] -> torch.Size([2, 3, 10, 256, 256])
    input_ = input_.transpose(0, dim).contiguous()

    return input_
    





def _drop_from_previous_rank(input_,
                             dim,
                             kernel_size):
    
    # torch.Size([2, 3, 10, 256, 256]) ->  torch.Size([10, 3, 2, 256, 256]
    input_ = input_.transpose(0, dim)
    # torch.Size([10, 3, 2, 256, 256] -> torch.Size([8, 3, 2, 256, 256]
    input_ = input_[kernel_size - 1 :]
    # torch.Size([8, 3, 2, 256, 256] -> torch.Size([2, 3, 8, 256, 256]
    input_ = input_.transpose(0, dim)

    return input_




