import torch 


_CONTEXT_PARALLEL_GROUP = 2     # init `None`


# the context=2 (group of GPU) so means => (2 GPU per group)
# how many context are initialized to parallelly
def is_context_parallel_initialized():

    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    
    else:
        return True
    


if __name__ == "__main__":

    
    output = is_context_parallel_initialized()
    print(output)



