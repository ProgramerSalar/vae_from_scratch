import torch 
import torch.distributed as dist 
import os 


def setup(rank, world_size):
    """ 
    Initializes the distributed environment.
    """

    # set the MASTER_ADDR and MASTER_PORT environment variables 
    # This is a common way to set up the communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group with a backend (e.g. 'gloo' for CPU)
    dist.init_process_group(backend='gloo',
                            rank=rank,
                            world_size=world_size)
    

def cleanup():

    """Destroys the distributed process group."""

    dist.destroy_process_group()


def run_distributed_job(rank,
                        world_size):
    
    print(f"Rank {rank}: Checking if distributed is initialized...")
    # This will be False before the setup function is called 
    print(f"Rank {rank}: Initialized status before setup: {dist.is_initialized()}")

    # Now that the process group is initialized, we can perform distributed operation:
    # for example, a simple All-Reduce to sum tensors across all processes 
    tensor = torch.tensor([float(rank)]) # Each process has a different value 
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank}: After all_reduce, the tensor value is {tensor.item()}")

    cleanup()
    print(f"Rank {rank}: Initialized status after cleanup: {dist.is_initialized()}")



if __name__ == "__main__":


    world_size = 2 
    print("A proper run would look like: `torchrun --nproc_per_node=2 your_script.py`")

    try:
        dist.init_process_group('gloo', rank=0, world_size=1)
        print("Example with a single process:")
        print(f"Is distributed initialized ? {dist.is_initialized()} \n")
        dist.destroy_process_group()
        print(f"Is distributed initialized after cleanup ? {dist.is_initialized()}")

    except Exception as e:
        print(e)
        
    



