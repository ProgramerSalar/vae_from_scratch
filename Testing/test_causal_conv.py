import sys
import os
from pathlib import Path

# --- Add project root to sys.path ---
current_dir = Path(__file__).resolve().parent  # /tools directory
project_root = current_dir.parent  # Project root (parent of /tools)
sys.path.append(str(project_root))
# -----------------------------------


import torch
import torch.distributed as dist
from utils import is_context_parallel_intialized as initialize_context_parallel

# Initialize distributed environment (for example, using 'nccl' backend)
dist.init_process_group(backend='nccl')

# Initialize context parallel group with your desired size (e.g., 2)
initialize_context_parallel(context_parallel_size=2)


from video_vae.modeling_causal_conv import CausalConv3d

# Example input: [batch, channels, time, height, width]
x = torch.randn(4, 3, 9, 32, 32).cuda()  # Use .cuda() for GPU tensors

conv = CausalConv3d(
    in_channels=3,
    out_channels=8,
    kernel_size=(3, 3, 3)
).cuda()

# This will call context_parallel_forward if context parallel is initialized
output = conv(x)

print(output.shape)

# torchrun --nproc_per_node=2 Testing/test_causal_conv.py