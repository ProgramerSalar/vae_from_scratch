GPUS=1

# testing 
torchrun --nproc_per_node $GPUS \
        video_vae/middleware/context_parallel_operation.py \
        --use_context_parallel \
        --context_size 1