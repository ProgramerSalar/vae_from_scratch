GPUS=1

# testing 
torchrun --nproc_per_mode $GPUS \
        video_vae/middleware/context_parallel_operation.py