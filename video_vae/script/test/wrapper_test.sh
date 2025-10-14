GPUS=1

# testing 
torchrun --nproc_per_node $GPUS \
        /content/vae_from_scratch/video_vae/train/test/wrapper_test.py \
        --use_context_parallel \
        --context_size 1        # what is the group of your gpu