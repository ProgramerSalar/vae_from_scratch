GPUS=1

# testing 
torchrun --nproc_per_node $GPUS \
        /content/vae_from_scratch/video_vae/train/test/conv_test.py \
        --use_context_parallel \
        --context_size 1