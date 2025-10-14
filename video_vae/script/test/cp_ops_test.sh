GPUS=1

# testing context parallel operation 
torchrun --nproc_per_node $GPUS \
        /content/vae_from_scratch/video_vae/train/test/cp_ops_test.py \
        --use_context_parallel \
        --context_size 1