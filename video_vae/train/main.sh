

GPUS=1


torchrun --nproc_per_node $GPUS \
    ../../vae_from_scratch/video_vae/train/main.py \
    --batch_size 4 \
    --opt adamw \
    --lr 1e-4 \
    --weight_decay 1e-3 \
    --opt_betas 0.9 0.95 \
    --perceptual_weight 1.0 \
    --pixelloss_weight 10.0 \
    --logvar_init 0.0 \
    --kl_weight 0.01 \
    --disc_start 250000 \
    --disc_weight 0.1 \
    --model_dtype fp32 \
    --add_discriminator True
