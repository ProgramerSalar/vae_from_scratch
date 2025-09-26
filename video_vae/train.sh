

GPUS=1
LPIPS_CKPT=video_vae/vgg_lpips.pth
OUTPUT_DIR=/PATH/output_dir
IMAGE_ANNO=annotation/image_text.jsonl
VIDEO_ANNO=annotation/video_text.jsonl
RESOLUTION=256
MAX_FRAMES=17
BATCH_SIZE=2


torchrun --nproc_per_node $GPUS \
        video_vae/train.py \
        --num_workers 2 \
        --model_dtype bf16 \
        --lpips_ckpt $LPIPS_CKPT \
        --output_dir $OUTPUT_DIR \
        --image_anno $IMAGE_ANNO \
        --video_anno $VIDEO_ANNO \
        --use_image_video_mixed_training \
        --image_mix_ratio 0.1 \
        --resolution $RESOLUTION \
        --max_frames $MAX_FRAMES \
        --disc_start 250000 \
        --kl_weight 1e-12 \
        --pixelloss_weight 10.0 \
        --disc_weight 0.5 \
        --batch_size $BATCH_SIZE \
        --opt_betas 0.9 0.95 \
        --seed 42 \
        --weight_decay 1e-3 \
        --clip_grad 1.0 \
        --lr 1e-4 \
        --lr_disc 1e-4 \
        --warmup_epochs 1 \
        --iters_per_epoch 2000 \
        --print_freq 40 \
        --save_ckpt_freq 1 \
        --epochs 100


