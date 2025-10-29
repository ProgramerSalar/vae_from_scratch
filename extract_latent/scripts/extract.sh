GPUS=1
ANNO_FILE=/home/manish/Desktop/projects/vae_from_scratch/annotation/video_text.jsonl
torchrun --nproc_per_node $GPUS \
    /home/manish/Desktop/projects/vae_from_scratch/extract_latent/extract_video_latent.py \
    --model_dtype bf16 \
    --batch_size 1 \
    --anno_file $ANNO_FILE \
    --width 640 \
    --height 384 \
    --num_frames 121 \
    


