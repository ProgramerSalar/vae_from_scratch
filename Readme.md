# This repo to implement the VAE from scratch

```
main (file)
    | -> one_epoch_train
    | -> create_optimizer
    | -> NativebaseGradOptimizer
    | -> scheduler 

model (vae)
```

last-Error: 

```
    Traceback (most recent call last):
    File "/content/vae_from_scratch/video_vae/../../vae_from_scratch/video_vae/train/main.py", line 80, in <module>
        main(args)
    File "/content/vae_from_scratch/video_vae/../../vae_from_scratch/video_vae/train/main.py", line 60, in main
        train_one_epoch(args,
    File "/content/vae_from_scratch/video_vae/train/ddp.py", line 74, in train_one_epoch
        metric_logger.update(loss_scaler=loss_scaler_value)
                                        ^^^^^^^^^^^^^^^^^
    UnboundLocalError: cannot access local variable 'loss_scaler_value' where it is not associated with a value
```