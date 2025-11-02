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
  File "/content/vae_from_scratch/video_vae/../../vae_from_scratch/video_vae/train/main.py", line 82, in <module>
    main(args)
  File "/content/vae_from_scratch/video_vae/../../vae_from_scratch/video_vae/train/main.py", line 62, in main
    train_one_epoch(args,
  File "/content/vae_from_scratch/video_vae/train/ddp.py", line 88, in train_one_epoch
    disc_grad_norm = loss_scaler_disc(gan_loss, optimizer_disc, parameters=model.loss.discriminator.parameters())
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/vae_from_scratch/video_vae/../../vae_from_scratch/video_vae/middleware/native_scaler.py", line 17, in __call__
    self._scaler.scale(loss).backward(create_graph=create_graph)
  File "/usr/local/lib/python3.12/dist-packages/torch/_tensor.py", line 647, in backward
    torch.autograd.backward(
  File "/usr/local/lib/python3.12/dist-packages/torch/autograd/__init__.py", line 354, in backward
    _engine_run_backward(
  File "/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py", line 829, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
```