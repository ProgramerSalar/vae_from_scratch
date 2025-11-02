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
    RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3, 128, 3, 3, 3]] is at version 3; expected version 1 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
```