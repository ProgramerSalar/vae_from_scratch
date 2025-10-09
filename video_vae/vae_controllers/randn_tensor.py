import torch 
from torch import nn 
from typing import Union, Tuple, List

def randn_tensor(
        shape: Union[Tuple, List],
        generator: Union[List["torch.Generator"], "torch.Generator"] = None,
        device: "torch.device" = None,
        dtype: "torch.dtype" = None,
        layout: "torch.layout" = None
    ):

    """ 
        A helper function to create a random tensor on the desire of `device` with desire of `dtype`. 
        when passing a list of generator. you can see bach size indivisually.
        If cpu generator are passed the tensor is always created on the CPU.
    """

    # device 
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    assert generator is not None, "make sure generator is not None."

    if generator is not None:
        gan_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gan_device_type != device.type and gan_device_type == "cpu":
            rand_device = "cpu"
            assert device != "mps", "make sure device is not map, only for cpu"

        else:
            assert gan_device_type != device.type and gan_device_type == "cuda", ValueError(f"can't generate a {device} tensor from generator of type {gan_device_type}")


    

    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(size=shape,
                        generator=generator[i],
                        device=rand_device,
                        dtype=dtype,
                        layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)


    else:
        latents = torch.randn(size=shape,
                              generator=generator,
                              device=rand_device,
                              dtype=dtype,
                              layout=layout).to(device)
        
    return latents


