

if __name__ == "__main__":
    from diffusers.models.attention_processor import SpatialNorm
    import torch 


    num_channels = 64
    zq_channels = 8

    spatial_norm = SpatialNorm(f_channels=num_channels,
                            zq_channels=zq_channels)

    print(spatial_norm)

    x = torch.randn(2, 3, 64, 64)
    temb = torch.randn(2, 128)

    spatial_norm = spatial_norm(f=x, zq=temb)
    spatial_norm



