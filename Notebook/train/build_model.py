from flow.video_vae.causal_video_vae_wrapper import CausalVideoVAELossWrapper



def build_model(args):

    model_dtype = args.model_dtype 
    model_path = args.model_path

    print(f"Load the base videoVAE checkpoint from path: {model_path} using dtype: {model_dtype}")

    model = CausalVideoVAELossWrapper(
        model_path=model_path,
        model_dtype='fp32',
        disc_start=args.disc_start,
        logvar_init=args.logvar_init,
        kl_weight=args.kl_weight,
        pixelloss_weight=args.pixelloss_weight,
        perceptual_weight=args.perceptual_weight,
        disc_weight=args.disc_weight,
        interpolate=False,
        add_discriminator=args.add_discriminator,
        freeze_encoder=args.freeze_encoder,
        load_loss_module=True,
        lpips_ckpt=args.lpips_ckpt
    )

    # if args.pretrained_vae_weight:
    #     pretrained_vae_weight = args.pretrained_vae_weight 
    #     print(f"Loading the vae checkpoint from {pretrained_vae_weight}")
    #     model.load_checkpoint(pretrained_vae_weight)

    return model