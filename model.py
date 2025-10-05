from vae.wrapper import CausalVideoVaeWrapper 



def build_model(args):

    model_dtype = args.model_dtype
    model = CausalVideoVaeWrapper()

    return model



