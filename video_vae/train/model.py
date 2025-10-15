from vae.wrapper import CausalVideoVAELossWrapper



def build_model(args):

    model_dtype = args.model_dtype
    model = CausalVideoVAELossWrapper()

    return model



