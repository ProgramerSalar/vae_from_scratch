import torch 

from vae.causal_vae import CausalVAE


def train_one_epoch(
                model,
                lr_schedule_value=None,
            ):
    

    if lr_schedule_value is not None:
        pass 



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CausalVAE(num_groups=1).to(device)


    lr_scheduler_values = cosine_scheduler()




    one_train = train_one_epoch(model=model,
                                lr_schedule_value=)