import os 
import sys
import yaml
import time 
import torch 
import wandb 
from munch import munchify 
import torch.nn as nn 
from tqdm import tqdm 
import numpy as np 
import copy
from datetime import datetime
from argparse import ArgumentParser

from arguments import TrainingConfig, DatasetConfig
from noise_scheduler import LinearNoiseEscheduler
from utils import get_data, load_config
from transformer import Transformer

bold = f"\033[1m"
reset = f"\033[0m"

def mkdir_p(checkpoint_dir : str):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print(f'directory created: {checkpoint_dir}')
    return checkpoint_dir

def train_step(
        args      : TrainingConfig,
        model     : nn.Module,
        optimizer : torch.optim.Optimizer,
        loader    : torch.utils.data.DataLoader,
        epoch_idx : int,
        epochs    : int,
        loss_func : torch.nn.functional,
        scheduler : LinearNoiseEscheduler,
) -> float:

        model.train()

        step_loss     = 0
        n_samples     = 0

        prog_bar = tqdm(loader)
        status   = f'Train:\tEpoch: {epoch_idx}/{epochs}'
        prog_bar.set_description(status)

        for x, label in prog_bar:
                x, label = x.to(args.DEVICE), label.to(args.DEVICE)

                optimizer.zero_grad()

                noise = torch.randn_like(x).to(args.DEVICE)                                 # sample random noise 

                t = torch.randint(0, args.NUM_TIMESTEPS, (x.shape[0],)).to(args.DEVICE)     # sample timesteps 

                noisy_x = scheduler.add_noise(original=x, noise=noise, t=t)                 # add noise to images according to time-step  
                noise_pred = model(noisy_x, t)                                              # noise prediction by the net 

                loss   = loss_func(noise_pred, noise)                                       # compute loss
                loss.backward()                                                             
                optimizer.step()

                n_samples     += x.shape[0]                                                 
                step_loss     += loss.item() * x.shape[0]               

                step_status  = status + f'\tLoss: {float(step_loss/n_samples):.4f},'
                prog_bar.set_description(step_status)

        epoch_loss = float(step_loss / len(loader.dataset))

        prog_bar.close()

        return epoch_loss

def val_step(
        args      : TrainingConfig,
        model     : nn.Module,
        loader    : torch.utils.data.DataLoader,
        epoch_idx : int,
        epochs    : int,
        loss_func : torch.nn.functional,
        scheduler : LinearNoiseEscheduler,
) -> float:

        model.eval()

        step_loss     = 0
        n_samples     = 0

        prog_bar = tqdm(loader)
        status   = f'Train:\tEpoch: {epoch_idx}/{epochs}'
        prog_bar.set_description(status)

        for x, label in prog_bar:
                x, label = x.to(args.DEVICE), label.to(args.DEVICE)

                with torch.no_grad():
                    noise = torch.randn_like(x).to(args.DEVICE)                                 # sample random noise 
                    t = torch.randint(0, args.NUM_TIMESTEPS, (x.shape[0],)).to(args.DEVICE)     # sample timesteps 
                    noisy_x = scheduler.add_noise(original=x, noise=noise, t=t)                 # add noise to images according to time-step  
                    noise_pred = model(noisy_x, t)                                              # noise prediction by the net 

                loss   = loss_func(noise_pred, noise)                                           # compute loss

                n_samples     += x.shape[0]                                                 
                step_loss     += loss.item() * x.shape[0]               

                step_status  = status + f'\tLoss: {float(step_loss/n_samples):.4f},'
                prog_bar.set_description(step_status)

        epoch_loss = float(step_loss / len(loader.dataset))

        prog_bar.close()

        return epoch_loss

def main(
        args         : TrainingConfig,
) -> dict:

        scheduler = LinearNoiseEscheduler(                                      # create the noise scheduler 
                num_timesteps=args.NUM_TIMESTEPS, 
                beta_start=args.BETA_START, 
                beta_end=args.BETA_END
        )

        train_loader, val_loader = get_data(                                    # get the data 
                DatasetConfig.DATA_ROOT, 
                (DatasetConfig.HEIGHT, DatasetConfig.WIDTH), 
                TrainingConfig.BATCH_SIZE, 
                TrainingConfig.NUM_WORKERS
        )

        model = Transformer(                                                    # instantiate the model 
                channels=args.CHANNELS, 
                t_emb_dim=args.T_EMB_DIM, 
                num_heads=args.NUM_HEADS, 
                sequence_length=args.SEQUENCE_LENGTH,
                num_blocks=args.NUM_BLOCKS
        )

        ckpt_dir = mkdir_p(args.CKPT_DIR)                                       # create checkpoint model 

        if os.path.exists(os.path.join(args.CKPT_DIR, args.CKPT_NAME)):         # load checkpoint if found 
                print('Load checkpoint as found one')
                model.load_state_dict(torch.load(
                        os.path.join(args.CKPT_DIR, args.CKPT_NAME), map_location=args.DEVICE
                ))

        num_epochs = args.NUM_EPOCHS                                            # set training parameters 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
        loss_func = torch.nn.MSELoss()

        best_loss = torch.tensor(np.inf)

        epoch_train_loss = []
        epoch_val_loss   = []

        t_begin = time.time() # time measurement
        for epoch in range(num_epochs):

                train_loss = train_step(
                        args, model, optimizer, train_loader, epoch+1, args.NUM_EPOCHS, loss_func, scheduler
                )
                val_loss   = val_step(
                        args, model, val_loader, epoch+1, args.NUM_EPOCHS, loss_func, scheduler
                )

                train_loss_stat = f"{bold}Train Loss: {train_loss:.4f}{reset}"
                val_loss_stat = f"{bold}Val Loss: {val_loss:.4f}{reset}"

                print(f"\n{train_loss_stat:<30}")
                print(f"{val_loss_stat:<30}")

                epoch_train_loss.append(train_loss)
                epoch_val_loss.append(val_loss)

                # save the best model
                if val_loss < best_loss:
                        best_loss = val_loss
                        print(f"\nModel improved. Saving model...", end="")
                        best_weights = copy.deepcopy(model.state_dict())
                        formatted_datetime = datetime.now().strftime("%Y_%m_%d")
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"checkpoint_{formatted_datetime}.pth"))
                        args.CKPT_NAME = f"checkpoint_{formatted_datetime}.pth"
                        print("Done.\n")

                print(f"{'='*72}\n")

        print(f"Total time: {(time.time() - t_begin):.2f}s, Best Loss: {best_loss:.3f}")

        # Load model with the best weights
        model.load_state_dict(best_weights)

        history = dict(
                model      = model,
                train_loss = epoch_train_loss,
                val_loss   = epoch_val_loss,
                settings   = args
        )

        return history

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)

    args = parser.parse_args(sys.argv[1:])

    config = load_config(args.config)
    save_dir = None

    if config["Results"]["SAVE_RESULTS"]:
        mkdir_p(config["Results"]["SAVE_DIR"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["DatasetConfig"]["DATA_ROOT"].split("/")
        save_dir = os.path.join(
            config["Results"]["SAVE_DIR"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["SAVE_DIR"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        run = wandb.init(
            project="LatentDifussionModel",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("MSELoss", step_metric="frame_idx")

    args = munchify(config["TrainingConfig"])
    history = main(args=args)

    # All done
    print("Done.")