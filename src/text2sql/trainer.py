"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

from karpathy/minGPT
"""

import time
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = "cpu"
        if torch.cuda.is_available():
            print("Model is now CUDA!!!!")
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print(f"Saving Model at {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, verbose = False):
        model, config = self.model, self.config
        optimizer = model.configure_optimizers(config)
        lrscheduler = OneCycleLR(
            optimizer,
            max_lr = config.lr,
            total_steps=config.num_batch*config.max_epochs*config.tmult
        )

        with SummaryWriter(log_dir=config.tb_path, flush_secs=20) as tb:
            
            def run_epoch(split, epoch, _gs):
                is_train = split == "train"
                model.train(is_train)
                data = self.train_dataset if is_train else self.test_dataset
                dl = DataLoader(
                    data,
                    shuffle = True,
                    pin_memory = True,
                    batch_size = config.batch_size,
                    num_workers = config.num_workers
                )

                losses = []
                pbar = tqdm(enumerate(dl))
                for it, d in pbar:

                    with torch.set_grad_enabled(is_train):
                        total_steps = d["input"].size(1)
                        for t_step in range(total_steps):
                            _l = -1 if not losses else losses[-1]
                            if is_train:
                                pbar.set_description(f"[TRAIN] GS: {_gs}, Time: {t_step}/{total_steps},"
                                f" Epoch: {epoch}, Loss: {round(_l, 5)}")
                            else:
                                pbar.set_description(f"[VAL] Epoch: {epoch}")

                            loss, logits = model(
                                **{k:v.to(self.device) for k,v in d.items()},
                                get_loss=True,
                                device = self.device
                            )
                            losses.append(loss.item())

                            if is_train:
                                # add things to tb, loss and attention images
                                tb.add_scalar("loss", loss.item(), global_step=_gs, walltime=time.time())
                                tb.add_scalar("lr", lrscheduler.get_lr()[0], global_step=_gs, walltime=time.time())
                                # for l, att in enumerate(out.attentions):
                                #     tb.add_image(
                                #         f"attention/layer_{l}", att[0][0],
                                #         global_step=gs, walltime=time.time(),
                                #         dataformats= "HW"
                                #     )

                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                                optimizer.step()
                                lrscheduler.step()
                                _gs += 1

                if not is_train:
                    # no sampling here, because that really doesn't make any sense
                    test_loss = float(np.mean(losses))
                    tb.add_scalar("test_loss", test_loss, global_step=_gs, walltime=time.time())
                    return test_loss
                return _gs

            # now write wrapper for each epoch
            best_loss = float("inf")
            gs = 1
            test_no_improve = 0
            for e in range(config.max_epochs):
                gs = run_epoch("train", e, gs)
                if self.test_dataset is not None:
                    test_loss = run_epoch("test", e, gs)
                    print(f"Test loss: {test_loss}")

                # early stopping based on the test loss of just save always if no test set is provided
                good_model = self.test_dataset is None or test_loss < best_loss
                if self.config.ckpt_path is not None and good_model:
                    best_loss = test_loss
                    self.save_checkpoint()
                    test_no_improve = 0
                else:
                    test_no_improve += 1
                
#                 if test_no_improve == config.patience:
#                     print(f"Stop Training after [patience = {config.patience}]: {e} epochs")
#                     break


class TrainerConfig:
    lr = 0.0001
    max_epochs = 10
    batch_size = 128
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    num_workers = 0 # for DataLoader
    weight_decay = 0.1 # only applied on matmul weights

    len_data = None # required for CosineAnnealing
    sample_every = 5 # after how many epochs to log
    num_batch = None
    
    patience = 5 # training stops after patience runs out

    memlen = None # memory lenght of the model
    seqlen = None # length on which it is trained

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

        self.num_batch = (self.len_data // self.batch_size) + int(self.len_data % self.batch_size != 0)
        self.tmult = self.memlen // self.seqlen # each sample has its own number of time steps

    def __repr__(self):
        return "---- TRAINER CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "max_epochs",
                "batch_size",
                "betas",
                "grad_norm_clip",
                "num_workers",
                "sample_every",
                "num_batch",
                "len_data",
                "patience"
            ] + self.attrs))
        ]) + "\n"

# funcs
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
