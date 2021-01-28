#!/usr/bin/env python

"""This script trains a convtranstormer model on a parallel corpus.

All adjustable parameters are specified in the class 'Config'. Before
executing this script, the training set should be built first by
executing 'preprocess.py'.
"""


import logging
import pathlib
import statistics
from typing import List, Dict, Union, Optional
from timeit import default_timer as timer
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from alphabet import Alphabet
from dataset import ParallelDataset
from convtransformer import ConvTransformerModel
from scheduler import InverseSquareRootLR

logger = logging.getLogger(__name__)

Log = List[Dict[str, Union[int, float, bool]]]


def calculate_mean(numbers: List[float]) -> float:
    """Computes the arithmetic mean of a list of numbers.

    Instead of raising an exception, returns -1.0 when the list is
    empty.

    Args:
        numbers: List of numbers.

    Returns:
        The mean of the numbers in the passed list.
    """
    if len(numbers) > 0:
        return statistics.mean(numbers)
    else:
        return -1.0


def save_checkpoint(
    filename_prefix: str,
    model: ConvTransformerModel,
    optimizer: optim.Adam,
    scheduler: InverseSquareRootLR,
    conf_alph: Dict[str, str],
    conf_model: Dict[str, Union[int, float]],
    conf_optimizer: Dict[str, float],
    conf_scheduler: Dict[str, float],
    log: Log,
) -> None:
    """Saves a checkpoint of the model to a file.

    The trained weights are saved along with all other data necessary
    to resume training at a later time. The file name has the form
    '{prefix}-{date}-{time}.pt'.

    Args:
        filename_prefix: The prefix added to the filename.
        model: The model to be saved.
        optimizer: The optimizer to be saved.
        scheduler: The scheduler to be saved.
        conf_alph: Dictionary containing the initial configuration of
            the alphabet.
        conf_model: Dictionary containing the initial configuration of
            the model.
        conf_optimizer: Dictionary containing the initial configuration
            of the optimizer.
        conf_scheduler: Dictionary containing the initial configuration
            of the scheduler.
        log: List containing the statistics gathered during training.
    """
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = "{}-{}.pt".format(filename_prefix, current_time)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "conf_alph": conf_alph,
            "conf_model": conf_model,
            "conf_optimizer": conf_optimizer,
            "conf_scheduler": conf_scheduler,
            "log": log,
        },
        filename,
    )

    logger.info("Model Saved ({})".format(filename))


def save_model(
    filename_prefix: str,
    model: ConvTransformerModel,
    conf_alph: Dict[str, str],
    conf_model: Dict[str, Union[int, float]],
) -> None:
    """Saves the model to a file.

    The trained weights are saved together only with the data necessary
    for doing inference. The file name has the form
    '{prefix}-{date}-{time}.pt'.

    Args:
        filename_prefix: The prefix added to the filename.
        model: The model to be saved.
        conf_alph: Dictionary containing the initial configuration of
            the alphabet.
        conf_model: Dictionary containing the initial configuration of
            the model.
    """
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = "{}-{}.pt".format(filename_prefix, current_time)

    torch.save(
        {
            "conf_alph": conf_alph,
            "conf_model": conf_model,
            "model_state_dict": model.state_dict(),
        },
        filename,
    )

    logger.info("Model Saved ({})".format(filename))


def evaluate_val(
    model: ConvTransformerModel, data_val: ParallelDataset, device: torch.device
) -> float:
    """Evaluates the model on a given data set.

    Args:
        model: The model to be evaluated.
        data_val: A parallel data set.
        device: The torch device to use.

    Returns:
        The mean loss over the whole dataset.
    """
    loss_all: List[float] = []

    model.eval()
    with torch.no_grad():
        for src, tgt in data_val.generator:
            loss = model(src.to(device), tgt.to(device))[0]
            loss_all.append(loss.item())
    model.train()

    return calculate_mean(loss_all)


def print_log(
    log: Log,
    epoch: int,
    scheduler: InverseSquareRootLR,
    start_time: int,
    loss_mean_train: float,
    loss_mean_val: float,
    epoch_finished: Optional[bool] = False,
) -> None:
    """Logs the current training status.

    Apart from writing the info to the logger, the statistics will
    also be added to the list 'log'.

    Args:
        log: List containing the statistics gathered during training.
        epoch: The current epoch.
        scheduler: The scheduler uses for training.
        start_time: The starting time of the training.
        loss_mean_train: The current mean training loss.
        loss_mean_val: The current mean loss over the validation set.
        epoch_finished: Value specifying if the epoch just finished.
    """
    current_step = scheduler.last_epoch
    current_time = (int(timer()) - start_time) // 60
    current_lr = scheduler.get_last_lr()[0]

    # Write info to the logger.
    if epoch_finished:
        str_log = (
            "Epoch {} (Finished) | Step {} | Time: {}m | Total Train "
            "Loss: {:1.5f} | Val Loss: {:1.5f} | Learning rate: {:1.7f}"
        )
    else:
        str_log = (
            "Epoch {} | Step {} | Time: {}m | Train Loss: {:1.5f} | "
            "Val Loss: {:1.5f} | Learning rate: {:1.7f}"
        )
    str_log = str_log.format(
        epoch, current_step, current_time, loss_mean_train, loss_mean_val, current_lr
    )
    logger.info(str_log)

    # Add info to the list 'log'.
    log.append(
        {
            "epoch": epoch,
            "step": current_step,
            "time": current_time,
            "loss_train": loss_mean_train,
            "loss_val": loss_mean_val,
            "lr": current_lr,
        }
    )
    if epoch_finished:
        log[-1]["finished"] = True


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # All default parameter values are stored in the class 'Config'.
    conf = Config()

    # The class 'Alphabet' organizes the conversion from characters to
    # numbered indicies.
    alph = Alphabet(conf.chars_special, conf.dir_data + conf.file_alph)

    # The training data set
    data_train = ParallelDataset(
        conf.dir_data + conf.file_train_src,
        conf.dir_data + conf.file_train_tgt,
        alph,
        conf.max_len,
        conf.sz_batch,
        conf.sz_bucket,
    )

    # The validation data set
    data_val = ParallelDataset(
        conf.dir_data + conf.file_val_src,
        conf.dir_data + conf.file_val_tgt,
        alph,
        conf.max_len,
    )

    # The actual model
    model = ConvTransformerModel(
        alph.size,
        conf.sz_emb,
        conf.max_len,
        alph.idx_pad,
        conf.num_lay,
        conf.sz_kernels,
        conf.sz_kernel_final,
        conf.nhead,
        conf.dim_ff,
        conf.label_smoothing,
        conf.dropout,
    )

    # We use the Adam optimizer.
    optimizer = optim.Adam(
        model.parameters(), conf.warmup_init_lr, conf.adam_betas, conf.adam_eps
    )

    # We use a inverse square root decay scheme.
    scheduler = InverseSquareRootLR(
        optimizer, conf.base_lr, conf.warmup_init_lr, conf.warmup_steps
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # All the initial configuration is stored in separate dictionaries,
    # to be passed along when saving the model.
    conf_model = {
        "sz_alph": alph.size,
        "sz_emb": conf.sz_emb,
        "max_len": conf.max_len,
        "idx_pad": alph.idx_pad,
        "num_lay": conf.num_lay,
        "sz_kernels": conf.sz_kernels,
        "sz_kernel_final": conf.sz_kernel_final,
        "nhead": conf.nhead,
        "dim_ff": conf.dim_ff,
        "label_smoothing": conf.label_smoothing,
        "dropout": conf.dropout,
    }

    conf_optimizer = {
        "lr": conf.warmup_init_lr,
        "betas": conf.adam_betas,
        "eps": conf.adam_eps,
    }

    conf_scheduler = {
        "base_lr": conf.base_lr,
        "warmup_init_lr": conf.warmup_init_lr,
        "warmup_steps": conf.warmup_steps,
    }

    conf_alph = {
        "from_string": alph.get_alph_as_str(),
        "chars_special": conf.chars_special,
    }

    # The trained models will be saved in the folder 'dir_saves'.
    pathlib.Path(conf.dir_saves).mkdir(parents=True, exist_ok=True)
    file_save = conf.dir_saves + "model"

    # In the list 'log' we collect different statistics during
    # training.
    log: Log = []

    nr_epochs = conf.nr_epochs
    epoch_start = 1

    start_time = int(timer())
    time_since_last_log = start_time
    time_since_last_save = start_time

    # We will log statistics and save the model in regular intervals
    # as specified by 'log_interval' and 'save_interval'.
    log_interval = conf.log_interval
    save_interval = conf.save_interval

    logger.info("Alphabet Size: {} Characters".format(alph.size))
    logger.info(
        "Training Data: {} Samples ({} Steps per Epoch)".format(
            len(data_train.samples_src), len(data_train)
        )
    )
    logger.info("Validation Data: {} Samples".format(len(data_val.samples_src)))

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.info("--- Start Training ({}) ---".format(current_time))

    # Start the actual training
    for epoch in range(epoch_start, nr_epochs + 1):
        losses_current_epoch: List[float] = []
        losses_since_last_output: List[float] = []

        model.train()
        optimizer.zero_grad()

        for src, tgt in data_train.generator:
            # Forward and backward pass
            loss = model(src.to(device), tgt.to(device))[0]
            loss.backward()

            losses_current_epoch.append(loss.item())
            losses_since_last_output.append(loss.item())

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log current statistics if the time has come
            if int(timer()) - time_since_last_log > log_interval:
                loss_mean_train = calculate_mean(losses_since_last_output)
                loss_mean_val = evaluate_val(model, data_val, device)

                print_log(
                    log, epoch, scheduler, start_time, loss_mean_train, loss_mean_val
                )

                time_since_last_log = int(timer())
                losses_since_last_output: List[float] = []

            # Save a checkpoint if the time has come
            if int(timer()) - time_since_last_save > save_interval:
                save_checkpoint(
                    file_save,
                    model,
                    optimizer,
                    scheduler,
                    conf_alph,
                    conf_model,
                    conf_optimizer,
                    conf_scheduler,
                    log,
                )
                time_since_last_save = int(timer())

        # After the epoch is finished, log the statistics and save a
        # checkpoint
        loss_mean_train = calculate_mean(losses_current_epoch)
        loss_mean_val = evaluate_val(model, data_val, device)
        print_log(
            log, epoch, scheduler, start_time, loss_mean_train, loss_mean_val, True
        )

        save_checkpoint(
            file_save + "-epoch-" + str(epoch),
            model,
            optimizer,
            scheduler,
            conf_alph,
            conf_model,
            conf_optimizer,
            conf_scheduler,
            log,
        )

    # After the training is finished, save the final model
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.info("--- End Training ({}) ---".format(current_time))

    save_model(file_save + "-final", model, conf_alph, conf_model)
