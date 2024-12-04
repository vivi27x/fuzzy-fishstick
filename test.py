from pprint import pprint
import argparse
import gc
from os.path import join
from datetime import datetime
import torch
import torch.ao.quantization
from torch.utils.data import DataLoader
from data.dataset import CIFAKEDataset
import time
from model.model import BNext4DFR
import os

from lib.util import load_config
import random
import numpy as np
import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy
from tqdm import tqdm

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/results_cifake_T_unfrozen.cfg",
    )
    args = parser.parse_args()
    return args

def calibrate_model(model, data_loader, n_batches = 5):
    """
    Calibrate the model using the provided data loader.
    
    Args:
        model: The quantization-aware model to calibrate.
        data_loader: DataLoader containing calibration data.
        n_batches: Number of batches to use for calibration.
    """
    model.eval()  # Set the model to evaluation mode

    # Calibrate the model
    print("Calibrating the model...")
    with torch.no_grad():
        for i, argz in tqdm(enumerate(data_loader)):
            if (i > n_batches):
                break
            op = model(argz['image'])
            continue

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    args = args_func()

    # load configs
    cfg = load_config(args.cfg)
    pprint(cfg)

    # preliminary setup
    torch.manual_seed(cfg["test"]["seed"])
    random.seed(cfg["test"]["seed"])
    np.random.seed(cfg["test"]["seed"])
    torch.set_float32_matmul_precision("medium")

    # get data
    print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
    test_dataset = CIFAKEDataset(
        dataset_path=cfg["dataset"]["cifake_path"],
        split="test",
        resolution=cfg["test"]["resolution"],
    )

    # loads the dataloaders
    num_workers = 4
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    tot = 0
    corr = 0
    t = 0
    with torch.no_grad():
        for i, argz in tqdm(enumerate(test_loader)):
            # Perform a forward pass of the model
            t1 = time.time()
            op = model2(argz['image'])["logits"]
            t2 = time.time()
            corr += ((op>0) == argz['is_real']).sum().item()
            tot += op.size(0)
            t += t2-t1
            # output = model(argz['image'])  
            print("Accuracy: ", corr/tot)
            print("Avg time/img", t/tot)
    print("Total time", t)
