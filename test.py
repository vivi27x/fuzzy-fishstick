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
os.environ["WANDB_DISABLED"] = "true"

import model
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
        magnitude=False
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

    # init model
    net = BNext4DFR() #.load_from_checkpoint(join(cfg["test"]["weights_path"], f"{cfg['dataset']['name']}_{cfg['model']['backbone'][-1]}{'_unfrozen' if not cfg['model']['freeze_backbone'] else ''}.ckpt"), strict=False)

    device = "cpu"
    net = net.to(device)

    param_size = 0
    for param in net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    net = net.to('cpu')

    net.eval()#
    model_to_quantize = copy.deepcopy(net)
    qconfig = torch.ao.quantization.default_dynamic_qconfig # 'fbgemm' for server, 'qnnpack' for mobile
    # qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)
    qconfig_mapping = get_default_qconfig_mapping('fbgemm')
    # Performs 16bit quantization

    
    # torch.backends.quantized.engine = 'fbgemm' 

    model_to_quantize.eval()
    # # prepare
    model_to_quantize.base_model = quantize_fx.prepare_fx(model_to_quantize.base_model, qconfig_mapping, test_loader)
    # model_to_quantize.eval()

    # # calibrate (not shown)
    # # Replace the forward function of the traced model    
    calibrate_model(model_to_quantize, test_loader)
    # # # quantize
    
    model_to_quantize.base_model = quantize_fx.convert_fx(model_to_quantize.base_model)


    # # # #
    # # # # quantization aware training for static quantization


    # # # print('Post Training Quantization: Calibration done')

    # # # Save the quantized model
    # loaded_quantized_model = model_to_quantize
    torch.save(model_to_quantize.state_dict(), 'pc.pt')
    # import copy
    model2 = copy.deepcopy(net)
    model2.base_model = quantize_fx.prepare_fx(model2.base_model, qconfig_mapping, test_loader)
    model2.base_model = quantize_fx.convert_fx(model2.base_model)
    model2.load_state_dict(torch.load('pc.pt'), strict=False)

    param_size = 0
    for param in model2.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model2.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    # start training
    # date = datetime.now().strftime("%Y%m%d_%H%M")
    # project = "DFAD_CVPRW24"
    # run_label = args.cfg.split("/")[-1].split(".")[0]
    # run = cfg["dataset"]["name"] + f"_test_{date}_{run_label}"
    # logger = WandbLogger(project=project, name=run, id=run, log_model=False)
    # trainer = L.Trainer(
    #     precision="16-mixed",
    #     limit_test_batches=cfg["test"]["limit_test_batches"],
    #     logger=logger,
    # )
    # trainer.test(model=loaded_quantized_model, dataloaders=test_loader)
    tot = 0
    corr = 0
    t = 0
    with torch.no_grad():
        for i, argz in tqdm(enumerate(test_loader)):
            if (i > 10):
                break
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