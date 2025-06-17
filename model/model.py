import gc
import cv2 as cv
import numpy as np 
from skimage import feature
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchmetrics.functional.classification import accuracy, auroc
import timm
import lightning as L
from fvcore.nn import FlopCountAnalysis, parameter_count

from model.bnext import BNext

class BNext4DFR(L.LightningModule):
    def __init__(self, num_classes = 2, backbone='BNext-T', freeze_backbone=False, learning_rate=1e-4, pos_weight=1.):
        super(BNext4DFR, self).__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epoch_outs = []
        
        # loads the backbone
        self.backbone = backbone
        size_map = {"BNext-T": "tiny", "BNext-S": "small", "BNext-M": "middle", "BNext-L": "large"}
        if backbone in size_map:
            size = size_map[backbone]
            # loads the pretrained model
            self.base_model = BNext(num_classes=1000, size=size)
        else:
            print(backbone)
            raise ValueError("Unsupported Backbone!")
        
        # update the preprocessing metas
        # FFT and LBP channel
        self.new_channels = 2
        
        # loss parameters
        self.pos_weight = pos_weight
        
        self.adapter = nn.Conv2d(3+self.new_channels, 3, 3, (1,1), (1, 1), (1, 1), 1, True)
            
        # disables the last layer of the backbone
        self.inplanes = self.base_model.fc.in_features
        self.base_model.deactive_last_layer=True
        self.base_model.fc = nn.Identity()

        # eventually freeze the backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # add a new linear layer after the backbone
        self.fc = nn.Linear(self.inplanes, num_classes if num_classes >= 3 else 1)

        self.save_hyperparameters()
    
    def forward(self, x):
        outs = {}
        
        # extracts the features
        x_adapted = self.adapter(x)
        
        # normalizes the input image
        x_adapted = (x_adapted - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN, device=self.device).view(1, -1, 1, 1)) / torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_STD, device=self.device).view(1, -1, 1, 1)
        features = self.base_model(x_adapted)
        
        # outputs the logits
        outs["logits"] = self.fc(features)
        return outs
    
    def configure_optimizers(self):
        modules_to_train = [self.adapter, self.fc]
        if not self.freeze_backbone:
            modules_to_train.append(self.base_model)
        optimizer = optim.AdamW(
            [parameter for module in modules_to_train for parameter in module.parameters()], 
            lr=self.learning_rate,
            )
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.1, total_iters=5)
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        return self._on_start()
    
    def on_test_start(self):
        return self._on_start()
    
    def on_train_epoch_start(self):
        self._on_epoch_start()
        
    def on_test_epoch_start(self):
        self._on_epoch_start()
        
    def training_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="train")
    
    def validation_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="val")
    
    def test_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="test")
    
    def on_train_epoch_end(self):   
        self._on_epoch_end()
        
    def on_test_epoch_end(self):
        self._on_epoch_end()
    
    def _step(self, batch, i_batch, phase=None):
        images = batch["image"].to(self.device)
        outs = {
            "phase": phase,
            "labels": batch["is_real"][:, 0].float().to(self.device),
        }
        outs.update(self(images))
        if self.num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(input=outs["logits"][:, 0], target=outs["labels"], pos_weight=torch.as_tensor(self.pos_weight, device=self.device))
        else:
            raise NotImplementedError("Only binary classification is implemented!")
        # transfer each tensor to cpu previous to saving them
        for k in outs:
            if isinstance(outs[k], torch.Tensor):
                outs[k] = outs[k].detach().cpu()
        if phase in {"train", "val"}:
            self.log_dict({f"{phase}_{k}": v for k, v in [("loss", loss.detach().cpu()), ("learning_rate", self.optimizers().param_groups[0]["lr"])]}, prog_bar=False, logger=True)
        else:
            self.log_dict({f"{phase}_{k}": v for k, v in [("loss", loss.detach().cpu())]}, prog_bar=False, logger=True)
        # saves the outputs
        self.epoch_outs.append(outs)
        return loss
    
    def _on_start(self):
        with torch.no_grad():
            flops = FlopCountAnalysis(self, torch.randn(1, 5, 224, 224, device=self.device))
            parameters = parameter_count(self)[""]
            self.log_dict({
                "flops": flops.total(),
                "parameters": parameters
                }, prog_bar=True, logger=True)

    def _on_epoch_start(self):
        self._clear_memory()
        self.epoch_outs = []
    
    def _on_epoch_end(self):
        self._clear_memory()
        with torch.no_grad():
            labels = torch.cat([batch["labels"] for batch in self.epoch_outs], dim=0)
            logits = torch.cat([batch["logits"] for batch in self.epoch_outs], dim=0)[:, 0]
            phases = [phase for batch in self.epoch_outs for phase in [batch["phase"]] * len(batch["labels"])]
            assert len(labels) == len(logits), f"{len(labels)} != {len(logits)}"
            assert len(phases) == len(labels), f"{len(phases)} != {len(labels)}"
            for phase in ["train", "val", "test"]:
                indices_phase = [i for i in range(len(phases)) if phases[i] == phase]
                if len(indices_phase) == 0:
                    continue                
                metrics = {
                    "acc": accuracy(preds=logits[indices_phase], target=labels[indices_phase], task="binary", average="micro"),
                    "auc": auroc(preds=logits[indices_phase], target=labels[indices_phase].long(), task="binary", average="micro"),
                }
                self.log_dict({f"{phase}_{k}": v for k, v in metrics.items() if isinstance(v, (torch.Tensor, int, float))}, prog_bar=True, logger=True)
                    
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()