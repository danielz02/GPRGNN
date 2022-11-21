import os
import sys
import time
import numpy as np

import torch
from dataset_utils import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test(model, data):
    model.eval()
    logits, accs, losses, preds = model(data), [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        loss = F.nll_loss(model(data)[mask], data.y[mask])

        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())


num_calibration_batches = 32

myModel = torch.load("../ckpt/cora_GPRGNN_8.pth").to('cpu')
myModel.eval()

dataset, data = DataLoader("cora")

# Fuse Conv, bn and relu
myModel.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myModel.qconfig = torch.ao.quantization.default_qconfig
print(myModel.qconfig)
torch.ao.quantization.prepare(myModel, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

# Calibrate with the training set
# evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
test(model, data)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.ao.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

