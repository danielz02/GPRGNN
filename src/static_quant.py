import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from dataset_utils import DataLoader
import torch.nn.functional as F


def test(model, data, is_training=True, device="cuda:0"):
    model.eval()
    logits, accs, losses, preds = model(data.to(device)), [], [], []
    if is_training:
        partitions = ['train_mask']
    else:
        partitions = ['train_mask', 'val_mask', 'test_mask']
    for _, mask in data(*partitions):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        loss = F.nll_loss(model(data)[mask], data.y[mask])

        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())

    if not is_training:
        train_acc, val_acc, test_acc = accs
        print(f"Training Accuracy {train_acc:.4f} \t Validation Accuracy {train_acc:.4f} \t Test Accuracy {test_acc:.4f}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--q-config", type=str, choices=["default", "fbgemm"])
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # "../ckpt/cora_GPRGNN_8.pth"
    model = torch.load(args.ckpt)
    model.eval()

    dataset, data = DataLoader("cora")

    # Nothing to fuse for GPRGNN
    # model.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    model.qconfig = torch.ao.quantization.default_qconfig
    print(model.qconfig)
    torch.ao.quantization.prepare(model, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print("Model After Observer Insertion")
    print(model)

    # Calibrate with the training set
    test(model, data, is_training=True, device=args.device)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.ao.quantization.convert(model, inplace=True)
    if not hasattr(model, "dequant"):
        torch.ao.quantization.add_quant_dequant(model)
    print('Post Training Quantization: Convert done')
    print("Model After Quantization")
    print(model)

    test(model, data, is_training=False)


if __name__ == "__main__":
    main()
