#Modification of 'train_model.py'
#When run the code, it will generate histogram file
#No argument needed (Update default argument same as example condition, --RPMAX 2 --net GPRGNN --train_rate 0.025 --val_rate 0.025 --dataset cora)
#Performing uniform quantization on weights and bias
#Check the accuracy of the model after quantization
#and printout the results

import argparse
from dataset_utils import DataLoader
from utils import random_planetoid_splits
from GNN_models_original import *

import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
import copy

def Unif_Quantization(weights,Nbit):
    Nbin=2**Nbit
    Shape=np.shape(weights.numpy())
    Weight_Np=weights.numpy().reshape(-1)
    Min=np.min(Weight_Np)
    Max=np.max(Weight_Np)
    StepSize=(Max-Min)/Nbin
    Quantized_weight=np.zeros(Weight_Np.size) 
    Quantized_weight=np.floor((Weight_Np-Min)/StepSize)
    #following codeis to deal with the maximum element
    Quantized_weight[np.where(Quantized_weight==Nbin)]=Nbin-1
    
    Quantized_weight=(Quantized_weight+0.5)*StepSize+Min
    Quantized_weight=Quantized_weight.reshape(Shape)
    MSE = np.square(np.subtract(weights,Quantized_weight)).mean().tolist()
    #print("MSE {}".format(MSE))
    return torch.nn.Parameter(torch.from_numpy(Quantized_weight)), MSE

def RunExp(args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

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
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    
    
    #Applying uniform quantization here and compare its performance

    [train_acc, val_acc, tmp_test_acc], preds, [
        train_loss, val_loss, tmp_test_loss] = test(model, data)
    print('Before quantization')
    print(f"Training Accuracy {train_acc:.4f} \t Validation Accuracy {val_acc:.4f} \t Test Accuracy {tmp_test_acc:.4f}")   
    plt.plot(model.prop1.temp.data)
    Led=['Original Gamma']
    data_save = np.load("quan.npz", allow_pickle=True)
    results = data_save['data'].reshape(-1)[0]
    result_bits = []
    for Nbit in range(9):
        result = []
        #Copy model to retain the trained model
        model_quantized=copy.deepcopy(model)

        model_quantized.lin1.weight.requires_grad=False
        model_quantized.lin1.bias.requires_grad=False
        model_quantized.lin2.weight.requires_grad=False
        model_quantized.lin2.bias.requires_grad=False
        
        #Quantize weights here
        model_quantized.lin1.weight.data, MSE_w1=Unif_Quantization(model_quantized.lin1.weight.data,Nbit)
        model_quantized.lin1.bias.data, MSE_b1=Unif_Quantization(model_quantized.lin1.bias.data,Nbit)
        model_quantized.lin2.weight, MSE_w2=Unif_Quantization(model_quantized.lin2.weight,Nbit)
        model_quantized.lin2.bias.data, MSE_b2=Unif_Quantization(model_quantized.lin2.bias.data,Nbit)
    

        #Check performance after quantization
        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model_quantized, data)
        print('After quantization with {} bit'.format(Nbit))
        print(f"Training Accuracy {train_acc:.4f} \t Validation Accuracy {val_acc:.4f} \t Test Accuracy {tmp_test_acc:.4f}")
        result.append(tmp_test_acc)
        
        #optimizer for the retraining. As it is only updating prop1 function, just put that
        optimizer_quant = torch.optim.Adam([{
            'params': model_quantized.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
        
        #retrain model
        for epoch_after_quantize in range(100):
            train(model_quantized, optimizer_quant, data, args.dprate)
        
        #Check performance again
        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model_quantized, data)
        print('After quantization with {} bit and retrain gamma'.format(Nbit))
        print(f"Training Accuracy {train_acc:.4f} \t Validation Accuracy {val_acc:.4f} \t Test Accuracy {tmp_test_acc:.4f}")
        result.append(tmp_test_acc)
        result += [MSE_w1, MSE_b1, MSE_w2, MSE_b2]
        result_bits.append(result)
        plt.plot(model_quantized.prop1.temp.data,'o')
        Led.append('Gamma after {} bit quantization'.format(Nbit))
    plt.legend(Led,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    results[args.dataset] = np.array(result_bits).T.tolist()
    np.savez("quan", data=results)
    return test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=2)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN'],
                        default='GPRGNN')
    args = parser.parse_args()

    
    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN

    dname = args.dataset
    dataset, data = DataLoader(dname)

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    # print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    #run only once
    
    Results0 = []
    test_acc, best_val_acc, Gamma_0 = RunExp(
        args, dataset, data, Net, percls_trn, val_lb)
    Results0.append([test_acc, best_val_acc, Gamma_0])
    