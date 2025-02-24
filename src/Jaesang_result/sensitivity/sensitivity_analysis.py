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
from sklearn.metrics import mean_squared_error

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
    return torch.nn.Parameter(torch.from_numpy(Quantized_weight))

def NonUnif_Quantization(weights,Nbit):
    #Non Uniform quantization using Lloyd's algorithm
    Nbin=2**Nbit
    Shape=np.shape(weights.numpy())
    Weight_Np=weights.numpy().reshape(-1)
    Min=np.min(Weight_Np)
    Max=np.max(Weight_Np)
    Quantized_weight=np.zeros(Weight_Np.size) 
    
    Thresholds=np.linspace(Min,Max,num=Nbin+1)
    
    # Centroids=np.zeros(Nbin)
    if Weight_Np.size>Nbin:
        random_idx=np.random.choice(Weight_Np.size,Nbin,replace=False)
        Centroids=np.sort(Weight_Np[random_idx])
        for i in range(Nbin-1): #no need to update t0 and t_M (they are min and max)
            Thresholds[i+1]=(Centroids[i]+Centroids[i+1])/2    
            
        for Itr in range(200):
            #update centroids
            for i in range(Nbin):
                if i==0:
                    idx_range_i=np.where(Weight_Np<=Thresholds[i+1])
                else:
                    idx_range_i=np.where((Thresholds[i]<Weight_Np) & (Weight_Np<=Thresholds[i+1]))
                Centroids[i]=np.mean(Weight_Np[idx_range_i])
                Quantized_weight[idx_range_i]=Centroids[i]
            #update threshold
            for i in range(Nbin-1): #no need to update t0 and t_M (they are min and max)
                Thresholds[i+1]=(Centroids[i]+Centroids[i+1])/2
    else: #we have more bin than data size. We don't need quantization here
        Quantized_weight=Weight_Np
    
    # plt.figure()
    # n, bins, patches = plt.hist(x=Weight_Np, bins='auto', color='#0504aa', rwidth=0.85)
    # plt.grid(axis='y')
    # plt.xlabel('Value')
    # plt.show()
    
    # plt.figure()
    # n, bins, patches = plt.hist(x=Quantized_weight, bins='auto', color='#0504aa', rwidth=0.85)
    # plt.grid(axis='y')
    # plt.xlabel('Value')
    # plt.show()
    
    Quantized_weight=Quantized_weight.reshape(Shape)
    return torch.nn.Parameter(torch.Tensor.double(torch.from_numpy(Quantized_weight)).to(dtype=torch.float32))

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
    def train_valdata(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.val_mask]
        nll = F.nll_loss(out, data.y[data.val_mask])
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
    Accuracy[RP,0]=tmp_test_acc

    Nbit=1
    for i in range(4):
        
        #Copy model to retain the trained model
        model_quantized=copy.deepcopy(model)

        model_quantized.lin1.weight.requires_grad=False
        model_quantized.lin1.bias.requires_grad=False
        model_quantized.lin2.weight.requires_grad=False
        model_quantized.lin2.bias.requires_grad=False

        #Quantize weights here
        if i==0:
            model_quantized.lin1.weight.data=Unif_Quantization(model_quantized.lin1.weight.data,Nbit)
        elif i==1:
            model_quantized.lin1.bias.data=Unif_Quantization(model_quantized.lin1.bias.data,Nbit)
        elif i==2:
            model_quantized.lin2.weight.data=Unif_Quantization(model_quantized.lin2.weight.data,Nbit)
        elif i==3:
            model_quantized.lin2.bias.data=Unif_Quantization(model_quantized.lin2.bias.data,Nbit)
   
        #Check performance after quantization
        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model_quantized, data)
        print('After quantization with {} bit'.format(Nbit))
        print(f"Training Accuracy {train_acc:.4f} \t Validation Accuracy {val_acc:.4f} \t Test Accuracy {tmp_test_acc:.4f}")
        Accuracy[RP,i+1]=tmp_test_acc
        
        #optimizer for the retraining. As it is only updating prop1 function, just put that
        optimizer_quant = torch.optim.Adam([{
            'params': model_quantized.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
        

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
    parser.add_argument('--RPMAX', type=int, default=20)
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
    Accuracy=np.zeros([RPMAX,5])
    Results0 = []
    for RP in tqdm(range(RPMAX)):
        test_acc, best_val_acc, Gamma_0 = RunExp(
            args, dataset, data, Net, percls_trn, val_lb)
        Results0.append([test_acc, best_val_acc, Gamma_0])
    # plt.plot(np.mean(Accuracy,axis=0),'o')
    # plt.show()
    # print(Accuracy)
    with open('Sensitivity_with_{}_run.txt'.format(RPMAX), 'w') as f:
        # for line in net1.i2h.weight.data.numpy():
        f.write(str(np.mean(Accuracy,axis=0)).replace('\n',' ').replace('[','').replace(']','\n'))
        f.write(str(np.std(Accuracy,axis=0)).replace('\n',' ').replace('[','').replace(']','\n'))

    labels=['Original','Quantize lin1 weights','Quantize lin1 bias','Quantize lin2 weights','Quantize lin2 bias']
    plt.errorbar(range(5),np.mean(Accuracy,axis=0),yerr=1.96*np.std(Accuracy,axis=0)/np.sqrt(RPMAX),fmt='o')
    plt.xticks(range(5),labels,rotation='vertical')
    plt.title('Accuracy with 95% confidence interval')
    plt.savefig('Sensitivity_analysis', bbox_inches="tight")
    plt.show()
