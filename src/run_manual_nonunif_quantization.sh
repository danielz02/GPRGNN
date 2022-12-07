#! /bin/sh
#
# This script is to reproduce our results in Table 2.
RPMAX=100

# Below is for homophily datasets, sparse split

python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset Cora \
        --lr 0.01 \
        --alpha 0.1 

python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset CiteSeer \
        --lr 0.01 \
        --alpha 0.1

python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset PubMed \
        --lr 0.05 \
        --alpha 0.2 
        
python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset Computers \
        --lr 0.05 \
        --alpha 0.5 \
        --weight_decay 0.0
        
python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset Photo \
        --lr 0.01 \
        --alpha 0.5 \
        --weight_decay 0.0

# Below is for heterophily datasets, dense split

python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset Chameleon \
        --lr 0.05 \
        --alpha 1.0 \
        --weight_decay 0.0 \
        --dprate 0.7 

python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset Film \
        --lr 0.01 \
        --alpha 0.9 \
        --weight_decay 0.0 

python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset Squirrel \
        --lr 0.05 \
        --alpha 0.0 \
        --weight_decay 0.0 \
        --dprate 0.7 
        
python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset Texas \
        --lr 0.05 \
        --alpha 1.0 \
        
python3 manual_Nonunif_quantization.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset Cornell \
        --lr 0.05 \
        --alpha 0.9 


