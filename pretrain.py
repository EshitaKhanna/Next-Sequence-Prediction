#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description : Pre-training
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4NSP, LIMUBertModel4Pretrain, LayerNorm, Transformer, gelu
from utils import LIBERTDataset4NSP, Preprocess4NSP, set_seeds, get_device \
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask



# def main(args, training_rate):
#     data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

#     #pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
#     # pipeline = [Preprocess4Mask(mask_cfg)]
#     data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)

#     data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
#     data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
#     data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
#     data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
#     model = LIMUBertModel4Pretrain(model_cfg)

#     criterion = nn.MSELoss(reduction='none')

#     optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
#     device = get_device(args.gpu)
#     trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

#     def func_loss(model, batch):
#         mask_seqs, masked_pos, seqs = batch #

#         seq_recon = model(mask_seqs, masked_pos) #
#         loss_lm = criterion(seq_recon, seqs) # for masked LM
#         return loss_lm

#     def func_forward(model, batch):
#         mask_seqs, masked_pos, seqs = batch
#         seq_recon = model(mask_seqs, masked_pos)
#         return seq_recon, seqs

#     def func_evaluate(seqs, predict_seqs):
#         loss_lm = criterion(predict_seqs, seqs)
#         return loss_lm.mean().cpu().numpy()

#     if hasattr(args, 'pretrain_model'):
#         trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test
#                       , model_file=args.pretrain_model)
#     else:
#         trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)

def main(args, training_rate):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    print("Model Config: ", model_cfg)
    predict_len=5
    input_len=85
    # Create pipeline for NSP instead of masking
    pipeline = [
        Preprocess4Normalization(model_cfg.feature_num),
        Preprocess4NSP(predict_len=predict_len, input_len=input_len)
    ]
    
    # Prepare datasets
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(
        data, labels, training_rate, seed=train_cfg.seed
    )
    
    # Create datasets with NSP preprocessing
    data_set_train = LIBERTDataset4NSP(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4NSP(data_test, pipeline=pipeline)
    
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    
    # Create NSP model
    model = LIMUBertModel4NSP(model_cfg, predict_len=predict_len)
    
    # Loss function
    criterion = nn.MSELoss(reduction='none')
    
    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    
    # Setup trainer
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)
    
    # Loss function for NSP
    def func_loss(model, batch):
        input_seqs, target_pos, target_seqs = batch 
        seq_pred = model(input_seqs, target_pos)
        loss_nsp = criterion(seq_pred, target_seqs)
        return loss_nsp
    
    # Forward function for NSP
    def func_forward(model, batch):
        input_seqs, target_pos, target_seqs = batch # Extract batch data
        seq_pred = model(input_seqs, target_pos)  # Get predictions

        # # Convert tensors to numpy for printing
        # predicted_seq = seq_pred.detach().cpu().numpy()
        # actual_seq = target_seqs.detach().cpu().numpy()

        # print("\n===== NSP Pretrain Forward Pass Results =====")
        # print(f"Predicted (first sample):\n {predicted_seq[0]}")
        # print(f"Actual (first sample):\n {actual_seq[0]}")


        return seq_pred, input_seqs, target_seqs
    
    # Evaluation function
    def func_evaluate(seqs, predict_seqs):
        loss_nsp = criterion(predict_seqs, seqs)

        # # Convert tensors to NumPy for visualization
        # actual_seqs = seqs.cpu().detach().numpy()
        # predicted_seqs = predict_seqs.cpu().detach().numpy()

        # # Print a few samples to compare
        # for i in range(3):  # Print 3 samples
        #     print(f"Sample {i+1}:")
        #     print("Actual:   ", actual_seqs[i])
        #     print("Predicted:", predicted_seqs[i])
        #     print("-" * 50)
            
        return loss_nsp.mean().cpu().numpy()
    
    # Train the model
    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, 
                      model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, 
                      model_file=None)
    
    return model



if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)
