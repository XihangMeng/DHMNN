import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import numpy as np
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pickle
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import torch
import argparse
from utils import *
from DHMNN_model import DHMNN


def loss_func (score):
    Se, De = score['Se'], score['De']
    l = len(Se)
    iSe, jSe = Se[:l // 2], Se[l // 2:]
    iDe, jDe = De[:l // 2], De[l // 2:]
    Mean_jSe, Mean_jDe = torch.mean(jSe), torch.mean(jDe)
    loss = 0
    loss += torch.log1p(torch.exp(Mean_jSe - iSe))
    loss += torch.log1p(torch.exp(Mean_jDe - iDe))
    if 'Pe' in score:
        iPe, jPe = score['Pe'][:l//2], score['Pe'][l//2:]
        Mean_jPe = torch.mean(jPe)
        loss += torch.log1p(torch.exp(Mean_jPe - iPe))
    return loss.mean()



@torch.no_grad()
def eval_func(model, data):
    model.eval()
    score = model.forward(data)
    Se, De = score['Se'], score['De']
    l = len(Se)
    iSe, jSe = Se[:l // 2], Se[l // 2:]
    iDe, jDe = De[:l // 2], De[l // 2:]
    score['iSe'], score['jSe'] = iSe, jSe
    score['iDe'], score['jDe'] = iDe, jDe
    if 'Pe' in score:
        iPe, jPe = score['Pe'][:l // 2], score['Pe'][l // 2:]
        score['iPe'], score['jPe'] = iPe, jPe
    true = [1] * len(iSe) + [0] * len(jSe)
    if 'Pe' in score:
        ie_proba = ((iSe + iDe + iPe) / 3.0).cpu().numpy()
        je_proba = ((jSe + jDe + jPe) / 3.0).cpu().numpy()
    else:
        ie_proba = ((iSe + iDe) / 2.0).cpu().numpy()
        je_proba = ((jSe + jDe) / 2.0).cpu().numpy()
    proba = np.vstack((ie_proba, je_proba))
    pred = (proba >= 0.5).astype(int)
    return loss_func(score), calculate_metrics(true, proba, pred)


def calculate_metrics(true, proba, pred):
    AUR = roc_auc_score(true, proba)
    ACC = accuracy_score(true, pred)
    F1 = f1_score(true, pred)
    return [AUR, ACC, F1]


def evaluate(model, train_data, valid_data, test_data):
    train_loss, train_eval = eval_func(model, train_data)
    valid_loss, valid_eval = eval_func(model, valid_data)
    test_loss, test_eval = eval_func(model, test_data)
    return [train_eval, valid_eval, test_eval, train_loss, valid_loss, test_loss]




def main(args):

    device = args.device

    print("####", args.data, 'cuda:' + str(args.cuda))

    # read dataset
    D = read_hyperlinks(args)

    # split the dataset for each run
    split_idx_lst = rand_train_test_idx(D, args.runs, train_prop=args.train_prop, valid_prop=args.valid_prop)
    logger = Logger(args.runs)

    for run in range(args.runs):

        print('############ run:', run,"/ ",args.runs)
        split_idx = split_idx_lst[run]

        # connectivity information
        D['tXl'], D['hXl'] = get_dir_incidence(D, split_idx['train'].tolist())
        args.dl = D['tXl'].shape[1]
        # structural information
        D['Xg'], Xe = dhmotif_embedding(D, split_idx)
        args.dg = D['Xg'].shape[1]

        train_data, valid_data, test_data = DHMNN_load_data(D, split_idx, Xe)
        model = DHMNN(args)
        model, train_data, valid_data, test_data = model.to(device), train_data.to(device), valid_data.to(device), test_data.to(device)


        # optimizer
        opt = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad()
            score = model.forward(train_data)

            loss = loss_func(score)
            loss.backward()
            opt.step()
            result = evaluate(model, train_data, valid_data, test_data)

            if epoch % 100 == 0:
                print(f'{epoch:03d} '
                      f'|Train Loss: {result[3].item():.3f},  '
                      f'AUR: {100 * result[0][0]:.2f}% '
                      f'|Valid Loss: {result[4].item():.3f},  '
                      f'AUR: {100 * result[1][0]:.2f}%  '
                      f'|Test Loss: {result[5].item():.3f},  '
                      f'AUR: {100 * result[2][0]:.2f}%  ACC: {100 * result[2][1]:.2f}%  F1: {100 * result[2][2]:.2f}%')

            logger.add_result(run, *result[:3])
    logger.print_statistics()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Hypergraph Motif-based Framework for Directed Hyperlink Prediction")

    parser.add_argument('--seed', type=float, default=2025)
    parser.add_argument('--cuda', type=int, default=0)


    parser.add_argument('--data', type=str, default='iAF1260b')
    # from 'iAF1260b', 'iJO1366', 'iYS1720', 'Recon3D', 'iCHOv1',  'iMM1415', 'iLB1027_lipid', 'Enron'


    # for negative sampling
    parser.add_argument('--re', type=float, default=0.8, help="ratio of vertices retained for generating negative samples")
    parser.add_argument('--candidate', type=bool, default=True, help="use candidate set or not")


    # split ratio
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.1)


    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--h', type=int, default=256, help="hidden dimension")


    parser.add_argument("--q", type=float, default=0.5, help='information fusion coefficient')
    parser.add_argument("--heads", type=int, default=8, help='head for attention')
    parser.add_argument('--Classifier_num_layers', default=2, type=int, help='for topological score P_e')
    parser.add_argument('--Classifier_hidden', default=64, type=int, help='for topological score P_e')


    args = parser.parse_args()


    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    args.device = device

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)








