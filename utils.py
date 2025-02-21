import multiprocessing
import os, sys
import time
import numpy as np
import torch
import itertools
import networkx as nx
from node2vec import Node2Vec
from scipy.sparse import csr_matrix
from collections import Counter
import pickle
import random
import scipy
from dh_motif_class import *
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from gensim.models import Word2Vec
import copy
import math
from torch_geometric.data import Data

class Logger:
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, train_eval, valid_eval, test_eval):
        result = [train_eval, valid_eval, test_eval]
        self.results[run].append(result)

    def get_statistics(self, run=None):
        if run is not None:
            auc = 100 * torch.tensor(self.results[run])[:,:,0]
            acc = 100 * torch.tensor(self.results[run])[:,:,1]
            f1 = 100 * torch.tensor(self.results[run])[:,:,2]

            max_train_auc = auc[:, 0].max().item()
            max_test_auc = auc[:, 2].max().item()

            argmax = auc[:, 1].argmax().item()   # max_auc for valid

            train_auc, valid_auc, test_auc = auc[argmax, 0].item(), auc[argmax, 1].item(), auc[argmax, 2].item()
            train_acc, valid_acc, test_acc = acc[argmax, 0].item(), acc[argmax, 1].item(), acc[argmax, 2].item()
            train_f1, valid_f1, test_f1 = f1[argmax, 0].item(), f1[argmax, 1].item(), f1[argmax, 2].item()

            return {'max_train_auc': max_train_auc, 'max_test_auc': max_test_auc,
                    'train_auc': train_auc, 'valid_auc': valid_auc, 'test_auc': test_auc,
                    'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc,
                    'train_f1': train_f1, 'valid_f1': valid_f1, 'test_f1': test_f1}
        else:
            keys = ['max_train_auc', 'max_test_auc',
                    'train_auc', 'valid_auc', 'test_auc',
                    'train_acc', 'valid_acc' , 'test_acc',
                    'train_f1', 'valid_f1', 'test_f1']

            best_results = []
            for r in range(len(self.results)):
                best_results.append([self.get_statistics(r)[k] for k in keys])

            ret_dict = {}
            best_result = torch.tensor(best_results)
            for i, k in enumerate(keys):
                ret_dict[k+'_mean'] = best_result[:, i].mean().item()
                ret_dict[k+'_std'] = best_result[:, i].std().item()

            return ret_dict


    def print_statistics(self):
        result = self.get_statistics()
        print(f"All runs:")
        print(f"Highest Train AUC: {result['max_train_auc_mean']:.2f} ± {result['max_train_auc_std']:.2f}")
        print(f"Highest Test AUC: {result['max_test_auc_mean']:.2f} ± {result['max_test_auc_std']:.2f}")
        print(f"Highest Valid AUC: {result['valid_auc_mean']:.2f} ± {result['valid_auc_std']:.2f}")
        print(f"Final Train AUC: {result['train_auc_mean']:.2f} ± {result['train_auc_std']:.2f}")
        print(f"Final Test AUC: {result['test_auc_mean']:.2f} ± {result['test_auc_std']:.2f},"
              f" ACC: {result['test_acc_mean']:.2f} ± {result['test_acc_std']:.2f},"
              f" F1: {result['test_f1_mean']:.2f} ± {result['test_f1_std']:.2f}")




def DHMNN_load_data(D, split_idx, Xe):

    n, m, E, F = D['n'], D['m'], D['E'], D['F']
    tXl, hXl = D['tXl'], D['hXl']
    Xg = D['Xg']
    train_Xe, valid_Xe, test_Xe = Xe
    train_idx = split_idx['train'].tolist()
    valid_idx = split_idx['valid'].tolist()
    test_idx = split_idx['test'].tolist()

    def load(split, Xe):

        L, l = {}, 0        # merge samples
        for i in split:
            L[l] = E[i]
            l += 1
        for i in split:
            L[l] = F[i]
            l += 1

        _, r, row, col = adjcoo(L, list(range(l)))       # clique expansion
        Xl = np.zeros((len(r), tXl.shape[1]),dtype=int)
        C_vertex, C_edge = [], []
        T_vertex, H_vertex =[], []
        T_edge, H_edge = [], []

        for x, (e, v, d) in enumerate(r):
            C_vertex.append(v)
            C_edge.append(e)
            if d == -1:          # tail
                Xl[x]= tXl[v]
                T_vertex.append(x)
                T_edge.append(e)
            elif d == 1:        # head
                Xl[x] = hXl[v]
                H_vertex.append(x)
                H_edge.append(e)

        data = Data()
        data.Xg = torch.FloatTensor(Xg)
        data.Xl = torch.FloatTensor(Xl)
        data.C_vertex, data.C_edge = torch.LongTensor(C_vertex), torch.LongTensor(C_edge)
        data.T_vertex, data.H_vertex = torch.LongTensor(T_vertex), torch.LongTensor(H_vertex)
        data.T_edge, data.H_edge = torch.LongTensor(T_edge), torch.LongTensor(H_edge)
        data.Xe = torch.FloatTensor(Xe)
        data.e_index = torch.LongTensor(np.array([row, col]))
        return data

    return load(train_idx, train_Xe), load(valid_idx, valid_Xe), load(test_idx, test_Xe)


def read_hyperlinks(args):
    ###### read raw data
    if args.data == 'Enron':
        with open('./data/'+ args.data + '/tail.pickle', 'rb') as file:
            tails = pickle.load(file)
        with open('./data/'+ args.data + '/head.pickle', 'rb') as file:
            heads = pickle.load(file)
        El, id, Es, EgoV = [], 0, set(), {}
        for t, h in zip(tails, heads):
            t, h = sorted(t), sorted(h)    # tail and head
            if  len(t) > 0 and len(h) > 0 and (tuple(t), tuple(h)) not in Es:  # valid and unduplicate
                for v in t + h:
                    if v not in EgoV:         # record the hyperarcs where each vertex is located
                        EgoV[v] = set([id])
                    else:
                        EgoV[v].add(id)
                id += 1
                El.append([tuple(t), tuple(h)])
                Es.add((tuple(t), tuple(h)))
        args.candidate = False
    else:
        mat_data = scipy.io.loadmat('./data/' + args.data + '.mat')
        CSC = mat_data[args.data]['S'][0, 0]
        incidence_matrix = np.array(CSC, dtype='int') # (V,E)
        El, id, Es, EgoV = [], 0, set(), {}
        for e in incidence_matrix.T:    # (E,V)
            t, h = [], []
            for index, x in enumerate(e):  # each reaction
                if x < 0:               # substrate
                    t.append(index)
                elif x > 0:            # product
                    h.append(index)
            if len(t) > 0 and len(h) > 0 and (tuple(t), tuple(h)) not in Es: # valid and unduplicate
                for v in t+h:
                    if v not in EgoV:          # record the hyperarcs where each vertex is located
                        EgoV[v] = set([id])
                    else:
                        EgoV[v].add(id)
                id += 1
                El.append([tuple(t), tuple(h)])
                Es.add((tuple(t), tuple(h)))

    ###### not consider isolated edges
    E, Vs, m = {}, set(), 0
    for i,e in enumerate(El):
        Ne = set().union(*(EgoV[t]  for t in e[0]), *(EgoV[h] for h in e[1]))  # neighbors of e
        Ne.remove(i)  # exclude itself
        if len(Ne) == 0:  # not consider isolated e
            continue
        else:
            E[m] = e
            m += 1
            for v in e[0] + e[1]:     # add to vertex set
                Vs.add(v)
    n = len(Vs)
    print('n:', n, 'e:', m)
    relabel_flag = 0
    if sorted(Vs) != list(range(n)):      # relabel vertex labels
        relabel_flag = 1

    ########### negative sampling
    re, F, Fs = args.re, {}, set()
    for k in E:
        t, h = E[k]
        ts, hs = set(t), set(h)
        while 1:
            re_t, re_h = int(len(t) * re), int(len(h) * re)
            rt, rh = random.sample(t, re_t), random.sample(h, re_h)  # retain
            st = random.sample(list(Vs - ts - hs), len(t) - re_t)
            sh = random.sample(list(Vs - ts - hs), len(h) - re_h)  # resample
            rt, rh, st, sh = tuple(sorted(rt)), tuple(sorted(rh)), tuple(sorted(st)), tuple(sorted(sh))
            f_t, f_h = set(rt + st), set(rh + sh)
            ft, fh = tuple(sorted(f_t)), tuple(sorted(f_h))
            if len(f_t & f_h) > 0 or (ft, fh) in Fs or (ft, fh) in Es:  # invalid or duplicate
                continue
            else:
                F[k] = [ft, fh]
                Fs.add((ft, fh))
                break

    ########### use candidate set
    if args.candidate:
        C_CSC = mat_data[args.data]['US'][0, 0]
        C_incidence_matrix = np.array(C_CSC, dtype='int')
        Cs = set()
        for c in C_incidence_matrix.T:
            ct, ch = [], []
            for index, x in enumerate(c): # each candidate reaction
                if x < 0:                # substrate
                    ct.append(index)
                elif x > 0:            # product
                    ch.append(index)
            ct, ch = tuple(ct), tuple(ch)
            if len(ct) > 0 and len(ch) > 0 and (ct, ch) not in Es and set(ct).issubset(Vs) and set(ch).issubset(Vs):   # valid and unduplicate
                Cs.add((ct,ch))
        replaced = set()
        for c in Cs:
            if c in Fs:
                continue
            ct, ch = c
            for f in F:
                ft, fh = F[f]
                if len(ct) == len(ft) and len(ch) == len(fh) and f not in replaced:  # have the same hyperarc size
                    replaced.add(f)
                    F[f] = [ct, ch]
                    break

    ########### check
    assert len(E) == len(F)
    for k in F:
        assert (F[k][0], F[k][1]) not in Es            ### Negative samples do not appear in positive samples
        assert len(F[k][0]) == len(E[k][0]) and len(F[k][1]) == len(E[k][1])    ### have the same hyperarc size distribution

    def check(E):
        uni = set()
        for i in E:
            t, h = E[i]
            if (t, h) not in uni:   ### no duplicate hyperarcs
                uni.add((t, h))
            else:
                return 0
            if len(t) == 0 or len(h) == 0:   ### no invalid hyperarcs
                return 0
            if len(set(t) & set(h)) > 0:    ### no invalid hyperarcs
                return 0
        return 1

    assert check(E) == 1 and check(F) == 1

    ##### relabel vertices
    if relabel_flag:
        relabel_d = { v : id for id, v in enumerate(sorted(Vs)) }
        for i in E:
            e = E[i]
            E[i] = [tuple([relabel_d[v] for v in e[0]]), tuple([relabel_d[v] for v in e[1]])]
        for i in F:
            e = F[i]
            F[i] = [tuple([relabel_d[v] for v in e[0]]), tuple([relabel_d[v] for v in e[1]])]

    D = {'n': n, 'm': m, 'E': E, 'F': F}
    return D

def rand_train_test_idx(D, runs, train_prop, valid_prop):
    n, m, E = D['n'], D['m'], D['E']
    test_prop = 1 - valid_prop - train_prop
    valid_num = int(m * valid_prop)
    test_num = int(m * test_prop)
    train_num = m - valid_num - test_num

    Ego = [set() for _ in range(n)]
    for e in E:
        t, h = E[e]
        for v in t + h:
            Ego[v].add(e)
    Ne = [set() for _ in range(m)]
    for s in Ego:
        for e1, e2 in itertools.combinations(s, 2):
            Ne[e1].add(e2)
            Ne[e2].add(e1)

    split_idx_list = []
    for r in range(runs):
        E_set, visited, unknown = set(list(range(m))), set(), set()     # unknown for vail and test
        this_Ego, this_Ne = copy.deepcopy(Ego), copy.deepcopy(Ne)

        while len(unknown) < valid_num + test_num:

            if len(list(E_set - visited - unknown)) == 0:
                print('warning: incomplete train set')
                unknown.update(set(random.sample(E_set - unknown, valid_num + test_num - len(unknown))))
                break

            e = random.sample(list(E_set - visited - unknown), 1)[0]  # random deletion

            this_t, this_h = E[e]
            flag1, flag2 = 1, 1
            for v in this_t + this_h:
                if len(this_Ego[v]) == 1:
                    flag1 = 0
                    break
            for nb in this_Ne[e]:
                if len(this_Ne[nb]) ==1:
                    flag2 = 0
                    break

            if flag1 & flag2:         # no isolated vertices and hyperarcs
                unknown.add(e)
                for v in this_t + this_h:
                    this_Ego[v].remove(e)
                for nb in this_Ne[e]:
                    this_Ne[nb].remove(e)
            else:
                visited.add(e)

        train_set = E_set - unknown
        valid_idx = random.sample(list(unknown), valid_num)   # random division
        valid_set = set(valid_idx)
        test_set = unknown - valid_set
        split_idx = {'train': torch.LongTensor(list(train_set)),
                     'valid': torch.LongTensor(valid_idx),
                     'test': torch.LongTensor(list(test_set))}
        split_idx_list.append(split_idx)
    print("train: ", train_num, ", valid: ", valid_num, ", test: ", test_num, ' x', runs, "runs")
    return split_idx_list

def external_embedding(exter_list, dirH, motif_list):
    dirH.exter_edge_list = exter_list
    CE = exter_multiprocess_run_census(dirH=dirH, workers=multiprocessing.cpu_count())
    Xe = np.zeros((len(exter_list), len(motif_list)), dtype=int)
    for index, T in enumerate(motif_list):
        if T in CE:
            for e in CE[T]:
                Xe[e][index] = CE[T][e]
    return Xe

def dhmotif_embedding(D, split_idx):
    train_idx = split_idx['train'].tolist()
    valid_idx = split_idx['valid'].tolist()
    test_idx = split_idx['test'].tolist()

    train_list = [D['E'][k] for k in train_idx]

    dirH = directed_hypergraph(N=D['n'], edge_list=train_list)

    Train_P_CE, Train_P_CV = multiprocess_run_census(dirH=dirH,workers=multiprocessing.cpu_count())

    motif_list = sorted([T for T in Train_P_CE])

    X = np.zeros((dirH.N, len(motif_list)), dtype=int)

    for index, T in enumerate(motif_list):
        for v in Train_P_CV[T]:
            X[v][index] = Train_P_CV[T][v]

    X = np.log1p(X)
    X = l2_norm(X)

    train_iXe = np.zeros((dirH.E, len(motif_list)), dtype=int)
    for index, T in enumerate(motif_list):
        for e in Train_P_CE[T]:
            train_iXe[e][index] = Train_P_CE[T][e]

    external_list = [D['F'][k] for k in train_idx]+[D['E'][k] for k in valid_idx]+[D['F'][k] for k in valid_idx]+[D['E'][k] for k in test_idx]+[D['F'][k] for k in test_idx]
    _Xe = external_embedding(external_list, dirH, motif_list)
    train_jXe = _Xe[:len(train_idx)]
    valid_iXe = _Xe[len(train_idx):len(train_idx) + len(valid_idx)]
    valid_jXe = _Xe[len(train_idx) + len(valid_idx):len(train_idx) + 2 * len(valid_idx)]
    test_iXe = _Xe[len(train_idx) + 2 * len(valid_idx):len(train_idx) + 2 * len(valid_idx) + len(test_idx)]
    test_jXe = _Xe[len(train_idx) + 2 * len(valid_idx) + len(test_idx):]

    train_Xe = np.vstack((train_iXe, train_jXe))
    valid_Xe = np.vstack((valid_iXe, valid_jXe))
    test_Xe = np.vstack((test_iXe, test_jXe))

    train_Xe = np.log1p(train_Xe)
    valid_Xe = np.log1p(valid_Xe)
    test_Xe = np.log1p(test_Xe)

    return X, [train_Xe, valid_Xe, test_Xe]


def adjcoo(H, I):
    n, d, r = 0, {}, []

    for j, i in enumerate(I):
        t, h = H[i]
        for v in t:
            p = (j, v, -1)
            d[p] = n
            r.append(p)
            n += 1
        for v in h:
            p = (j, v, 1)
            d[p] = n
            r.append(p)
            n += 1

    row, col = [], []
    for j, i in enumerate(I):
        t, h = H[i]
        for v in t:           # t<->t
            for u in t:
                if u == v:
                    continue
                k, l = d[(j, v, -1)], d[(j, u, -1)]  # -1 -1
                row.append(k)
                col.append(l)
                row.append(l)
                col.append(k)
            for u in h:          # t<->h
                assert v != u
                k, l = d[(j, v, -1)], d[(j, u, 1)]  # -1  1
                row.append(k)
                col.append(l)
                row.append(l)
                col.append(k)
        for v in h:                # h<->h
            for u in h:
                if u == v:
                    continue
                k, l = d[(j, v, 1)], d[(j, u, 1)]  # 1 1
                row.append(k)
                col.append(l)
                row.append(l)
                col.append(k)

    row, col = np.array(row), np.array(col)
    data = np.array([1] * len(row))
    return sp.coo_matrix((data, (row, col)), shape=(n, n)), r, row, col


def get_dir_incidence(D, index, add_loops=False):
    n, E = D['n'], D['E']
    tH = np.zeros( (n,len(index) ), dtype=float)
    hH = np.zeros( (n,len(index) ), dtype=float)
    for i, e in enumerate(index):
        t, h = E[e]
        for v in t:
            tH[v][i] = 1
        for v in h:
            hH[v][i] = 1
    if add_loops:
        tH = np.hstack((tH, np.eye(n)))
        hH = np.hstack((hH, np.eye(n)))
    return tH, hH


def l2_norm(X):
    norms = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


