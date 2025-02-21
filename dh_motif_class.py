import random
from itertools import permutations
import numpy as np
import pickle
import time
import os
import math
import multiprocessing
import copy

class directed_hypergraph:

    def __init__(self, edge_list, N):
        self.edge_list = edge_list
        self.E = len(edge_list)
        self.N = N
        self.order = 2
        self.Ego = self.get_Ego()
        self.Bro = self.get_elder_neibors()
        self.W = [3**(self.order-1-i) for i in range(self.order)]


    def get_Ego(self):
        Ego = [set() for _ in range(self.N)]
        for i, e in enumerate(self.edge_list):
            for v in e[0] + e[1]:
                Ego[v].add(i)
        return Ego

    def get_elder_neibors(self):
        Bro=[]
        for i,e in enumerate(self.edge_list):
            this_brother_set = set()
            for v in e[0] + e[1]:
                for nb in self.Ego[v]:
                    if nb > i and nb not in this_brother_set:
                        this_brother_set.add(nb)
            Bro.append(this_brother_set)
        return Bro


    def encode(self, Esub):
        index_map, node_set = {}, set()
        for i,e in enumerate(Esub):
            for tail in e[0]:
                node_set.add(tail)
                if tail in index_map:
                    index_map[tail] += 1 * self.W[i]
                else:
                    index_map[tail] = 1 * self.W[i]
            for head in e[1]:
                node_set.add(head)
                if head in index_map:
                    index_map[head] += 2 * self.W[i]
                else:
                    index_map[head] = 2 * self.W[i]

        T = [0] * (3 ** self.order - 1)
        for v in index_map:
            T[index_map[v] - 1] += 1

        return tuple(T), node_set



    def binary_dh_motif_census(self, slice):
        CE, CV = {}, {}
        for e0 in slice:
            for e1 in self.Bro[e0]:
                e_0, e_1= self.edge_list[e0], self.edge_list[e1]
                T0, node_set = self.encode([e_0,e_1])
                T1, node_set = self.encode([e_1,e_0])
                T = max(T0, T1)

                if T not in CE:
                    CE[T] = {}
                if T not in CV:
                    CV[T] = {}

                for e in [e0,e1]:
                    if e in CE[T]:
                        CE[T][e] += 1
                    else:
                        CE[T][e] = 1
                for v in node_set:
                    if v in CV[T]:
                        CV[T][v] += 1
                    else:
                        CV[T][v] = 1
        return [CE, CV]


    def exter_binary_dh_motif_census(self, exter_slice):
        CE = {}
        for e0 in exter_slice:
            this_nb = set()
            e_0 = self.exter_edge_list[e0]
            for v in e_0[0]+e_0[1]:
                for nb in self.Ego[v]:
                    this_nb.add(nb)
            for e1 in this_nb:
                e_1 = self.edge_list[e1]
                T0, node_set = self.encode([e_0, e_1])
                T1, node_set = self.encode([e_1, e_0])
                T = max(T0, T1)

                if T not in CE:
                    CE[T] = {}
                if e0 in CE[T]:
                    CE[T][e0] += 1
                else:
                    CE[T][e0] = 1
        return CE


def multiprocess_run_census(dirH, workers):
    num_of_processe=workers
    slices = split_into_k_parts(list(range(dirH.E)), num_of_processe)
    pool = multiprocessing.Pool(num_of_processe)
    process_list = []
    for taskid in range(num_of_processe):
        process_list.append(pool.apply_async(dirH.binary_dh_motif_census, (slices[taskid],)))
    result_list = [r.get() for r in process_list]
    pool.close()
    pool.join()

    total_CE, total_CV = {}, {}
    for r in result_list:
        this_CE, this_CV= r
        for T in this_CE:
            if T in total_CE:
                update_d1_to_d2(this_CE[T], total_CE[T])
                update_d1_to_d2(this_CV[T], total_CV[T])
            else:
                total_CE[T] = this_CE[T]
                total_CV[T] = this_CV[T]
    return total_CE, total_CV



def exter_multiprocess_run_census(dirH, workers):
    num_of_processe = workers
    slices = split_into_k_parts(list(range(len(dirH.exter_edge_list))), num_of_processe)
    pool = multiprocessing.Pool(num_of_processe)
    process_list = []
    for taskid in range(num_of_processe):
        process_list.append(pool.apply_async(dirH.exter_binary_dh_motif_census, (slices[taskid],)))
    result_list = [r.get() for r in process_list]
    pool.close()
    pool.join()

    total_CE = {}
    for this_CE in result_list:
        for T in this_CE:
            if T in total_CE:
                update_d1_to_d2(this_CE[T], total_CE[T])
            else:
                total_CE[T] = this_CE[T]
    return total_CE



def split_into_k_parts(in_list, k):
    np.random.shuffle(in_list)
    result = np.array_split(in_list, k)
    return [ list(l) for l in result]

def update_d1_to_d2(d1,d2):
    for key in d1:
        if key not in d2:
            d2[key] = d1[key]
        else:
            d2[key]+= d1[key]
