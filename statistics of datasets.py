
' BASIC STATISTICS OF EIGHT REAL-WORLD DIRECTED HYPERGRAPHS '

import pickle
import scipy
import numpy as np


data_list = ['iAF1260b', 'iJO1366', 'iYS1720', 'Recon3D', 'iCHOv1',  'iMM1415', 'iLB1027_lipid', 'Enron']

for data in data_list:
    if data == 'Enron':
        with open('./data/' + data + '/tail.pickle', 'rb') as file:
            tails = pickle.load(file)
        with open('./data/' + data + '/head.pickle', 'rb') as file:
            heads = pickle.load(file)

        El, id, Es, EgoV = [], 0, set(), {}
        for t, h in zip(tails, heads):
            t, h = sorted(t), sorted(h)
            if len(t) > 0 and len(h) > 0  and (tuple(t), tuple(h)) not in Es:       # valid hyperarcs
                for v in t + h:
                    if v not in EgoV:
                        EgoV[v] = set([id])
                    else:
                        EgoV[v].add(id)
                id += 1
                El.append([tuple(t), tuple(h)])
                Es.add((tuple(t), tuple(h)))

    else:
        mat_data = scipy.io.loadmat('./data/' + data + '.mat')
        CSC = mat_data[data]['S'][0, 0]
        incidence_matrix = np.array(CSC, dtype='int')  # (V,E)
        El, id, Es, EgoV = [], 0, set(), {}
        for e in incidence_matrix.T:  # (E,V)
            t, h = [], []
            for index, x in enumerate(e):
                if x < 0:         # substrates as tail vertices
                    t.append(index)
                elif x > 0:        # products as hrad vertices
                    h.append(index)
            if len(t) > 0 and len(h) > 0 and (tuple(t), tuple(h)) not in Es:    # valid hyperarcs
                for v in t + h:
                    if v not in EgoV:
                        EgoV[v] = set([id])
                    else:
                        EgoV[v].add(id)
                id += 1
                El.append([tuple(t), tuple(h)])
                Es.add((tuple(t), tuple(h)))

    ###### not consider isolated hyperarcs
    E, Vs, m, Ne_list = {}, set(), 0, []
    for i, e in enumerate(El):
        Ne = set().union(*(EgoV[t] for t in e[0]), *(EgoV[h] for h in e[1]))
        Ne.remove(i)
        if len(Ne) == 0:  # isolated hyperarc
            continue
        else:
            E[m] = e
            m += 1
            Ne_list.append(len(Ne))
            for v in e[0] + e[1]:
                Vs.add(v)
    n = len(Vs)
    assert len(Ne_list) == m

    e_min, t_min, h_min = float('inf'), float('inf'), float('inf')
    e_max, t_max, h_max = float('-inf'), float('-inf'), float('-inf')
    e_mean = 0
    for  i in E:
        t , h = E[i]
        e_min = min(e_min, len(t + h))
        e_max = max(e_max, len(t + h))
        e_mean += len(t + h)

        t_min = min(t_min, len(t))
        t_max = max(t_max, len(t))

        h_min = min(h_min, len(h))
        h_max = max(h_max, len(h))

    print(data, '&', n, '&', m, '&', e_mean/m, '&', e_min, '&', e_max, '&', t_min, '&',t_max, '&',h_min, '&', h_max, '&', sum(Ne_list)/m)
