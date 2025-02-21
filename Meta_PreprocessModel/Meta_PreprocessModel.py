import scipy.io
import numpy as np
import json
from tqdm import tqdm

model_list = ['iAF1260b', 'iJO1366', 'iYS1720', 'Recon3D', 'iCHOv1',  'iMM1415', 'iLB1027_lipid']

for model_name in model_list:

    model_data = scipy.io.loadmat(f'../Meta_Raw_Data/{model_name}.mat')[model_name][0][0]
    model = {name: model_data[name] for name in model_data.dtype.names}

    R = model['rxns']   # reactions R
    M = model['mets']   # metabolites M

    with open('unrnames.txt', 'r') as f:
        unrnames = f.read().splitlines()
    with open('unrmet.txt', 'r') as f:
        unrmet = json.load(f)

    US = np.zeros((M.shape[0], len(unrmet)))  # universal reactions' S matrix (ONLY KEEP METABOLITES that appear IN ORIGINAL M)
    num_met_ur = []  # number of metabolites in each unr

    for i, cr in tqdm(enumerate(unrmet),total=len(unrmet)):
        cmet = cr['metabolites'] #current metabolites
        dS = np.zeros(M.shape[0])

        for j, met in enumerate(cmet):   #for each metabolite of the reaction, compare with M and keep the matches and form dS
            cname = f"{met['bigg_id']}_{met['compartment_bigg_id']}" #unify the format
            for index,name in enumerate(M):
                if cname == name[0][0]:
                    dS [index] = met['stoichiometry']
                    break

        num_met_ur.append(len(cmet))  # record the number of metabolites for each reaction in unr
        US[:, i] = dS  # append dS to US

    LUS = np.array(US, dtype=bool)
    ind_US = np.where(LUS.sum(axis=0) == np.array(num_met_ur))[0]  # only keep those reactions (columns) in US whose metabolites are all in M
    US = US[:, ind_US] # the reduced US

    print(model_name, "valid negative simples", US.shape)

    model['US'] = US
    scipy.io.savemat(f'../data/{model_name}.mat', {model_name: model})