import torch
import pandas as pd
import numpy as np
import pickle
from dataset import create_train_val_dataset, create_test_dataset, collate_gnn_cpi
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgllife.utils import EarlyStopping
from tqdm import tqdm
from torch.distributions import Normal
import yaml
import os
import time
import matplotlib.pyplot as plt
from utils import Print, load_models, get_cindex
from dataset import Protein
from models import PLUS_RNN, MOL_GNN
from sklearn.metrics import auc, roc_auc_score, r2_score


def test(prot_model, mol_model, model_name, hp, output, **kwargs):
    for key, value in kwargs.items():
        hp[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)

    if os.path.exists(hp['data_dir']+"molgraph_dict.pkl"):
        with open(os.path.join(hp['data_dir'], "molgraph_dict.pkl"), 'rb') as f:
            molgraph_dict = pickle.load(f)
    else:
        molgraph_dict = {}

    sample_g = molgraph_dict[list(molgraph_dict.keys())[0]]
    x_size = sample_g.ndata['h'].shape[1]
    e_size = sample_g.edata['e'].shape[1]
    virtual_node = hp['num_virtual_nodes'] > 0
    
    encoder = Protein()
    prot_input_dim = len(encoder)
    
    models_list = []
    
    if prot_model == None:
        Print("Creating Base Protein LM Model...", output)
        prot_model = PLUS_RNN(hp, prot_input_dim)
        prot_model = prot_model.to(device)

    if mol_model == None:  
        Print("Creating Molecule GNN Model...", output)
        mol_model = MOL_GNN(hp, x_size)
        mol_model = mol_model.to(device)
    
    models_list.append([prot_model, "", True, True, False])
    models_list.append([mol_model, "mol", False, True, False])
        
    if model_name == None:
        load_models(hp, models_list, device, output)

    for model, idx, frz, _, _ in models_list: model.eval()
    
    Print('Creating Dataset...', output)
    test_ds = create_test_dataset(hp, encoder, device, molgraph_dict, output) 
    
    test_params = {'batch_size': hp['batch_size_val'],
                 'shuffle': False}
    collate_fn = collate_gnn_cpi(molgraph_dict)
    
    test_dl = DataLoader(test_ds, collate_fn=collate_fn, **test_params)

    if model_name == None:
        model_name = hp['gnn_type'] + "_" + str(int(round(time.time() * 1000)))
        os.mkdir("results/" + model_name)
    loss_func = F.mse_loss
    metrics = [get_cindex, r2_score]
            
    test_loss = 0
    test_ci = 0
    test_r2 = 0
    predlist = torch.Tensor()
    labellist = torch.Tensor()
    for batch_id, (g, p_block, lengths, y) in enumerate(tqdm(test_dl, total=int(len(test_dl)), leave=False)):
        g = g.to(torch.device(device))
        p_block = p_block.to(torch.device(device))
        lengths = lengths.to(torch.device(device))
        y = y.to(torch.device(device))[:,0]
        with torch.no_grad():
            z0, r0 = prot_model(p_block, lengths)
            prediction = mol_model(g, g.ndata['h'], z0)[:,0]
            
        loss = loss_func(prediction, y)
        y = y.cpu()
        prediction = prediction.cpu()

        ci = get_cindex(y, prediction)
        r2 = r2_score(y, prediction)
        #if batch_id % 10 == 0: 
            #Print('# test_loss {:.1%} loss={:.4f}'.format(
                #batch_id / len(test_dl), loss.item()), output)

        test_loss += loss.detach().item()
        test_ci += ci
        test_r2 += r2
        predlist = torch.cat([predlist, prediction])
        labellist = torch.cat([labellist, y])
        
    test_loss /= (batch_id + 1)
    test_ci /= (batch_id + 1)
    test_r2 /= (batch_id + 1)
    
    inline_log = 'Test_loss: {:.4f}, Test_ci: {:.4f}, Test_r2: {:.4f}'.format(test_loss,test_ci, test_r2)
    Print(inline_log, output)
    
    pltfile = "results/" + model_name + "_acc.png"
    plt.figure()
    plt.title('Accuracy Scatter Plot')
    plt.scatter(labellist, predlist, marker='o')
    plt.plot([torch.min(labellist), torch.max(labellist)], [torch.min(labellist), torch.max(labellist)], "r--")
    plt.xlabel("Actual Affinity")
    plt.ylabel("Predicted Affinity")
    plt.savefig(pltfile)
    plt.close()

    return test_loss, test_ci, test_r2



if __name__ == "__main__":
    import sys
    
    with open('default.yaml') as f:
        hp = yaml.load(f, Loader=yaml.FullLoader)
        
    if len(sys.argv) > 1:
        for i in range(len(sys.argv) - 1):
            key, value_raw = sys.argv[i+1].split("=")
            print(str(key) + ": " + value_raw)
            try:
                hp[key] = int(value_raw)
            except ValueError:
                try:
                    hp[key] = float(value_raw)
                except ValueError:
                    hp[key] = value_raw

    logfile = hp['logfile']
    output = open(logfile, "a")
        
    test_loss, test_ci, test_r2 = test(None, None, None, hp, output)
    Print('Testing done', output)
    