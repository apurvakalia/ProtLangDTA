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
from utils import Print, load_models, get_cindex, save_model, print_hp
from test_model import test
from dataset import Protein
from models import PLUS_RNN, MOL_GNN, PROT_CNN, PROT_RNN
from sklearn.metrics import auc, roc_auc_score, r2_score

def train(hp, output, **kwargs):
    if len(kwargs) > 1:
        Print("Training with default with the following/above modifications:", output)
    for key, value in kwargs.items():
        hp[key] = value
        Print(str(key) + ": " + str(value), output)
    encoder = Protein()
    mode = 'both'
    prot_input_dim = len(encoder)
    
    if os.path.exists(hp['data_dir']+"molgraph_dict.pkl"):
        with open(os.path.join(hp['data_dir'], "molgraph_dict.pkl"), 'rb') as f:
            molgraph_dict = pickle.load(f)
    else:
        molgraph_dict = {}
    
    models_list = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    print('Creating Dataset...')
    train_ds, val_ds = create_train_val_dataset(mode, hp, encoder, device, molgraph_dict, output) 
    
    dl_params = {'batch_size': hp['batch_size_train'],
                 'shuffle': True}
    val_params = {'batch_size': hp['batch_size_val'],
                 'shuffle': False}
    
    collate_fn = collate_gnn_cpi(molgraph_dict)
    train_dl = DataLoader(train_ds, collate_fn=collate_fn, **dl_params)
    val_dl = DataLoader(val_ds, collate_fn=collate_fn, **val_params)
    
    sample_g = molgraph_dict[list(molgraph_dict.keys())[0]]
    x_size = sample_g.ndata['h'].shape[1]
    e_size = sample_g.edata['e'].shape[1]
    virtual_node = hp['num_virtual_nodes'] > 0
    
    
    # Create model
        
    if not os.path.exists("early_stopping/"):
        os.mkdir("early_stopping/")
    if not os.path.exists("results/"):
        os.mkdir("results/")
    
    if hp['activation'] is None:
        actv = None
    else:
        actv = F.relu
    
    Print("Creating Base Protein LM Model...", output)
        
    # Create model
    if hp['prot_model'] == 'plus':
        prot_model = PLUS_RNN(hp, prot_input_dim)
        models_list.append([prot_model, "", True, True, False])
    elif hp['prot_model'] == 'cnn':
        prot_model = PROT_CNN(hp, prot_input_dim)
        models_list.append([prot_model, "", False, True, False])
    elif hp['prot_model'] == 'rnn':
        prot_model = PROT_RNN(hp, prot_input_dim)
        models_list.append([prot_model, "", False, True, False])

            
    prot_model = prot_model.to(device)
    
    Print("Creating Molecule GNN Model...", output)
    
    mol_model = MOL_GNN(hp, x_size)
    models_list.append([mol_model, "mol", False, True, False])

    load_models(hp, models_list, device, output)

    model_name = hp['gnn_type'] + "_" + str(int(round(time.time() * 1000)))
    os.mkdir("results/" + model_name)
    
    loss_func = F.mse_loss
    metrics = [get_cindex, r2_score]
    params, pr_params = [], []
    for model, idx, frz, _, _ in models_list:
        if frz: continue
        elif idx != "mol": params    += [p for p in model.parameters() if p.requires_grad]
        else:             pr_params += [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam([{'params':params,    'lr':hp['prot_lr'],      'weight_decay': hp['l2']   },
                              {'params':pr_params, 'lr':hp['mol_lr'],   'weight_decay': hp['l2']   }])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25)
    stopper = EarlyStopping(
        mode='lower', patience = hp['early_stopping_patience'], 
        filename = "early_stopping/mol_"+ str(int(round(time.time() * 1000))))
    prot_stopper = EarlyStopping(
        mode='lower', patience = hp['early_stopping_patience'], 
        filename = "early_stopping/prot_"+ str(int(round(time.time() * 1000))))
    
    train_losses = []
    val_losses = []
    val_cis = []
    val_r2s = []
    
    Print("Starting Training...", output)
        
    for epoch in range(hp['num_epoch']):
        train_loss = 0
        for model, idx, frz, _, _ in models_list: model.train()

        for batch_id, (g, p_block, lengths, y) in enumerate(tqdm(train_dl, total=int(len(train_dl)), leave=False)):

            g = g.to(torch.device(device))
            p_block = p_block.to(torch.device(device))
            lengths = lengths.to(torch.device(device))
            y = y.to(torch.device(device))[:,0]
            z0, r0 = prot_model(p_block, lengths)
            
            prediction = mol_model(g, g.ndata['h'], z0)[:,0]
            loss = loss_func(prediction, y)
            loss.backward()
            for model, _, _, clip_grad, _ in models_list:
                if clip_grad: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            optimizer.zero_grad()
            #if batch_id % 10 == 0: 
                #Print('# epoch [{}/{}] train {:.1%} loss={:.4f}'.format(
                    #epoch + 1, hp['num_epoch'], batch_id / len(train_dl), loss.item()), output)

            train_loss += loss.detach().item()
        train_loss /= (batch_id + 1)
        
        val_loss = 0
        val_ci = 0
        val_r2 = 0
        for model, idx, frz, _, _ in models_list: model.eval()

        for batch_id, (g, p_block, lengths, y) in enumerate(val_dl):
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
            if batch_id % 10 == 0: 
                Print('# val_loss {:.1%} loss={:.4f}'.format(
                    batch_id / len(val_dl), loss.item()), output)

            val_loss += loss.detach().item()
            val_ci += ci
            val_r2 += r2
            
        val_loss /= (batch_id + 1)
        val_ci /= (batch_id + 1)
        val_r2 /= (batch_id + 1)
        inline_log = 'Epoch {} / {}, train_loss: {:.4f}, val_loss: {:.4f}, val_ci: {:.4f}, val_r2: {:.4f}'.format(
            epoch + 1, hp['num_epoch'], train_loss, val_loss, val_ci, val_r2
        )
        Print(inline_log, output)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_cis.append(val_ci)
        val_r2s.append(val_r2)
        scheduler.step()
        early_stop = prot_stopper(val_loss, prot_model)
        early_stop = stopper.step(val_loss, mol_model)
        if early_stop:
            break
    
        
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.ylim([0, 2])
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Loss')
    plt.savefig("results/" + model_name + "/training_loss.png")
            
    return prot_model, mol_model, val_ci, val_r2, train_loss, val_loss, model_name


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
    print_hp(hp, output)
    
    prot_model, mol_model, val_ci, val_r2, train_loss, val_loss, model_name = train(hp, output)
    test_loss, test_ci, test_r2 = test(prot_model, mol_model, model_name, hp, output)
    Print('Training done', output)
    