import torch
import pandas as pd
import numpy as np
import sys
import yaml
from train import train, test, test_cand
from utils import Print, print_hp


        
def run_experiment(title, **kwargs):
    with open('default.yaml') as f:
        hp = yaml.load(f, Loader=yaml.FullLoader)

    logfile = hp['logfile']
    output = open(logfile, "a")
        
    print_hp(hp, output)

    if len(kwargs) == 0:
        changing_var = 'num_epoch' # Using num_epochs as placeholder in case no override given
        changing_values = [hp[changing_var]]
        Print("Running experiments with default options", output)
    else:
        for key, value in kwargs.items():
            if len(value) > 1:
                changing_var = key
                changing_values = value
            else:
                hp[changing_var] = changing_values[0]
                Print("With "+ changing_var + " = " + str(changing_values[0]), output)
        
        Print("Running Experiments by adjusting " + changing_var + " among " + str(changing_values), output)
    
    assert changing_var in hp.keys(), "Parameter is not available. Please check default.yaml for list of available parameters."
    
    pd_title = []
    pd_model_name = []
    pd_var = []
    pd_value = []
    pd_train_similarity = []
    pd_val_similarity = []
    pd_test_similarity = []
    pd_train_loss = []
    pd_val_loss = []
    pd_test_loss = []
    pd_ranks = []
    
    for i in range(len(changing_values)):
        Print(str(i + 1) + ". Now " + changing_var + " = " + str(changing_values[i]), output)
        hp[changing_var] = changing_values[i]
        
        model, train_uws, val_uws, train_loss, val_loss, model_name = train(hp, output)
        test_loss, test_uws = test(model, model_name, hp)
        ranks = test_cand(model, model_name, hp)
        
        pd_title.append(title)
        pd_model_name.append(model_name)
        pd_var.append(changing_var)
        pd_value.append(changing_values[i])
        pd_train_similarity.append(train_uws)
        pd_val_similarity.append(val_uws)
        pd_test_similarity.append(test_uws)
        pd_train_loss.append(train_loss)
        pd_val_loss.append(val_loss)
        pd_test_loss.append(test_loss)
        pd_ranks.append(str(ranks))
    
    pd_results = pd.DataFrame({
        "Title": pd_title,
        "modelName": pd_model_name,
        "paramName": pd_var,
        "paramValue": pd_value,
        "train_similarity": pd_train_similarity,
        "val_similarity": pd_val_similarity,
        "test_similarity": pd_test_similarity,
        "train_loss": pd_train_loss,
        "val_loss": pd_val_loss,
        "test_loss": pd_test_loss,
        "ranks": pd_ranks
    })
    
    pd_results.to_csv("experiment_results.csv", mode='a', header=False, index=False)
    return pd_results
    
if __name__ == "__main__":
    import sys
    import yaml
    
    if len(sys.argv) == 1:
        run_experiment("Default")
    elif sys.argv[1] == "gnn_type":
        run_experiment("GNN Types", gnn_type=['wln', 'gat', 'gcn', 'gin'])
    elif sys.argv[1] == "num_gnn_layers":
        run_experiment("Number of GNN Layers", num_gnn_layers=[2,3,4,5,7])
    elif sys.argv[1] == "gnn_out_feat":
        run_experiment("Channel Size", gnn_out_feat=[320, 256, 192, 128,96])
    elif sys.argv[1] == "mode":
        run_experiment("Precursor Types (Mode)", mode=['negative', 'positive'])
    elif sys.argv[1] == "global_pooling":
        run_experiment("Pooling methods", global_pooling=['max', 'avg', 'attn'])
    elif sys.argv[1] == "data_dir":
        run_experiment("Dataset Sampling Size", data_dir=['data/sample7/', 'data/sample5/', 'data/sample3/'])
    elif sys.argv[1] == "virtual_node":
        run_experiment("Virtual Node", num_virtual_nodes=[1, 0], instrument_on_graph=[False])
    elif sys.argv[1] == "glu":
        run_experiment("GLU", glu=[False, True])
    elif sys.argv[1] == "lr":
        run_experiment("Learning Rate", lr=[1.0E-3, 2.0E-3, 5.0E-4])
    elif sys.argv[1] == "l2":
        run_experiment("L2 regularization", l2=[0.4, 0.2, 0.1, 0.05, 0.01])
    elif sys.argv[1] == "gat_num_layers":
        run_experiment("GAT num layers", gnn_type='gat', num_gnn_layers=[2,3,4,5,7])
    elif sys.argv[1] == "gnn_type2":
        run_experiment("GNN Types 2", gnn_type=['gat', 'wln'])
    elif sys.argv[1] == "attention":
        run_experiment("Attention Type", attn_type=[0,1,2,3])
