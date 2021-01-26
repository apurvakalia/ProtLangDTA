# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# - Seonwoo Min's code (https://github.com/mswzeus/PLUS)

import os
import sys
import argparse

import torch
import json
from collections import defaultdict

import src.config as config
from src.data.alphabets import Protein
import src.data.cpi as cpi
import src.data.dataset as dataset
import src.model.plus_rnn as plus_rnn
import src.model.plus_tfm as plus_tfm
import src.model.p_elmo as p_elmo
import src.model.gnn as gnn
from src.train import Trainer
from src.utils import Print, set_seeds, set_output, load_models
import dgl

parser = argparse.ArgumentParser('Train a Model on CPI Datasets')
parser.add_argument('--data-config',  help='path for data configuration file')
parser.add_argument('--model-config', help='path for model configuration file')
parser.add_argument('--lm-model-config', help='path for lm-model configuration file (for P-ELMo)')
parser.add_argument('--pr-model-config', help='path for pr-model configuration file (for P-ELMo and PLUS-RNN)')
parser.add_argument('--run-config', help='path for run configuration file')
parser.add_argument('--pretrained-model', help='path for pretrained model file')
parser.add_argument('--pretrained-lm-model', help='path for pretrained lm-model file (for P-ELMo)')
parser.add_argument('--pretrained-pr-model', help='path for pretrained pr-model file (for P-ELMo and PLUS-RNN)')
parser.add_argument('--device', help='device to use; multi-GPU if given multiple GPUs sperated by comma (default: cpu)')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')
parser.add_argument('--output-index', help='prefix for outputs')
parser.add_argument('--sanity-check', default=False, action='store_true', help='sanity check flag')


def main():
    set_seeds(2020)
    args = vars(parser.parse_args())

    alphabet = Protein()
    cfgs = []
    #initialize dictionaries

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    molgraph_dict = {}
    dicts = cpi.Dict(atom_dict, bond_dict, fingerprint_dict, molgraph_dict)
    
    data_cfg  = config.DataConfig(args["data_config"]);   cfgs.append(data_cfg)
    if args["lm_model_config"] is None:
        model_cfg = config.ModelConfig(args["model_config"], input_dim=len(alphabet), num_classes=2)
        cfgs += [model_cfg]
    else:
        lm_model_cfg = config.ModelConfig(args["lm_model_config"], idx="lm_model_config", input_dim=len(alphabet))
        model_cfg = config.ModelConfig(args["model_config"], input_dim=len(alphabet),
                      lm_dim=lm_model_cfg.num_layers * lm_model_cfg.hidden_dim * 2, num_classes=2)
        cfgs += [model_cfg, lm_model_cfg]
    if model_cfg.model_type == "RNN":
        pr_model_cfg = config.ModelConfig(args["pr_model_config"], idx="pr_model_config", model_type="MLP", num_classes=2)
        if pr_model_cfg.projection: pr_model_cfg.set_input_dim(model_cfg.embedding_dim)
        else:                       pr_model_cfg.set_input_dim(model_cfg.hidden_dim * 2)
        pr_model_cfg.set_embedding_dim(model_cfg.embedding_dim)
        cfgs.append(pr_model_cfg)
    run_cfg = config.RunConfig(args["run_config"], sanity_check=args["sanity_check"]);  cfgs.append(run_cfg)
    output, save_prefix = set_output(args, "train_dta_log")
    #os.environ['CUDA_VISIBLE_DEVICES'] = args["device"] if args["device"] is not None else ""
    device, data_parallel = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.cuda.device_count() > 1
    #device = torch.device("cpu")
    config.print_configs(args, cfgs, device, output)
    flag_rnn = (model_cfg.model_type == "RNN")
    flag_lm_model = (args["lm_model_config"] is not None)
    flag_lm_loss = (run_cfg.lm_loss_lambda != -1)
    run_cfg.pure_gnn = data_cfg.pure_gnn
    
    ## load a train dataset
    start = Print(" ".join(['start loading a train dataset:', data_cfg.path["train"]]), output)
    dataset_train = cpi.load_cpi(data_cfg, "train", alphabet, dicts, device, args["sanity_check"])

    dataset_train = dataset.CPI_dataset(*dataset_train, alphabet, run_cfg, flag_rnn, model_cfg.max_len)
    collate_fn = dataset.collate_cpi_sequences if flag_rnn else None
    iterator_train = torch.utils.data.DataLoader(dataset_train, run_cfg.batch_size_train, collate_fn=collate_fn, shuffle=True)
    end = Print(" ".join(['loaded', str(len(dataset_train)), 'sequences']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## load a dev dataset
    start = Print(" ".join(['start loading a dev dataset:', data_cfg.path["dev"]]), output)
    dataset_dev = cpi.load_cpi(data_cfg, "dev", alphabet, dicts, device, args["sanity_check"])
    dataset_dev = dataset.CPI_dataset(*dataset_dev, alphabet, run_cfg, flag_rnn, model_cfg.max_len)
    iterator_dev = torch.utils.data.DataLoader(dataset_dev, run_cfg.batch_size_eval, collate_fn=collate_fn)
    end = Print(" ".join(['loaded', str(len(dataset_dev)), 'sequences']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)
    
    ## load a test dataset just to update dictionaries
    start = Print(" ".join(['start loading a test dataset:', data_cfg.path["test"]]), output)
    dataset_test = cpi.load_cpi(data_cfg, "test", alphabet, dicts, device, args["sanity_check"])
    dataset_test = dataset.CPI_dataset(*dataset_test, alphabet, run_cfg, flag_rnn, model_cfg.max_len)
    #iterator_test = torch.utils.data.DataLoader(dataset_test, run_cfg.batch_size_eval, collate_fn=collate_fn)
    end = Print(" ".join(['loaded', str(len(dataset_test)), 'sequences']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    #save dictionaries

    torch.save(dict(atom_dict), save_prefix + "atom_dict.pt")
    torch.save(dict(bond_dict), save_prefix + "bond_dict.pt")
    torch.save(dict(fingerprint_dict), save_prefix + "fingerprint_dict.pt")
    torch.save(dict(molgraph_dict), save_prefix + "molgraph_dict.pt")
    model_cfg.cpi = data_cfg.cpi
    run_cfg.cpi = data_cfg.cpi
    
    if flag_rnn:
        if run_cfg.pure_gnn:
            collate_fn = dataset.collate_gnn_cpi(molgraph_dict)
        else:
            collate_fn = dataset.collate_cpi_sequences
    else:
        collate_fn = None
        
    iterator_train = torch.utils.data.DataLoader(dataset_train, run_cfg.batch_size_train, collate_fn=collate_fn, shuffle=True)
    iterator_dev = torch.utils.data.DataLoader(dataset_dev, run_cfg.batch_size_eval, collate_fn=collate_fn)
    
    if run_cfg.pure_gnn:
        mol_graph_hidsize = molgraph_dict[list(molgraph_dict.keys())[0]].ndata['h'].shape[1]
    ## initialize a model
    start = Print('start initializing a model', output)
    models_list = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    ### model
    if not flag_rnn:                model = plus_tfm.PLUS_TFM(model_cfg)
    elif not flag_lm_model:         model = plus_rnn.PLUS_RNN(model_cfg)
    else:                           model = p_elmo.P_ELMo(model_cfg)
    models_list.append([model, "", flag_lm_model, flag_rnn, False])
    ### lm_model
    if flag_lm_model:
        lm_model = p_elmo.P_ELMo_lm(lm_model_cfg)
        models_list.append([lm_model, "lm", True, False, False])
    ### pr_model
    if flag_rnn:
        if run_cfg.pure_gnn:
            pr_model = gnn.PUREGNN(pr_model_cfg, mol_graph_hidsize)
        else:
            pr_model = gnn.GNN(pr_model_cfg, len(fingerprint_dict))
        models_list.append([pr_model, "pr", False, True, False])
    params, pr_params = [], []
    for model, idx, frz, _, _ in models_list:
        if frz: continue
        elif idx != "pr": params    += [p for p in model.parameters() if p.requires_grad]
        else:             pr_params += [p for p in model.parameters() if p.requires_grad]
    load_models(args, models_list, device, data_parallel, output, tfm_cls=flag_rnn)
    get_loss = plus_rnn.get_loss if flag_rnn else plus_tfm.get_loss
    end = Print('end initializing a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    optim = torch.optim.Adam([{'params':params,    'lr':run_cfg.learning_rate,      'weight_decay': 0.001   },
                              {'params':pr_params, 'lr':run_cfg.pr_learning_rate,   'weight_decay': 0.001   }])
    tasks_list = [] # list of lists [idx, metrics_train, metrics_eval]
    tasks_list.append(["cls", [], ["ci", "r2"]])
    if flag_lm_loss: tasks_list.append(["lm", [], ["acc"]])
    trainer = Trainer(models_list, get_loss, run_cfg, tasks_list, optim)
    trainer_args = {}
    trainer_args["data_parallel"] = data_parallel
    trainer_args["paired"] = False
    trainer_args["evaluate_cpi"] = True
    trainer_args["regression"] = True
    
    if   flag_rnn: trainer_args["projection"] = pr_model_cfg.projection
    if   flag_rnn: trainer_args["evaluate_cls"] = plus_rnn.evaluate_cls_dta
    else:          trainer_args["evaluate_cls"] = plus_tfm.evaluate_cls_protein
    end = Print('end setting trainer configurations', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## train a model
    start = Print('start training a model', output)
    cls_loss_list, lm_loss_list, train_loss_list = [], [], []
    Print(trainer.get_headline(), output)
    for epoch in range(run_cfg.num_epochs):
        ### train
        dataset_train.set_augment(flag_lm_loss)
        trainer.set_exec_flags(["cls", 'lm'], [True, True]) # AK
        trainer_args["class_weight"] = torch.Tensor([1.0, float(run_cfg.class_weight)])
        for B, batch in enumerate(iterator_train):
            batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
            trainer.train(batch, trainer_args)
            if B % 10 == 0: 
                Print('# epoch [{}/{}] train {:.1%} loss={:.4f}'.format(
                    epoch + 1, run_cfg.num_epochs, B / len(iterator_train), trainer.loss_train), output)
        print(' ' * 150, end='\r', file=sys.stderr)
        train_loss_list.append(trainer.loss_train)
        #trainer.gen_loss_graph(save_prefix, epoch, mode="train", title="Train")
        
        ### evaluate cls
        dataset_dev.set_augment(False)
        trainer.set_exec_flags(["cls", 'lm'], [True, False])
        _ = trainer_args.pop("class_weight", None)
        for b, batch in enumerate(iterator_dev):
            batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
            trainer.evaluate(batch, trainer_args)
            if b % 10 == 0: 
                Print('# cls {:.1%} loss={:.4f}'.format(
                    b / len(iterator_dev), trainer.loss_eval), output)
        print(' ' * 150, end='\r', file=sys.stderr)
        #cls_loss_list = cls_loss_list + trainer.loss_eval_list.copy()
        cls_loss_list.append(trainer.loss_eval)
        #trainer.gen_loss_graph(save_prefix, epoch, mode="eval", title="cls")
        trainer.reset_loss_list(eval_list=True)

        ### evaluate lm
        if flag_lm_loss:
            dataset_dev.set_augment(True)
            trainer.set_exec_flags(["cls", 'lm'], [False, True])
            for b, batch in enumerate(iterator_dev):
                batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
                trainer.evaluate(batch, trainer_args)
                if b % 10 == 0: 
                    Print('# lm {:.1%} loss={:.4f}'.format(
                        b / len(iterator_dev), trainer.loss_eval), output)
            print(' ' * 150, end='\r', file=sys.stderr)
            #trainer.gen_loss_graph(save_prefix, epoch, mode="eval", title="lm")
            lm_loss_list = lm_loss_list + trainer.loss_eval_list.copy()
            
        ### print log and save models
        trainer.save(save_prefix)
        Print(trainer.get_log(epoch + 1, args=trainer_args), output)
        trainer.set_exec_flags(["cls", "lm"], [True, True])
        trainer.reset()
        if trainer.patience == 0: break

    end = Print('end training a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)
    trainer.loss_train_list = train_loss_list
    #trainer.gen_loss_graph(save_prefix, epoch+1, mode="train", title="train_loss")
    trainer.loss_eval_list = cls_loss_list
    #trainer.gen_loss_graph(save_prefix, epoch+1, mode="eval", title="dev_loss_cls")
    trainer.gen_loss_graph(save_prefix, epoch+1, mode="both", title="train_dev_loss")
    if flag_lm_loss:
        trainer.loss_eval_list = lm_loss_list
        trainer.gen_loss_graph(save_prefix, epoch+1, mode="eval", title="lm")
    

if __name__ == '__main__':
    main()