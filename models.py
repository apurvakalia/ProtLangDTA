# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)
# Some parts of the code were referenced from or inspired by below
# - Seonwoo Min's code (https://github.com/mswzeus/PLUS)


""" PLUS-RNN model classes and functions """

import sys
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dgl

from dgllife.model import GCN, GAT

class PROT_CNN(nn.Module):
    def __init__(self, hp, input_dim):
        super(PROT_CNN, self).__init__()
        self.embed = nn.Embedding(input_dim, hp['prot_embedding_dim'], padding_idx=input_dim - 1)
        self.conv_prot = nn.Conv1d(in_channels=hp['cnn_prot_length']+2, out_channels=hp['cnn_out_channels'], 
                                   kernel_size=hp['cnn_kernel_size'])
        self.fc_prot = nn.Linear(93, hp['prot_embedding_dim'])


    def forward(self, x, lengths):
        h = self.embed(x)
        h = self.conv_prot(h)

        # final projection
        z = self.fc_prot(h)

        return z, h

    def load_weights(self, pretrained_model):
        # load pretrained model weights
        state_dict = self.state_dict()
        for key, value in torch.load(pretrained_model, map_location=torch.device('cpu')).items():
            if key.startswith("module"): key = key[7:]
            if key in state_dict and value.shape == state_dict[key].shape: state_dict[key] = value
        self.load_state_dict(state_dict)

    
class PLUS_RNN(nn.Module):
    """ PLUS-RNN model """
    def __init__(self, hp, input_dim):
        super(PLUS_RNN, self).__init__()
        self.embed = nn.Embedding(input_dim, input_dim - 1, padding_idx=input_dim - 1)

        # bidirectional rnn
        self.rnn = nn.LSTM(input_dim - 1, hp['prot_hidden_dim'], hp['prot_num_layers'], bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hp['prot_hidden_dim'], hp['prot_embedding_dim'])

        # language modeling decoder
        self.decoder = nn.Linear(2 * hp['prot_hidden_dim'], input_dim-1)

        # ordinal classification
        # initialize ordinal regression bias with 10 following ICLR19_Bepler
        bias_init = 10
        if hp['num_classes'] is not None:
            self.ordinal_weight = nn.Parameter(torch.ones(1, hp['num_classes']))
            self.ordinal_bias = nn.Parameter(torch.zeros(hp['num_classes']) + bias_init)

    def forward(self, x, lengths):
        order = torch.argsort(lengths, descending=True)
        order_rev = torch.argsort(order)
        x, lengths = x[order], lengths[order]

        # feed-forward bidirectional RNN
        self.rnn.flatten_parameters()
        total_length = x.shape[1]
        h = self.embed(x)
        h = pack_padded_sequence(h, lengths, batch_first=True)
        h, _ = self.rnn(h)
        h = pad_packed_sequence(h, batch_first=True, total_length=total_length)[0][:, 1:-1][order_rev]

        # final projection
        z = self.fc(h)

        return z, h

    def load_weights(self, pretrained_model):
        # load pretrained model weights
        state_dict = self.state_dict()
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if len(loaded_dict) == 1: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): key = key[7:]
            if key in state_dict and value.shape == state_dict[key].shape: state_dict[key] = value
        self.load_state_dict(state_dict)

    def clip(self):
        # clip the weights of ordinal regression to be non-negative
        self.ordinal_weight.clamp(min=0)

class PROT_RNN(nn.Module):
    """ PLUS-RNN model """
    def __init__(self, hp, input_dim):
        super(PROT_RNN, self).__init__()
        self.embed = nn.Embedding(input_dim, input_dim - 1, padding_idx=input_dim - 1)

        # bidirectional rnn
        self.rnn = nn.LSTM(input_dim - 1, hp['prot_hidden_dim'], hp['prot_num_layers'], bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hp['prot_hidden_dim'], hp['prot_embedding_dim'])

        # language modeling decoder
        self.decoder = nn.Linear(2 * hp['prot_hidden_dim'], input_dim-1)

        # ordinal classification
        # initialize ordinal regression bias with 10 following ICLR19_Bepler
        bias_init = 10
        if hp['num_classes'] is not None:
            self.ordinal_weight = nn.Parameter(torch.ones(1, hp['num_classes']))
            self.ordinal_bias = nn.Parameter(torch.zeros(hp['num_classes']) + bias_init)

    def forward(self, x, lengths):
        order = torch.argsort(lengths, descending=True)
        order_rev = torch.argsort(order)
        x, lengths = x[order], lengths[order]

        # feed-forward bidirectional RNN
        self.rnn.flatten_parameters()
        total_length = x.shape[1]
        h = self.embed(x)
        h = pack_padded_sequence(h, lengths, batch_first=True)
        h, _ = self.rnn(h)
        h = pad_packed_sequence(h, batch_first=True, total_length=total_length)[0][:, 1:-1][order_rev]

        # final projection
        z = self.fc(h)

        return z, h

    def load_weights(self, pretrained_model):
        # load pretrained model weights
        state_dict = self.state_dict()
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if len(loaded_dict) == 1: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): key = key[7:]
            if key in state_dict and value.shape == state_dict[key].shape: state_dict[key] = value
        self.load_state_dict(state_dict)

    def clip(self):
        # clip the weights of ordinal regression to be non-negative
        self.ordinal_weight.clamp(min=0)

class MOL_GNN(nn.Module):
    def __init__(self, hp, in_dim):
        super(MOL_GNN, self).__init__()
        dropout = [hp['gnn_dropout'] for _ in range(len(hp['gnn_channels']))]
        batchnorm = [False for _ in range(len(hp['gnn_channels']))]
        gnn_map = {
            "gcn": GCN(in_dim, hp['gnn_channels'], batchnorm = batchnorm),
            "gat": GAT(in_dim, hp['gnn_channels'], hp['attn_heads'])
        }
        self.GNN = gnn_map[hp['gnn_type']]
        self.pool = dgl.nn.pytorch.glob.MaxPooling()
        self.mlp_layers_pool = nn.ModuleList()
        self.mlp_layers_graph = nn.ModuleList()
 
        self.mlp_layers_pool.append(nn.Linear(hp['gnn_channels'][len(hp['gnn_channels']) - 1], hp['gnn_hidden_dim'] * 2))
        for i in range(hp['num_mlp_layers'] - 1):
            self.mlp_layers_pool.append(nn.Linear(hp['gnn_hidden_dim'] * 2, hp['gnn_hidden_dim'] * 2))
            
        self.mlp_layers_graph.append(nn.Linear(hp['gnn_channels'][len(hp['gnn_channels']) - 1], hp['gnn_hidden_dim'] * 2))
        for i in range(hp['num_mlp_layers'] - 1):
            self.mlp_layers_graph.append(nn.Linear(hp['gnn_hidden_dim'] * 2, hp['gnn_hidden_dim'] * 2))
            
        self.gnn_final_pool = nn.Linear(hp['gnn_hidden_dim'] * 2, hp['prot_embedding_dim'])
        self.gnn_final_graph = nn.Linear(hp['gnn_hidden_dim'] * 2, hp['prot_embedding_dim'])
        self.W_attention_prot = nn.Linear(hp['prot_embedding_dim'], hp['prot_embedding_dim'])
        self.W_attention_comp = nn.Linear(hp['prot_embedding_dim'], hp['prot_embedding_dim'])
        self.W_attention_comp_graph = nn.Linear(hp['prot_embedding_dim'], hp['prot_embedding_dim'])
        self.norm_comp = nn.BatchNorm1d(hp['prot_embedding_dim'])
        self.norm_prot = nn.BatchNorm1d(hp['prot_embedding_dim'])
        self.W_out = nn.Linear(2 * hp['prot_embedding_dim'], hp['num_classes'])
        #self.W_out1 = nn.Linear(2 * hp['prot_embedding_dim'], hp['gnn_hidden_dim'] * 2)
        #self.W_out2 = nn.Linear(hp['gnn_hidden_dim'] * 2, hp['gnn_hidden_dim'])
        #self.W_out3 = nn.Linear(hp['gnn_hidden_dim'], hp['num_classes'])
        self.attn_type = hp['attn_type']
        self.embedding_dim = hp['prot_embedding_dim']

        
    def graphembedding(self, g, f):
        f = self.GNN(g, f)
        h = self.pool(g, f)
        return h, f

    def cross_attention(self, xg, xgraph, g, gf, xs):
        save_xg = xg
        save_xs = xs
        xs_sum_orig = torch.sum(xs, 1)
        xs_sum = torch.unsqueeze(xs_sum_orig, 2)
        xsub_graphs = dgl.unbatch(g)
        xg = torch.relu(self.W_attention_comp(xg))
        xgraph = torch.relu(self.W_attention_comp_graph(xgraph))
        xg = torch.unsqueeze(xg, 2)
        hs = torch.relu(self.W_attention_prot(xs))
        weights_prot = torch.tanh(torch.matmul(hs, xg))
        #weights_prot = torch.nn.Softmax(torch.matmul(hs, xg) / torch.sqrt(hs.shape[-1]))
        xs = weights_prot * hs
        cnt = 0
        xg_attn = save_xg.detach().clone()
        for i in range(len(gf)):
            xgraph_sub = xgraph[cnt:cnt+gf[i],:]
            xs_sum_sub = xs_sum[i,:,:]
            weights_comp = torch.tanh(torch.matmul(xgraph_sub, xs_sum_sub))
            #weights_comp = torch.nn.Softmax(torch.matmul(xgraph_sub, xs_sum_sub) / torch.sqrt(xgraph_sub.shape[-1]))
            xgraph_sub = xgraph_sub * weights_comp
            xg_attn[i,:] = self.pool(xsub_graphs[i], xgraph_sub)
            cnt = cnt + gf[i]
               
        #xg = torch.t(weights) * xg
        
        xs = torch.sum(xs, 1)
        xg = torch.squeeze(xg)
        if self.attn_type == 0:
            comp, protein = save_xg, xs_sum_orig
        elif self.attn_type == 1:
            comp, protein = save_xg, xs
        elif self.attn_type == 2:
            comp, protein = xg_attn, xs_sum_orig
        elif self.attn_type == 3:
            comp, protein = xg_attn, xs
        
        comp = self.norm_comp(comp)
        protein = self.norm_prot(protein)
        y_cat = torch.cat((comp, protein), 1).squeeze(0)
            
        return y_cat
    
    def forward(self, g, f, prot_bat):
        h, f = self.graphembedding(g, f)
        g_features = g.batch_num_nodes()
        
        for mlp in self.mlp_layers_pool:
            h = mlp(h)
            
        h = F.relu(self.gnn_final_pool(h))
        
        for mlp in self.mlp_layers_graph:
            f = mlp(f)
            
        f = F.relu(self.gnn_final_graph(f))
            
        y_cat = self.cross_attention(h, f, g, g_features, prot_bat) 
        #if not self.cpi:
            #y_cat = F.relu(y_cat)
        #h = self.W_out1(y_cat)
        #h = self.W_out2(h)
        #z_interaction = self.W_out3(h)
        z_interaction = self.W_out(y_cat)


        return z_interaction

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if len(loaded_dict) == 1: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)

def get_loss(batch, models_dict, cfg, tasks_dict, args, test=False):
    """ feed-forward and evaluate PLUS_RNN model """
    models, models_idx, tasks_idx = models_dict["model"], models_dict["idx"], tasks_dict["idx"]
    cpi_pred = False
    if not test: tasks_flag = tasks_dict["flags_train"]
    else:        tasks_flag = tasks_dict["flags_eval"]
    if   len(batch) == 2:
        tokens0, lengths0 = batch
        pelmo_lm_training, pair = True, False
    elif len(batch) == 3:
        tokens0, lengths0, labels = batch
        pelmo_lm_training, pair = False, False
    elif len(batch) == 5:
        if args["evaluate_cpi"]:
            cpi_pred = True
            (compounds, adjacencies, tokens0, lengths0, labels) = batch
            pelmo_lm_training, pair = False, False
        else:
            if len(batch[3].shape) == 1:
                tokens0, lengths0, tokens1, lengths1, labels = batch
                pelmo_lm_training, pair = False, True
            else:
                tokens0, lengths0, masked_pos0, masked_tokens0, masked_weights0 = batch
                pelmo_lm_training, pair = False, False
    elif len(batch) == 6:
        (tokens0, lengths0, masked_pos0, masked_tokens0, masked_weights0, labels) = batch
        pelmo_lm_training, pair = False, False
    elif len(batch) == 7:
        tokens0, lengths0, tokens1, lengths1, labels, cmaps0, cmaps1 = batch
        pelmo_lm_training, pair = False, True
    elif len(batch) == 8:
        cpi_pred = True
        (compounds, adjacencies, tokens0, lengths0, masked_pos0, masked_tokens0, masked_weights0, labels) = batch
        pelmo_lm_training, pair = False, False
    elif len(batch) == 11:
        (tokens0, lengths0, masked_pos0, masked_tokens0, masked_weights0,
         tokens1, lengths1, masked_pos1, masked_tokens1, masked_weights1, labels) = batch
        pelmo_lm_training, pair = False, True
    elif len(batch) == 13:
        (tokens0, lengths0, masked_pos0, masked_tokens0, masked_weights0,
         tokens1, lengths1, masked_pos1, masked_tokens1, masked_weights1, labels, cmaps0, cmaps1) = batch
        pelmo_lm_training, pair = False, True


    # compute protein representations
    if pelmo_lm_training:
        model = models[models_idx.index("")]
    elif "lm" not in models_idx:
        model = models[models_idx.index("")]
        z0, r0 = model(tokens0, lengths0)
        if pair: z1, r1 = model(tokens1, lengths1)
    else:
        model, model_lm = models[models_idx.index("")], models[models_idx.index("lm")]
        if args["data_parallel"]: tokens0_lm = model_lm.module.encode(tokens0, lengths0)
        else:                     tokens0_lm = model_lm.encode(tokens0, lengths0)
        z0, r0 = model(tokens0, tokens0_lm, lengths0)

        if pair:
            if args["data_parallel"]: tokens1_lm = model_lm.module.encode(tokens1, lengths1)
            else:                     tokens1_lm = model_lm.encode(tokens1, lengths1)
            z1, r1 = model(tokens1, tokens1_lm, lengths1)

    results = []
    for task_idx, flag in zip(tasks_idx, tasks_flag):
        if task_idx not in ["lm", "cls", "cm"]: sys.exit("# ERROR: invalid task index [%s]; Supported indices are [lm, cls, cm]" % task_idx)

        result = {"n": 0, "avg_loss": 0}
        if pelmo_lm_training and task_idx == "lm" and flag["exec"]:
            logits0_lm = model(tokens0, lengths0)
            result = evaluate_lm_pelmo(logits0_lm, tokens0, flag, args["num_alphabets"])
            if cfg.lm_loss_lambda != -1: result["avg_loss"] = result["avg_loss"] * cfg.lm_loss_lambda

        elif task_idx == "lm" and flag["exec"]:
            logits0_lm = model.module.lm(r0, masked_pos0) if args["data_parallel"] else model.lm(r0, masked_pos0)
            result = evaluate_lm(logits0_lm, masked_tokens0, masked_weights0, flag)
            if cfg.lm_loss_lambda != -1: result["avg_loss"] = result["avg_loss"] * cfg.lm_loss_lambda
            if pair:
                logits1_lm = model.module.lm(r1, masked_pos1) if args["data_parallel"] else model.lm(r1, masked_pos1)
                result1 = evaluate_lm(logits1_lm, masked_tokens1, masked_weights1, flag)
                if cfg.lm_loss_lambda != -1: result1["avg_loss"] = result["avg_loss"] * cfg.lm_loss_lambda
                for k in result.keys(): result[k] += result1[k]

        elif (task_idx == "cls" and flag["exec"]) or (task_idx == "cm" and flag["exec"]):
            if pair:
                z0_list = model.module.em(z0, lengths0) if args["data_parallel"] else model.em(z0, lengths0)
                z1_list = model.module.em(z1, lengths1) if args["data_parallel"] else model.em(z1, lengths1)
            elif args["projection"]:
                z0_list = model.module.em(z0, lengths0) if args["data_parallel"] else model.em(z0, lengths0)
            else:
                r0_list = model.module.em(r0, lengths0) if args["data_parallel"] else model.em(r0, lengths0)

            if task_idx == "cls" and flag["exec"]:
                if pair:
                    logits_cls = model.module.sm(z0_list, z1_list) if args["data_parallel"] else model.sm(z0_list, z1_list)
                    result = args["evaluate_cls"](logits_cls, labels, flag)
                    if cfg.cls_loss_lambda != -1: result["avg_loss"] = result["avg_loss"] * cfg.cls_loss_lambda
                else:
                    if cpi_pred:
                        if cfg.pure_gnn:
                            logits_cls = models[models_idx.index("pr")](compounds, compounds.ndata['h'], z0)
                        else:
                            logits_cls = models[models_idx.index("pr")](compounds, adjacencies, z0)                            
                    else:
                        if args["projection"]: logits_cls = models[models_idx.index("pr")](z0_list)
                        else:                  logits_cls = models[models_idx.index("pr")](r0_list)
                    result = args["evaluate_cls"](logits_cls, labels, flag, args, cfg.pure_gnn)
                    if cfg.cls_loss_lambda != -1: result["avg_loss"] = result["avg_loss"] * cfg.cls_loss_lambda

            elif task_idx == "cm" and flag["exec"]:
                model_cm = models[models_idx.index("cm")]
                selection = np.random.choice(len(z0_list), cfg.cm_batch_size, replace=False)
                logits0_cm = torch.cat([model_cm(z0_list[s].unsqueeze(0)).view(-1) for s in selection], 0)
                labels0_cm = torch.cat([cmaps0[s].view(-1).to(logits0_cm.device) for s in selection], 0)
                result = evaluate_cm(logits0_cm, labels0_cm, flag)
                if cfg.cm_loss_lambda != -1: result["avg_loss"] = result["avg_loss"] * cfg.cm_loss_lambda
                if pair:
                    logits1_cm = torch.cat([model_cm(z1_list[s].unsqueeze(0)).view(-1) for s in selection], 0)
                    labels1_cm = torch.cat([cmaps1[s].view(-1).to(logits1_cm.device) for s in selection], 0)
                    result1 = evaluate_cm(logits1_cm, labels1_cm, flag)
                    if cfg.cm_loss_lambda != -1: result1["avg_loss"] = result["avg_loss"] * cfg.cm_loss_lambda
                    for k in result.keys(): result[k] += result1[k]

        results.append(result)

    return results

def get_embedding(batch, models_dict, args):
    """ get protein embeddings from PLUS_RNN model """
    models, models_idx = models_dict["model"], models_dict["idx"]
    tokens, lengths = batch

    # compute protein representations
    if "lm" not in models_idx:
        model = models[models_idx.index("")]
        z, r = model(tokens, lengths)
    else:
        model, model_lm = models[models_idx.index("")], models[models_idx.index("lm")]
        if args["data_parallel"]: tokens_lm = model_lm.module.encode(tokens, lengths)
        else:                     tokens_lm = model_lm.encode(tokens, lengths)
        z, r = model(tokens, tokens_lm, lengths)

    z_list = model.module.em(z, lengths, cpu=True) if args["data_parallel"] else model.em(z, lengths, cpu=True)
    r_list = model.module.em(r, lengths, cpu=True) if args["data_parallel"] else model.em(r, lengths, cpu=True)
    embeddings = [z_list, r_list]

    return embeddings

def evaluate_lm_pelmo(logits_lm, tokens, flag, num_alphabets):
    """ evaluate language modeling """
    result = {}

    mask = (tokens != 0) * (tokens != (num_alphabets - 1))
    result["n"] = torch.sum(mask).item()
    loss_lm = -logits_lm.gather(2, (tokens * mask.long()).unsqueeze(2)).squeeze(2)
    result["avg_loss"] = torch.mean(loss_lm.masked_select(mask))

    if flag["acc"]:
        _, tokens_hat = torch.max(logits_lm, 2)
        result["correct"] = torch.sum((tokens_hat == tokens).masked_select(mask)).item()

    return result


def evaluate_lm(logits_lm, masked_tokens, masked_weights, flag):
    """ evaluate (masked) language modeling """
    result = {}

    result["n"] = torch.sum(masked_weights).item()
    loss_lm = -logits_lm.gather(2, masked_tokens.unsqueeze(2)).squeeze()
    result["avg_loss"] = torch.mean(loss_lm.masked_select(masked_weights))

    if flag["acc"]:
        _, masked_tokens_hat = torch.max(logits_lm, 2)
        result["correct"] = torch.sum((masked_tokens_hat == masked_tokens).masked_select(masked_weights)).item()

    return result


def evaluate_sfp(logits_cls, labels, flag):
    """ evaluate same family prediction """
    result = {}

    result["n"] = len(logits_cls)
    result["avg_loss"] = F.binary_cross_entropy_with_logits(logits_cls, labels.float())

    if flag["acc"]:
        samefamily_hat = logits_cls > 0.5
        result["correct"] = torch.sum((samefamily_hat == labels)).item()

    return result


def evaluate_homology(logits_cls, labels, flag):
    """ evaluate protein pair structural similarity classification """
    result = {}

    result["n"] = len(logits_cls)
    result["avg_loss"] =  F.binary_cross_entropy_with_logits(logits_cls, labels.float())

    if flag["acc"] or flag["pred"]:
        prob_cls = torch.sigmoid(logits_cls)
        ones = logits_cls.new_ones(prob_cls.size(0), 1)
        prob_cls_ge = torch.cat([ones, prob_cls], 1)
        prob_cls_lt = torch.cat([1 - prob_cls, ones], 1)
        prob_cls = prob_cls_ge * prob_cls_lt
        similarity_levels_hat = prob_cls / prob_cls.sum(1, keepdim=True)

        if flag["acc"]:
            _, similarity_hat = torch.max(similarity_levels_hat, 1)
            similarity = torch.sum(labels, 1)
            result["correct"] = torch.sum((similarity_hat == similarity)).item()
        if flag["pred"]:
            result["logits"] = [similarity_levels_hat.cpu()]
            result["labels"] = [similarity.cpu()]

    return result


def evaluate_cls_protein(logits_cls, labels, flag, args):
    """ evaluate protein-level classification task """
    result = {}
    logits_cls, labels = torch.stack(logits_cls, 0), labels[:, 0]

    result["n"] = len(logits_cls)
    if "regression" in args and args["regression"]:
        result["avg_loss"] = F.mse_loss(logits_cls, labels)
    else:
        class_weight = args["class_weight"].to(logits_cls.device) if "class_weight" in args else None
        result["avg_loss"] = F.cross_entropy(logits_cls, labels, weight=class_weight)

    if flag["acc"]:
        _, labels_hat = torch.max(logits_cls, 1)
        result["correct"] = torch.sum((labels_hat == labels)).item()
    if flag["pred"]:
        result["logits"] = [logits_cls.cpu()]
        result["labels"] = [labels.cpu()]

    return result

def evaluate_cls_cpi(logits_cls, labels, flag, args, pure_gnn=False):
    """ evaluate CPI interaction classification task """
    result = {}
    if not pure_gnn:
        logits_cls = torch.stack(logits_cls, 0)
        
    labels = labels[:, 0]

    result["n"] = len(logits_cls)
    if "regression" in args and args["regression"]:
        result["avg_loss"] = F.mse_loss(logits_cls, labels)
    else:
        class_weight = args["class_weight"].to(logits_cls.device) if "class_weight" in args else None
        result["avg_loss"] = F.cross_entropy(logits_cls, labels, weight=class_weight)

    if flag["acc"]:
        _, labels_hat = torch.max(logits_cls, 1)
        result["correct"] = torch.sum((labels_hat == labels)).item()
    if flag["pred"]:
        result["logits"] = [logits_cls.cpu()]
        result["labels"] = [labels.cpu()]

    return result

def evaluate_cls_dta(logits_cls, labels, flag, args, pure_gnn=False):
    """ evaluate CPI interaction classification task """
    result = {}
    if not pure_gnn:
        logits_cls = torch.stack(logits_cls, 0)
    else:
        logits_cls = logits_cls[:,0]
        
    labels = labels[:, 0]

    result["n"] = len(logits_cls)
    if "regression" in args and args["regression"]:
        result["avg_loss"] = F.mse_loss(logits_cls, labels)
    else:
        class_weight = args["class_weight"].to(logits_cls.device) if "class_weight" in args else None
        result["avg_loss"] = F.cross_entropy(logits_cls, labels, weight=class_weight)

    if flag["acc"]:
        _, labels_hat = torch.max(logits_cls, 1)
        result["correct"] = torch.sum((labels_hat == labels)).item()
    if flag["pred"]:
        result["logits"] = [logits_cls.cpu()]
        result["labels"] = [labels.cpu()]

    return result

def evaluate_cls_amino(logits_cls, labels, flag, args):
    """ evaluate amino-acid-level classification task """
    result = {}

    class_weight = args["class_weight"].to(logits_cls[0].device) if "class_weight" in args else None
    n, avg_loss, correct = 0, 0, 0
    for i in range(len(logits_cls)):
        length = len(logits_cls[i][0])
        n += length
        avg_loss += F.cross_entropy(logits_cls[i][0], labels[i][:length], weight=class_weight) * length

        if flag["acc"]:
            _, labels_hat = torch.max(logits_cls[i][0], 1)
            correct += torch.sum((labels_hat == labels[i][:length])).item()

    result["n"] = n
    result["avg_loss"] = avg_loss / n

    if flag["acc"]:
        result["correct"] = correct
    if flag["pred"]:
        result["logits"] = [logit_cls.cpu() for logit_cls in logits_cls]
        result["labels"] = [labels.cpu()]

    return result


def evaluate_transmembrane(logits_cls, labels, flag, args):
    """ evaluate transmembrane classification task """
    grammar = Grammar()
    result = {}

    n, avg_loss, correct = 0, 0, 0
    n_p, correct_p = len(logits_cls), 0
    for i in range(len(logits_cls)):
        length = len(logits_cls[i][0])
        n += length
        avg_loss += F.cross_entropy(logits_cls[i][0], labels[i][:length]) * length

        if flag["acc"]:
            _, labels_hat = torch.max(logits_cls[i][0], 1)
            correct += torch.sum((labels_hat == labels[i][:length])).item()

            log_p_hat = F.log_softmax(logits_cls[i][0], 1).detach().cpu().numpy()
            label_hat, _ = grammar.decode(log_p_hat)
            correct_p += is_prediction_correct(label_hat, labels[i][:length])

    result["n"] = n
    result["avg_loss"] = avg_loss / n

    if flag["acc"]:
        result["correct"] = correct
        result["acc_p"] = float(correct_p) / float(n_p)
    if flag["pred"]:
        result["logits"] = [logit_cls.cpu() for logit_cls in logits_cls]
        result["labels"] = [labels.cpu()]

    return result


def evaluate_cm(logits_cm, labels_cm, flag):
    """ evaluate contact map prediction """
    mask = (labels_cm < 0)
    logits_cm, labels_cm = logits_cm[~mask], labels_cm[~mask]
    result = {}

    result["n"] = mask.sum()
    result["avg_loss"] = F.binary_cross_entropy_with_logits(logits_cm, labels_cm)

    if flag["acc"]:
        predictions = logits_cm > 0.5
        result["correct"] = torch.sum((predictions == labels_cm)).item()
    elif flag["conf"]:
        predictions = logits_cm > 0.5
        result["tn"], result["fp"], result["fn"], result["tp"] = confusion_matrix(labels_cm.cpu(), predictions.cpu()).ravel()

    return result
