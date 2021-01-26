# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# - Seonwoo Min's code (https://github.com/mswzeus/PLUS)

""" Utility functions """

import os
import sys
import math
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import auc, roc_auc_score, r2_score
from scipy.stats import pearsonr, spearmanr, t

import torch
import torch.nn as nn


def Print(string, output, newline=False):
    """ print to stdout and a file (if given) """
    time = datetime.now()
    print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=output)
        if newline: print("", file=output)

    output.flush()
    return time


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_output(args, string, test=False, embedding=False):
    """ set output configurations """
    output, save_prefix, index = sys.stdout, None, ""
    if args["output_path"] is not None:
        if not test and not embedding:
            if not os.path.exists(args["output_path"] + "/weights/"):
                os.makedirs(args["output_path"] + "/weights/", exist_ok=True)
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/weights/" + index

        elif not embedding:
            if not os.path.exists(args["output_path"]):
                os.makedirs(args["output_path"], exist_ok=True)
            if args["pretrained_model"] is not None:
                index += os.path.splitext(args["pretrained_model"])[0].split("/")[-1] + "_"
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/weights/"

        else:
            if not os.path.exists(args["output_path"] + "/embeddings/"):
                os.makedirs(args["output_path"] + "/embeddings/", exist_ok=True)
            if args["pretrained_model"] is not None:
                index += os.path.splitext(args["pretrained_model"])[0].split("/")[-1] + "_"
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/embeddings/"
            
    return output, save_prefix


def load_models(hp, models, device, output):
    """ load models if pretrained_models are available """
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_model" if idx == "" else "pretrained_%s_model" % idx
        idx_path = hp.get(idx)
        if idx_path is not None:
            Print('loading %s weights from %s' % (idx, hp[idx]), output)
            models[m][0].load_weights(hp[idx])

        models[m][0] = models[m][0].to(device)

def save_model(model, filename, output):
    Print('saving %s weights to %s' % (model.__class__.__name__, filename), output)
    torch.save(model.state_dict(), filename)
            
def evaluate_result(dict, metric):
    """ calculate the evaluation metric from the given results """
    result = None
    if metric   == "acc"  : result = (dict["correct"]) / (dict["n"]+ np.finfo(float).eps)
    elif metric == "pr"   : result = dict["tp"] / (dict["tp"] + dict["fp"]+ np.finfo(float).eps)
    elif metric == "re"   : result = dict["tp"] / (dict["tp"] + dict["fn"]+ np.finfo(float).eps)
    elif metric == "sp"   : result = dict["tn"] / (dict["tn"] + dict["fp"]+ np.finfo(float).eps)
    elif metric == "f1"   : result = 2 * dict["tp"] / (2 * dict["tp"] + dict["fp"] + dict["fn"]+ np.finfo(float).eps)
    elif metric == "mcc"  : result = ((dict["tp"] * dict["tn"] - dict["fp"] * dict["fn"]) /
                                    (math.sqrt((dict["tp"] + dict["fp"]) * (dict["tp"] + dict["fn"]) *
                                               (dict["tn"] + dict["fp"]) * (dict["tn"] + dict["fn"])) + np.finfo(float).eps))

    if result is None:
        if "labels" in dict and len(dict["labels"].shape) == 2: dict["labels"] = dict["labels"][:, 0]
        if "logits" in dict and len(dict["logits"].shape) == 2: dict["logits"] = dict["logits"][:, 0]
    if metric   == "auc"  : result = auc(dict["labels"], dict["logits"])
    elif metric == "aupr" : result = roc_auc_score(dict["labels"], dict["logits"])
    elif metric == "ci"   : result = ci(dict["labels"], dict["logits"])
    elif metric == "r2"   : result = r2_score(dict["labels"], dict["logits"])
    elif metric == "r"    : result = pearsonr(dict["labels"], dict["logits"])[0]
    elif metric == "rho"  : result = spearmanr(dict["labels"], dict["logits"])[0]
    return result


def steiger_test(xy, xz, yz, n):
    """ One-tailed Steiger's test for the statistic significance between two dependent correlation coefficients """
    # ab: correlation coefficient between a and b (xy, xz, yz)
    # n: number of elements in x, y and z
    d = xy - xz
    determin = 1 - xy ** 2 - xz ** 2 - yz ** 2 + 2 * xy * xz * yz
    av = (xy + xz)/2
    cube = (1 - yz) * (1 - yz) * (1 - yz)
    e = (n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + (av ** 2) * cube))
    if e < 0:
        return np.nan, np.nan
    t2 = d * np.sqrt(e)
    p = 1 - t.cdf(abs(t2), n - 2)

    return t2, p

def ci(y,f):
    y = y.numpy()
    f = f.numpy()
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def get_cindex(Y, P):
    Y = Y.numpy()
    P = P.numpy()

    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair != 0:
        return summ/pair
    else:
        return 0
    
def nan_hook(self, inp, output):
            if type(output) == torch.nn.utils.rnn.PackedSequence:
               outputs = [output.data]
            elif not isinstance(output, tuple):
                if type(output[0]) == torch.nn.utils.rnn.PackedSequence:
                    outputs = [output.data]
                else:
                    outputs = [output]
            else:
                if type(output[0]) == torch.nn.utils.rnn.PackedSequence:
                    outputs = [output[0].data]
                else:
                    outputs = output

            for i, out in enumerate(outputs):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
                inf_mask = torch.isinf(out)
                if inf_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found INF in output {i} at indices: ", inf_mask.nonzero(), "where:", out[inf_mask.nonzero()[:, 0].unique(sorted=True)])


def print_hp(hp, output):
    hplist = []
    hplist.append("Values of Hyperparameters:")
    for key, value in hp.items():
        hplist.append("{} : {}".format(key, value))
    for str in hplist:
        Print(str, output)