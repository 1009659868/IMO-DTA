import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import os
from tqdm import tqdm
from math import exp

os.environ['PYTHONHASHSEED'] = '0'
import matplotlib
import torch.nn.functional as F

matplotlib.use('Agg')

from dataparser import *
from arguments import argparser, logging
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from metrics import *
from model import net
from sklearn.metrics import roc_auc_score, accuracy_score
from utils.run_helper import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

auc_score=None

def show_device():
    torch.cuda.init()  
    if torch.cuda.is_available():
        gpu_num= torch.cuda.device_count()
        for i in range(gpu_num):
            print("\t GPU {}.: {}".format(i, torch.cuda.get_device_name(i)))

def get_device():
    show_device()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda")  
    else:
        device = torch.device("cpu")
    return device

device = get_device()

def get_random_folds(tsize, foldcount):
    folds = []
    indices = set(range(tsize))
    foldsize = tsize / foldcount
    leftover = tsize % foldcount
    for i in range(foldcount):
        sample_size = foldsize
        if leftover > 0:
            sample_size += 1
            leftover -= 1
        fold = random.sample(indices, int(sample_size))
        indices = indices.difference(fold)
        folds.append(fold)

    # assert stuff
    foldunion = set([])
    for find in range(len(folds)):
        fold = set(folds[find])
        assert len(fold & foldunion) == 0, str(find)
        foldunion = foldunion | fold
    assert len(foldunion & set(range(tsize))) == tsize
    return folds


def get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount):
    assert len(np.array(label_row_inds).shape) == 1, 'label_row_inds should be one dimensional array'
    row_to_indlist = {}
    rows = sorted(list(set(label_row_inds)))
    for rind in rows:
        alloccs = np.where(np.array(label_row_inds) == rind)[0]
        row_to_indlist[rind] = alloccs
    drugfolds = get_random_folds(drugcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        drugfold = drugfolds[foldind]
        for drugind in drugfold:
            fold = fold + row_to_indlist[drugind].tolist()
        folds.append(fold)
    return folds


def get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount):
    assert len(np.array(label_col_inds).shape) == 1, 'label_col_inds should be one dimensional array'
    col_to_indlist = {}
    cols = sorted(list(set(label_col_inds)))
    for cind in cols:
        alloccs = np.where(np.array(label_col_inds) == cind)[0]
        col_to_indlist[cind] = alloccs
    target_ind_folds = get_random_folds(targetcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        targetfold = target_ind_folds[foldind]
        for targetind in targetfold:
            fold = fold + col_to_indlist[targetind].tolist()
        folds.append(fold)
    return folds


def loss_f(predict_x, x, mu, logvar):
    # Adjust the shapes of predict_x and x to ensure a match
    predict_x = predict_x.view(x.size())
    #recon_loss = F.mse_loss(predict_x, x, reduction='sum')
    elementwise_mse = F.mse_loss(predict_x, x, reduction='none')

    # sum the seven values in each row
    recon_loss = torch.sum(elementwise_mse, dim=1)
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

    return torch.mean(recon_loss + KLD)

def prepare():
    Tags = argparser()
    Tags.log_dir = Tags.log_dir + "/pre/tmp" + str(time.time()) + "\\"

    if not os.path.exists(Tags.log_dir):
        os.makedirs(Tags.log_dir)

    logging(str(Tags), Tags)
    print("select the dataset :\n")
    print("1.davis (input 1 or davis or d)")
    print("2.kiba  (input 2 or kiba or k)")

    select = input("choose or defalut(kiva):")
    if (select == '1' or select == 'd' or select == 'davis'):
        Tags.dataset_path = './data/davis/'
        print("dataset:davis")
    elif (select == '2' or select == 'k' or select == 'kiba'):
        Tags.dataset_path = './data/kiba/'
        print("dataset:kiba")
    else:
        print("select defalut dataset:kiba")

    global auc_score
    if 'davis' in Tags.dataset_path:
        auc_score = 7
    else:
        auc_score = 12.1

    return Tags

def initialize_dataset(Tags):
    dataset = DataSet(
        fpath=Tags.dataset_path,
        setting_no=Tags.problem_type,
        seqlen=Tags.max_seq_len,
        smilen=Tags.max_smi_len,
        need_shuffle=False
    )

    # Set character set size
    Tags.charseqset_size = dataset.charseqset_size
    Tags.charsmiset_size = dataset.charsmiset_size

    Drug, Target, Affinity, Atom, Protein = dataset.parse_data(Tags)

    # Convert to numpy arrays
    Drug, Target, Affinity, Atom, Protein = map(np.asarray, [Drug, Target, Affinity, Atom, Protein])

    # Update Tags with drug and target counts
    Tags.drug_count = Drug.shape[0]
    Tags.target_count = Target.shape[0]

    label_row_inds, label_col_inds = np.where(~np.isnan(Affinity))

    return Drug, Target, Affinity, Atom, Protein, label_row_inds, label_col_inds, Tags.drug_count, Tags.target_count

def train(train_loader, model, Tags, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2, lamda):
    model.train()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0002)
    with tqdm(train_loader) as t:
        for drug_SMILES, target_protein, affinity, atom_feature, res_feature in t:
            res_feature = pad_sequence(res_feature, batch_first=True, padding_value=0)
            # 下游任务,训练
            drug_SMILES = torch.Tensor(drug_SMILES).float().to(device)
            
            target_protein = torch.Tensor(target_protein).float().to(device)
            
            affinity = torch.Tensor(affinity).float().to(device)
            
            atom_feature = torch.Tensor(atom_feature).float().to(device)
           
            res_feature = torch.Tensor(res_feature).float().to(device)
            
            optimizer.zero_grad()
            
            affinity = Variable(affinity).to(device)

            pre_affinity, pre_atom_feature, pre_res_feature, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(
                drug_SMILES, target_protein, Tags, NUM_FILTERS,
                FILTER_LENGTH1, FILTER_LENGTH2)
            loss_affinity = loss_func(pre_affinity, affinity)

            loss_drug = loss_f(pre_atom_feature, atom_feature, mu_drug, logvar_drug)

            loss_target = loss_f(pre_res_feature, res_feature, mu_target, logvar_target)

            c_index = get_cindex(affinity.cpu().detach().numpy(),
                                 pre_affinity.cpu().detach().numpy())
            torch.cuda.empty_cache()  # manually_release_cuda_memory
            loss = loss_affinity + 10 ** lamda * (loss_drug + loss_target)

            loss.backward()
            optimizer.step()
            
            mse = loss_affinity.item()
            t.set_postfix(train_loss=loss.item(), mse=mse, train_cindex=c_index)

    return model

def evaluate(model, test_loader, Tags, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2, lamda):
    model.eval()
    loss_func = nn.MSELoss()
    affinities = []
    pre_affinities = []
    loss_d = 0
    loss_t = 0

    # create_error_log
    error_file_path = "./error_log.txt"
    error_log = open(error_file_path, "a")
    with torch.no_grad():
        for i, (drug_SMILES, target_protein, affinity, atom_feature, res_feature) in enumerate(test_loader):

            drug_SMILES = torch.Tensor(drug_SMILES).to(device)
            target_protein = torch.Tensor(target_protein).to(device)
            affinity = torch.Tensor(affinity).to(device)
            atom_feature = torch.Tensor(atom_feature).to(device)
            res_feature = torch.Tensor(res_feature).to(device)

            try:
                pre_affinity, pre_atom_features, pre_res_features, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(
                    drug_SMILES, target_protein, Tags, NUM_FILTERS,
                    FILTER_LENGTH1, FILTER_LENGTH2)

                pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
                affinities += affinity.cpu().detach().numpy().tolist()
                loss_d += loss_f(pre_atom_features, atom_feature, mu_drug, logvar_drug)
                loss_t += loss_f(pre_res_features, res_feature, mu_target, logvar_target)


            except Exception as e:
                # Record error information to the error log when an exception occurs
                error_log.write(f"Error occurred in iteration {i}:\n")
                error_log.write(f"Exception: {str(e)}\n")
                error_log.write(f"drug_SMILES: {drug_SMILES}\n")
                error_log.write(f"target_protein: {target_protein}\n")
                error_log.write(f"affinity: {affinity}\n")
                error_log.write(f"atom_feature: {atom_feature}\n")
                error_log.write(f"protein_feature: {res_feature}\n\n")
        # turn_off_error_log
        error_log.close()
        
        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)

        loss = loss_func(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        cindex = get_cindex(affinities, pre_affinities)
        r2=get_r2(affinities, pre_affinities)
        rm2 = get_rm2(affinities, pre_affinities)
        
        y_ture=np.int32(affinities > auc_score)
        y_score=pre_affinities
        auc = roc_auc_score(y_ture, y_score)
        
    return cindex, loss,r2, rm2, auc, loss_d, loss_t

def optimize_hyperparameters(Drug, Target, Affinity, label_row_inds, label_col_inds, Tags, labeled_sets, val_sets):
    paramset1 = Tags.num_windows
    paramset2 = Tags.smi_window_lengths
    paramset3 = Tags.seq_window_lengths
    lamda_set = Tags.lamda
    batchsz = Tags.batch_size  # 256

    logging("---Parameter Search-----", Tags)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3) * len(lamda_set)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]

    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1value in paramset1:
        for param2value in paramset2:
            for param3value in paramset3:
                for lamda in lamda_set:
                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)

                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1value, param2value, param3value, lamda, ]

                    pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses

def optimal_param_training(Drug, Target, Affinity, Atom, Protein, label_row_indexs, label_col_indexs, Tags, labeled_sets, val_sets, best_param_list, i):
    param1value, param2value, param3value, lamda = best_param_list
    batchsz = Tags.batch_size

    logging("---Parameter Search-----", Tags)

    w = len(val_sets)

    all_predictions, all_losses, all_auc, all_aupr = [0] * w, [0] * w, [0] * w, [0] * w
    all_preaffinities, all_affinities = [], []

    for foldindex in range(len(val_sets)):
        valindexs, labeledindexs = val_sets[foldindex], labeled_sets[foldindex]
        train_loader, test_loader = prepare_loaders(Drug, Target, Affinity, Atom, Protein, label_row_indexs,
                                                    label_col_indexs, labeledindexs, valindexs, batchsz)
        model = initialize_model(Tags, param1value, param2value, param3value)

        cindex_list = train_and_evaluate(model, train_loader, test_loader, Tags, param1value, param2value, param3value,
                                         lamda)

        affinities, pre_affinities = test_model(model, test_loader, Tags, param1value, param2value, param3value)

        mse, mae, rm2, r2, cindex, auc, aupr, loss = calculate_and_log_metrics(affinities, pre_affinities, Tags,
                                                                               param1value, param2value, param3value,
                                                                               foldindex)

        all_predictions[foldindex], all_losses[foldindex] = cindex, loss
        all_auc[foldindex], all_aupr[foldindex] = auc, aupr
        all_affinities.append(affinities)
        all_preaffinities.append(pre_affinities)

    # save affinities and preaffinites for further analysis
    save_affinities(all_affinities, all_preaffinities, i)

    best_param_list = [param1value, param2value, param3value, lamda]
    best_perf = np.mean(all_predictions)

    return best_param_list, best_perf, all_predictions, all_losses, all_auc, all_aupr

def perform_cross_validation(Drug, Target, Affinity, Atom, Protein, label_row_inds, label_col_inds, Tags, nfolds, i):
    """
        Perform n-fold cross-validation and return performance metrics.

        Args:
            Drug: Array of drug data.
            Target: Array of target data.
            Affinity: Array of affinity data.
            Atom: atom's properties.
            Protein: protein's properties.
            label_row_inds: Array of label row indices.
            label_col_inds: Array of label column indices.
            Tags: Configuration or tags containing various parameters.
            nfolds: List of folds for cross-validation.
            i: Index of the current fold.

        Returns:
            avg_perf: Average performance.
            avg_loss: Average loss.
            perf_std: Standard deviation of performance.
            loss_std: Standard deviation of loss.
            avg_auc: Average area under the ROC curve.
            auc_std: Standard deviation of AUC.
            avg_aupr: Average area under the Precision-Recall curve.
            aupr_std: Standard deviation of AUPR.
    """
    test_set = nfolds[-1]
    outer_train_sets = nfolds[:-1]

    val_sets, train_sets, test_sets = create_train_val_test_sets(test_set=test_set, outer_train_sets=outer_train_sets)

    best_param_list = optimize_hyperparameters_wrapper(
        Drug, Target, Affinity, label_row_inds, label_col_inds, Tags, train_sets, test_sets)

    metrics = optimal_param_training_wrapper(
        Drug, Target, Affinity, Atom, Protein, label_row_inds, label_col_inds, Tags, train_sets, test_sets,
        best_param_list, i)

    return calculate_final_metrics(metrics, Tags)


def execute(Tags, foldcount=6+1):  # 6-fold cross validation + test

    Drug, Target, Affinity, Atom, Protein, label_row_inds, label_col_inds, drugcount, targetcount = initialize_dataset(Tags)
    perf, mseloss, auc, aupr = [], [], [], []
    fold_functions = {
        1: lambda: get_random_folds(len(label_row_inds), foldcount),
        2: lambda: get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount),
        3: lambda: get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount)
    }
    for i in range(1):
        random.seed(1001)
        nfolds = fold_functions.get(Tags.problem_type, lambda: None)()
        avg_perf, avg_loss, test_std, loss_std, avg_auc, auc_std, avg_aupr, aupr_std = perform_cross_validation(Drug, Target, Affinity,
                                                                                                                Atom,
                                                                                                                Protein,
                                                                                                                label_row_inds,
                                                                                                                label_col_inds,
                                                                                                                Tags,
                                                                                                                nfolds, i)
        
        log_results(Tags.problem_type, avg_perf, avg_loss, test_std, loss_std, avg_auc, auc_std, avg_aupr, aupr_std,
                    Tags)
        
        perf.append(avg_perf)
        mseloss.append(avg_loss)
        auc.append(avg_auc)
        aupr.append(avg_aupr)

    log_final_results(perf, mseloss, auc, aupr, Tags)

if __name__ == "__main__":

    Tags = prepare()

    execute(Tags=Tags, foldcount=6+1)
