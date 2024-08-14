from arguments import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader

from metrics import *
from sklearn.metrics import roc_auc_score

from model import net
## perform_cross_validation
from run import optimize_hyperparameters, optimal_param_training, train, evaluate, device

def create_train_val_test_sets(outer_train_sets, test_set):
    val_sets, train_sets, test_sets = [], [], []
    foldinds = len(outer_train_sets)
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = [fold for idx, fold in enumerate(outer_train_sets) if idx != val_foldind]
        train_sets.append([item for sublist in otherfolds for item in sublist])
        test_sets.append(test_set)
    return val_sets, train_sets, test_sets


def optimize_hyperparameters_wrapper(Drug, Target, Affinity, label_row_inds, label_col_inds, Tags, train_sets, test_sets):
    bestparamind, best_param_list, bestperf, all_predictions, all_losses = optimize_hyperparameters(
        Drug, Target, Affinity, label_row_inds, label_col_inds, Tags, train_sets, test_sets)
    return best_param_list


def optimal_param_training_wrapper(Drug, Target, Affinity, Atom, Protein, label_row_inds, label_col_inds, Tags, train_sets, test_sets, best_param_list, i):
    best_param, bestperf, all_predictions, all_losses, all_auc, all_aupr = optimal_param_training(
        Drug, Target, Affinity, Atom, Protein, label_row_inds, label_col_inds, Tags, train_sets, test_sets, best_param_list, i)
    logging("-----FINAL RESULTS-----", Tags)
    logging(f"best param = {best_param_list}", Tags)
    return all_predictions, all_losses, all_auc, all_aupr


def calculate_final_metrics(metrics, Tags):
    all_predictions, all_losses, all_auc, all_aupr = metrics
    testperfs = [foldperf for foldperf in all_predictions]
    testloss = [foldloss for foldloss in all_losses]
    testauc = [auc for auc in all_auc]
    testaupr = [aupr for aupr in all_aupr]

    avg_perf = np.mean(testperfs)
    avg_loss = np.mean(testloss)
    perf_std = np.std(testperfs)
    loss_std = np.std(testloss)
    avg_auc = np.mean(testauc)
    auc_std = np.std(testauc)
    avg_aupr = np.mean(testaupr)
    aupr_std = np.std(testaupr)

    logging("Test Performance CI:", Tags)
    logging(testperfs, Tags)
    logging("Test Performance MSE:", Tags)
    logging(testloss, Tags)

    return avg_perf, avg_loss, perf_std, loss_std, avg_auc, auc_std, avg_aupr, aupr_std
##----------------------------------------------------------------------------------------------------------------------

## execute
def log_results(problem_type, avg_perf, avg_loss, test_std, loss_std, avg_auc, auc_std, avg_aupr, aupr_std, Tags):
    logging("problem_type: " + str(problem_type), Tags)
    logging(
        "avg_perf = %.5f, avg_mse = %.5f, std = %.5f, loss_std = %.5f, auc = %.5f, auc_std = %.5f, aupr = %.5f, aupr_std = %.5f" %
        (avg_perf, avg_loss, test_std, loss_std, avg_auc, auc_std, avg_aupr, aupr_std), Tags)


def log_final_results(perf, mseloss, auc, aupr, Tags):
    logging(
        "avg_perf = %.5f, avg_mse = %.5f, std = %.5f, loss_std = %.5f, auc = %.5f, auc_std = %.5f, aupr = %.5f, aupr_std = %.5f" %
        (np.mean(perf), np.mean(mseloss), np.std(perf), np.std(mseloss), np.mean(auc), np.std(auc), np.mean(aupr), np.std(aupr)),
        Tags)
    logging("------------------All over---------------------", Tags)
##----------------------------------------------------------------------------------------------------------------------


## optimal_param_training
def prepare_loaders(Drug, Target, Affinity, Atom, Protein, label_row_indexs, label_col_indexs, labeledindexs, valindexs,
                    batchsz):
    trrows, trcols = label_row_indexs[labeledindexs], label_col_indexs[labeledindexs]
    terows, tecols = label_row_indexs[valindexs], label_col_indexs[valindexs]

    train_dataset = prepare_datasets(Drug, Target, Affinity, Atom, Protein, trrows, trcols)
    test_dataset = prepare_datasets(Drug, Target, Affinity, Atom, Protein, terows, tecols)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsz, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsz)

    return train_loader, test_loader

def prepare_datasets(Drug, Target, Affinity, Atom, Protein, rows, cols):
    dataset = [[]]
    for pair_ind in range(len(rows)):
        drug = Drug[rows[pair_ind]]
        atom = Atom[rows[pair_ind]]
        dataset[pair_ind].append(np.array(drug, dtype=np.float32))
        target = Target[cols[pair_ind]]
        pro = Protein[cols[pair_ind]]

        dataset[pair_ind].append(np.array(target, dtype=np.float32))
        dataset[pair_ind].append(np.array(Affinity[rows[pair_ind], cols[pair_ind]], dtype=np.float32))
        dataset[pair_ind].append(atom)
        dataset[pair_ind].append(pro)
        if pair_ind < len(rows) - 1:
            dataset.append([])

    return dataset

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        if isinstance(m, nn.LSTM):
            init.orthogonal_(m.all_weights[0][0])
            init.orthogonal_(m.all_weights[0][1])
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)

def initialize_model(Tags, param1value, param2value, param3value):
    torch.cuda.empty_cache()
    model = net(Tags, param1value, param2value, param3value, atom_feature_size=1 * 7, res_feature_size=1 * 7)
    model.apply(weights_init)
    model = nn.DataParallel(model)
    return model.to(device)


def train_and_evaluate(model, train_loader, test_loader, Tags, param1value, param2value, param3value, lamda):
    cindex_list = []
    for epochind in range(Tags.num_epoch):
        model = train(train_loader, model, Tags, param1value, param2value, param3value, lamda)
        if (epochind + 1) % 2 == 0:
            cindex = evaluate_and_log(model, test_loader, Tags, param1value, param2value, param3value, lamda,
                                      cindex_list)
            cindex_list.append(cindex)
    return cindex_list


def evaluate_and_log(model, test_loader, Tags, param1value, param2value, param3value, lamda, cindex_list):
    cindex, loss, r2, rm2, auc, loss_d, loss_t = evaluate(model, test_loader, Tags, param1value, param2value,
                                                          param3value, lamda)
    logging(
        f"test: epoch:{len(cindex_list)}, p1:{param1value}, p2:{param2value}, p3:{param3value}, loss:{loss:.5f}, cindex:{cindex:.5f}, r2:{r2:.5f}, rm2:{rm2:.5f}",
        Tags)

    if cindex >= max(cindex_list, default=0):
        torch.save(model, 'checkpoint0.pth')
    return cindex


def test_model(model, test_loader, Tags, param1value, param2value, param3value):
    model = torch.load('checkpoint0.pth')
    model.eval()

    pre_affinities, affinities = [], []
    for data in test_loader:
        pre_affinity = model(*data[:4], Tags, param1value, param2value, param3value)[0]
        pre_affinities.extend(pre_affinity.cpu().detach().numpy())
        affinities.extend(data[2].cpu().detach().numpy())

    return np.array(affinities), np.array(pre_affinities)


def calculate_and_log_metrics(affinities, pre_affinities, Tags, param1value, param2value, param3value, foldindex):
    dataset_path = Tags.dataset_path
    if 'davis' in dataset_path:
        auc, aupr = calculate_roc_auc_aupr(affinities, pre_affinities, threshold=7.0)
    elif 'kiba' in dataset_path:
        auc, aupr = calculate_roc_auc_aupr(affinities, pre_affinities, threshold=12.1)

    mse, mae, rm2, r2, cindex = calculate_metrics(pre_affinities, affinities)
    loss_func = nn.MSELoss()
    loss = loss_func(torch.Tensor(pre_affinities), torch.Tensor(affinities))

    logging(
        f"best: p1:{param1value}, p2:{param2value}, p3:{param3value}, loss:{loss:.5f}, cindex:{cindex:.5f}, rm2:{rm2:.5f}, r2:{r2:.5f}",
        Tags)
    logging(
        f"best: P1 = {param1value}, P2 = {param2value}, P3 = {param3value}, Fold = {foldindex}, CI-i = {cindex}, MSE = {loss}, auc = {auc}, aupr = {aupr}, rm2 = {rm2}, r2 = {r2}",
        Tags)

    return mse, mae, rm2, r2, cindex, auc, aupr, loss


def calculate_roc_auc_aupr(affinities, pre_affinities, threshold):
    labels = np.int32(affinities > threshold)
    auc = roc_auc_score(labels, pre_affinities)
    aupr = get_aupr(labels, pre_affinities)
    return auc, aupr


def save_affinities(all_affinities, all_preaffinities, i):
    np.savetxt(f"./result/pre/iter{i}affinities.txt", np.array(all_affinities))
    np.savetxt(f"./result/pre/iter{i}preaffinities.txt", np.array(all_preaffinities))

##----------------------------------------------------------------------------------------------------------------------






