import os
import random
import torch
import math
import mat73
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score, fowlkes_mallows_score, normalized_mutual_info_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_mask(data_size, view_num, missing_ratio):
    """
    :param data_size: size of data
    :param view_num: number of views
    :param missing_ratio: missing ratio
    :return: mask matrix
    """
    assert view_num >= 2
    miss_sample_num = math.floor(data_size*missing_ratio)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, view_num])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(view_num)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if np.sum(rand_v) > 0 and np.sum(rand_v) < view_num:
                break
        mask[miss_ind[j]] = rand_v

    return mask


class MultiviewData(Dataset):
    def __init__(self, data_name, device='cpu', path="datasets/"):
        data_path = os.path.join(path, data_name + '.mat')
        try:
            contents = mat73.loadmat(data_path)
        except:
            contents = scio.loadmat(data_path)

        if data_name in ['ORL_fea', 'Mfeat_fea', 'DryBean_fea', 'Flower17_fea']:
            X, gt = contents['X'], contents['Y'].astype(dtype=np.int32).squeeze()
            self.X = [x[0].astype(dtype=np.float32).T for x in X]
            self.gt = gt - gt.min()
        elif data_name in ['100Leaves']:
            X, gt = contents['X'], contents['y'].astype(dtype=np.int32).squeeze()
            self.X = [x[0].astype(dtype=np.float32) for x in X]
            self.gt = gt - gt.min()
        elif data_name in ['MSRCV1', 'Caltech-5V']:
            X, gt = contents['X'], contents['Y'].astype(dtype=np.int32).squeeze()
            self.X = [x[0].astype(dtype=np.float32) for x in X]
            self.gt = gt - gt.min()
        elif data_name in ['scene15_mtv']:
            X, gt = contents['X'], contents['gt'].astype(dtype=np.int32).squeeze()
            self.X = [x.astype(dtype=np.float32).T for x in X.squeeze()]
            self.gt = gt - gt.min()
        else:
            raise NotImplementedError

        self.n_views = len(self.X)
        self.n_clusters = self.gt.max() + 1
        self.n_samples = len(self.gt)

        scaler = MinMaxScaler()
        for idx in range(self.n_views):
            self.X[idx] = scaler.fit_transform(self.X[idx])
            self.X[idx] = torch.from_numpy(self.X[idx]).to(device)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        X = [self.X[iv][idx, :] for iv in range(self.n_views)]
        gt = self.gt[idx]
        return X, gt, idx


def accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int32)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int32)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(y_true, y_pred):
    acc = accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    pur = purity(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    f_score = fowlkes_mallows_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    return acc, nmi, pur, ari, f_score, v_measure
