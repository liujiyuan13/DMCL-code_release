from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from models import *
from utils import *

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_dtype(torch.float32)

ALG_NAME = 'DMCL'
LOG_DIR = './log'
SMALL_DATA = ['Mfeat_fea', 'ORL_fea', '100Leaves', 'MSRCV1', 'Caltech-5V', 'Flower17_fea', 'scene15_mtv', 'DryBean_fea']
BIG_DATA = []


def prepare_args(args, is_best=False):
    # param
    if args.data_name == 'Mfeat_fea':
        args.n_views, args.n_clusters, args.n_samples = 6, 10, 2000
        if is_best:
            args.dim_latents = 60
        args.dims_mapping = [
            (args.dim_latents, 64, 216),
            (args.dim_latents, 64, 76),
            (args.dim_latents, 32, 64),
            (args.dim_latents, 16, 6),
            (args.dim_latents, 64, 240),
            (args.dim_latents, 32, 47)
        ]
    else:
        raise NotImplementedError

    if args.data_name in SMALL_DATA:
        args.batch_size = args.n_samples
        print('@ batch size is set to n_samples for small datasets ...')

    if args.batch_size == -1:
        args.batch_size = args.n_samples

    return args


def train(args):
    # load data
    mv_data = MultiviewData(args.data_name, args.device)
    args.n_views, args.n_clusters, args.n_samples = mv_data.n_views, mv_data.n_clusters, mv_data.n_samples  # double confirm
    data_loader = torch.utils.data.DataLoader(mv_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # log
    args.log_path = '{}/{}/iter_{}/{}/D_{}_alpha_{}_beta_{}_bz_{}'.format(
        LOG_DIR, ALG_NAME, args.iter, args.data_name, args.dim_latents, args.alpha, args.beta, args.batch_size)
    print('#', args.log_path)
    writer = SummaryWriter(args.log_path)

    # init model
    model = DMCL(args.dim_latents, args.dims_mapping, args.n_samples, device=args.device)

    # def optimizer
    param_dict = []
    for mapping in model.mappings:
        param_dict.append({'params': mapping.parameters(), 'lr': args.lr})
    param_dict.append({'params': model.latents, 'lr': args.lr})
    param_dict.append({'params': model.target_diag, 'lr': args.lr})
    optimizer = torch.optim.Adam(param_dict)

    if args.save_rep:
        latents_save = np.zeros([int(args.epochs / 10) + 1, args.n_samples, args.dim_latents])
        accs, nmis, purs, aris, f_scores, v_measures = [], [], [], [], [], []
        losses, losses_rec, losses_sep, losses_KA = [], [], [], []

    # train
    for epoch in range(args.epochs + 1):
        loss_item, loss_rec_item, loss_sep_item, loss_KA_item = 0, 0, 0, 0
        for data in data_loader:
            xs, _, idx = data
            idx = idx.cpu().numpy()

            xrs, latents = model.fit(idx)

            # loss
            loss_rec = loss_fun_rec(xs, xrs)
            loss_sep = loss_fun_orth(latents, None)
            loss_KA = loss_fun_adaptive_kernel_alignment(xs, latents)

            loss = loss_rec + args.alpha * loss_sep + args.beta * loss_KA

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_rec_item += loss_rec.item()
            loss_sep_item += loss_sep.item()
            loss_KA_item += loss_KA.item()
            loss_item += loss.item()

        if epoch % 10 == 0:
            latents_np = model.latents.clone().detach().cpu().numpy()
            kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(latents_np)
            acc, nmi, pur, ari, f_score, v_measure = evaluate(data_loader.dataset.gt, kmeans.labels_)

            print('-%02d, loss=%.4f, loss_rec=%.4f, loss_sep=%.4f, loss_KA=%.4f, acc=%.4f' % (
                epoch, loss_item, loss_rec_item, loss_sep_item, loss_KA_item, acc))

            writer.add_scalar('loss', loss_item, epoch)
            writer.add_scalar('loss_rec', loss_rec_item, epoch)
            writer.add_scalar('loss_sep', loss_sep_item, epoch)
            writer.add_scalar('loss_KA', loss_KA_item, epoch)
            writer.add_scalar('acc', acc, epoch)
            writer.add_scalar('nmi', nmi, epoch)
            writer.add_scalar('pur', pur, epoch)
            writer.add_scalar('ari', ari, epoch)
            writer.add_scalar('f_score', f_score, epoch)
            writer.add_scalar('v_measure', v_measure, epoch)

            if args.save_rep:
                latents_save[int(epoch / 10), :, :] = latents_np
                accs.append(acc)
                nmis.append(nmi)
                purs.append(pur)
                aris.append(ari)
                f_scores.append(f_score)
                v_measures.append(v_measure)
                losses.append(loss_item)
                losses_rec.append(loss_rec_item)
                losses_sep.append(loss_sep_item)
                losses_KA.append(loss_KA_item)

    writer.close()

    if args.save_rep:
        scio.savemat(args.log_path + '/res.mat',
                     {'data_name': args.data_name, 'latents': latents_save, 'gt': data_loader.dataset.gt, 'accs': accs,
                      'nmis': nmis, 'purs': purs, 'aris': aris, 'f_scores': f_scores, 'v_measures': v_measures,
                      'losses': losses, 'losses_rec': losses_rec, 'losses_sep': losses_sep, 'losses_KA': losses_KA})


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='DMCL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_name', type=str, default='Mfeat_fea',
                        choices=['Mfeat_fea', 'ORL_fea', '100Leaves', 'MSRCV1', 'Caltech-5V', 'Flower17_fea', 'scene15_mtv', 'DryBean_fea'])
    parser.add_argument('--dim_latents', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=-1, choices=[-1, 128, 256, 512, 1024]) # -1 is num_samples
    parser.add_argument('--alpha', type=float, default=1.0, choices=[0.01, 0.1, 1.0, 10, 100])
    parser.add_argument('--beta', type=float, default=1.0, choices=[0.01, 0.1, 1.0, 10, 100])
    parser.add_argument('--epochs', type=float, default=2000)
    parser.add_argument('--iter', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 10])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_rep', type=bool, default=False, choices=[True, False])
    args = parser.parse_args()

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # run
    setup_seed(args.seed)
    args = prepare_args(args, is_best=False)
    train(args)
