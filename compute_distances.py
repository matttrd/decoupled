import torch
import numpy as np 
import pandas as pd
from argparse import ArgumentParser
import os
import glob2
from IPython import embed
import pyemd
from itertools import combinations, product
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--results', default='results/feat_extr_imagenet/standard/', type=str, help='path of extraction folders')
parser.add_argument('--source', default='imagenet', type=str, help='source dataset')

args = parser.parse_args()


@torch.no_grad()
def kl_divergence(p, q):
    mu0, sigma0 = p
    mu1, sigma1 = q

    if len(mu0.shape) == 1:
        mu0, mu1 = mu0[:, None], mu1[:, None]

    diff = mu1 - mu0
    inv_sigma1 = torch.inverse(sigma1)
    diff_log_det = torch.logdet(sigma1) - torch.logdet(sigma0)
    return 0.5 * (torch.trace(inv_sigma1 @ sigma0) + diff.t() @ inv_sigma1 @ diff + diff_log_det - sigma0.shape[0])


def comput_centroid_distances():
    features_files = glob2.glob(os.path.join(args.results, 'training_*.pt'))
    centroids_dt = {}

    for file in features_files:
        dataset = file.split('/')[-1].split("_")[2]
        type_ = file.split("_")[-1].split('.')[0]
        d = torch.load(file)
        reps = d['reps']

        max_idx = min(reps.shape[0], 10000)
        idx = torch.randperm(reps.shape[0])[0:max_idx]
        reps = reps[idx]

        mu = reps.mean(axis=0)
        reps = reps - mu
        cov = reps.t() @ reps / reps.shape[1]

        centroids_dt[dataset] = (mu, cov)

    ref_cetroid, ref_std = centroids_dt[args.source]
    distances = {}

    for k, (mu, std) in centroids_dt.items():
        if k == args.source:
            continue
        distances[k] = (mu - ref_cetroid).norm().item() #kl_divergence(v, ref_cetroid).item()  

    s_dist = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    torch.save(s_dist, os.path.join(args.results, 'kl_distances.pt'))
    for k, v in s_dist.items():
        print(k, v)


def compute_distances():
    features_files = glob2.glob(os.path.join(args.results, 'training_*.pt'))

    centroids_dt = {}
    for file in features_files:
        dataset = file.split('/')[-1].split("_")[2]
        type_ = file.split("_")[-1].split('.')[0]
        d = torch.load(file)
        classes = torch.unique(d['targs'])
        #centroids = {k : [] for k in classes}
        centroids = []
        weights = []
        for cl in classes:
            centroids.append((d['reps'][d['targs'] == cl]).mean(0, keepdim=True))
            weights.append((d['targs'] == cl).sum().view(1,1))

        centroid_vec = torch.cat(centroids, dim=0)
        weights_vec = torch.cat(weights, dim=0)
        centroids_dt[dataset] = (centroid_vec, weights_vec)

    source, source_weights = centroids_dt[args.source]
    source_weights = source_weights.float() / source_weights.sum()
    distances = {}
    for k,v in centroids_dt.items():
        if k == args.source: 
            continue
        target, target_weights = v[0], v[1].float() / v[1].sum()
        f_s, f_t = source.numpy(), target.numpy()
        data = np.float64(np.append(f_s, f_t, axis=0))
        tmp_source = np.zeros((len(source_weights) + len(target_weights),), np.float64)
        tmp_target = np.zeros((len(source_weights) + len(target_weights),), np.float64)
        tmp_source[:len(source_weights)] = source_weights.squeeze()
        tmp_target[len(source_weights):] = target_weights.squeeze()
        M = np.linalg.norm(data[:,None,...] - data[None,...], axis=2)
        emd = pyemd.emd(np.float64(tmp_source), np.float64(tmp_target), np.float64(M))
        distances[k] = np.exp(-0.01 * emd)

    torch.save(distances, os.path.join(args.results, 'distances.pt'))
    s_dist = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    for k, v in s_dist.items():
        print(k, v)


def compute_self_distances():
    features_files = glob2.glob(os.path.join(args.results, 'training_*.pt'))
    distances = {}
    for file in features_files:
        dataset = file.split('/')[-1].split("_")[2]
        print(f'Computing distance for dataset {dataset}')
        if dataset == 'imagenet':
            continue
        type_ = file.split("_")[-1].split('.')[0]
        d = torch.load(file)
        classes = torch.unique(d['targs'])
        avg_dist, n_comb = 0, 0
        for c1, c2 in combinations(classes.squeeze(), 2):
            rep_1 = d['reps'][d['targs'] == c1].numpy()
            rep_2 = d['reps'][d['targs'] == c2].numpy()
            dist = features_dist(rep_1, rep_2)
            avg_dist += dist
            n_comb += 1
        distances[dataset] = avg_dist / n_comb
    torch.save(distances, os.path.join(args.results, 'self_distances.pt'))


def features_dist(feat_1, feat_2):
    dist = 0
    n = 0
    feat_1 = feat_1[0:min(feat_1.shape[0], 1000)]
    feat_2 = feat_1[0:min(feat_2.shape[0], 1000)]
    for f1, f2 in tqdm(product(feat_1, feat_2), total=feat_1.shape[0] * feat_2.shape[0]):
        dist += np.linalg.norm(f1 - f2)
        n += 1
    return dist / n


if __name__ == "__main__":
    print("Computing centroid distances")
    comput_centroid_distances()
    print("Computing emd distances")
    compute_distances()
    #compute_self_distances()
