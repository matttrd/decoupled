import torch
import numpy as np 
import pandas as pd
from argparse import ArgumentParser
import os
import glob2
from IPython import embed
import pyemd

parser = ArgumentParser()
parser.add_argument('--results', default='results/feat_extr_imagenet/standard/', type=str, help='path of extraction folders')
parser.add_argument('--source', default='imagenet', type=str, help='source dataset')

args = parser.parse_args()

def main():
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
        if k == 'imagenet': 
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


if __name__ == "__main__":
    main()
