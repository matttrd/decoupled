import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from argparse import ArgumentParser
import numpy as np
from glob2 import glob
# from exp_library.model_utils import make_and_restore_model
import pandas as pd
import os
from exp_library.datasets import DATASETS
# from exp_library.model_utils import make_and_restore_model, \
#                                 check_experiment_status, \
#                                 model_dataset_from_store
from cox import readers
from matplotlib import rc
#rc('text', usetex=True)


parser = ArgumentParser()
parser.add_argument('--data', default='', type=str, help='path to dataset')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--results', default='', type=str, help='path of model folders')
parser.add_argument('--tsne', type=bool, default=False, help='plot tsne')
parser.add_argument('--snn', type=bool, default=False, help='soft nearest neighbor')
parser.add_argument('--temp', type=float, default=100., help='temperature')
parser.add_argument('--norm', type=bool, default=False, help='norm of weights')
parser.add_argument('--few-shot', type=bool, default=False, help='results on few shot')
parser.add_argument('--compare-dyn', type=bool, default=False, help='results on comparison rob st')


args = parser.parse_args()

### SOFT NEAREST NEIGHBOR
def pairwise_distance(x):
    xx = x.unsqueeze(0)
    yy = x.unsqueeze(1)
    # (B, B, p) -> (B,B)
    return torch.norm(yy - xx, dim=2) ** 2

def soft_exp_minus_1(dist_matrix, temp):
    return torch.exp(-(dist_matrix / temp))

def same_label_mask(y, y2):
    return y.eq(y2.unsqueeze(1)).squeeze()

def snn(x, target, temp=10.):
    EPS = 0.00001
    dist_matrix = pairwise_distance(x)
    f =  soft_exp_minus_1(dist_matrix, temp) - torch.eye(dist_matrix.shape[0]).cuda()
    norm = f.sum(dim=1)
    f = (f * same_label_mask(target, target)).sum(dim=1) / norm
    return (-torch.log(EPS + f)).mean()


plt.clf()
### T-SNE
def plot_tsne(x, y, npca=100, markersize=10):
    Xlow = PCA(n_components=npca).fit_transform(x)
    Y = manifold.TSNE(n_components=2,init='pca').fit_transform(Xlow)
    palette = sns.color_palette("Paired", len(np.unique(y)))
    color_dict = {l: c for l, c in zip(range(len(np.unique(y))), palette)}
    colors = [color_dict[l] for l in y]
    fig, ax = plt.subplots()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], markersize, c=colors, marker='o')
    return fig


def get_type(name):
    if 'random' in name:
        return 'random'
    elif 'linear' in name:
        return 'linear'
    elif 'standard' in name:
        return 'standard'


def load(fol, shot=False):
    dfs = readers.CollectionReader(fol)
    logs = dfs.df('logs')
    tmp = logs['exp_id'].apply(lambda x: x.split('_'))
    tr_dt, mode = tmp.apply(lambda x: x[0]), tmp.apply(lambda x: x[1])
    logs['dataset'] = tr_dt
    logs['mode'] = mode
    logs['mode'] = logs['mode'].apply(lambda x: int(x))
    if shot:
        sh = tmp.apply(lambda x: x[2])
        logs['shot'] = sh
    return logs


def concat(dfs, types):
    for i, df in enumerate(dfs):
        df['type'] = types[i]
    return pd.concat(dfs)


def filter(x):
    if int(x['mode'].unique()) == 0:
        return x.iloc[0:-1]
    else:
        return x


def main():
    # data_path = os.path.expandvars(args.data)
    # dataset = DATASETS[args.dataset](data_path)
    # train_loader, val_loader, _ = dataset.make_loaders(workers=8, 
    #                                       batch_size=args.batch_size)


    # experiments = glob(args.results + '/**/checkpoint.pt.latest')

    # df = pd.DataFrame(columns=['eps', 'snn', 'type'])
    # d = {'eps':[], 'snn':[], 'type':[]}

    if args.norm:
        dfs = readers.CollectionReader(args.results)
        norms = dfs.df('custom')
        fig, ax = plt.subplots()
        norms['wn'] = norms['weight_norm']
        for name, group in norms.groupby('exp_id'):
            group.plot(x='epoch', y='wn', label=name, ax=ax, legend=False)
        
        handles, labels = ax.get_legend_handles_labels()
        labels = list(map(lambda x: x.split('_')[1], labels))
        labels = list(map(lambda x: r"$\varepsilon =" + x + "$", labels))
        plt.ylabel(r"$\|w\|_2$")
        ax.legend(handles, labels)
        plt.savefig(os.path.join(args.results, 'norms.pdf'))
        return

    if args.tsne or args.snn:
        from exp_library.model_utils import model_dataset_from_store
        for exp in experiments:
            target_folder = exp.rsplit('/',1)[0]
            tmp = target_folder.rsplit('/',1)
            out_dir, exp_name = tmp[0], tmp[1]
            model, checkpoint, _, store, args_model = model_dataset_from_store((out_dir, exp_name), 
                            overwrite_params={}, which='last', mode='r', parallel=False)

            iterator = enumerate(val_loader)
            _, (im, targ) = next(iterator)
            im = im.cuda() / 255.
            targ = targ.cuda()
            (op, rep), _ = model(im, target=targ, with_latent=True)
            
            rep = rep.detach()
            if args.tsne:
                fig = plot_tsne(rep.cpu().numpy(), targ.cpu().numpy(), npca=100, markersize=10)
                fn = '_'.join(['tsne', 'bsz', str(args.batch_size), 'temp', str(args.temp)])
                plt.savefig(os.path.join(target_folder, fn) +'.pdf')
                plt.clf()

            if args.snn:
                # 3 tables for random-eps, linear-eps, standard
                score = snn(rep, targ, temp=args.temp)
                d['eps'].append(args_model.eps)
                d['snn'].append(score.item())
                d['type'].append(get_type(out_dir))

    if args.snn:
        df = pd.DataFrame(d)
        df.sort_values(by='eps').groupby('type')
        df.to_json(os.path.join(target_folder, 'snn.json'))
        df = df.set_index(['type', 'eps']).sort_index()
        macro = df.to_latex(multirow=True)
        with open(args.results + 'latex.tex', 'w') as h:
            h.write(macro)

    if args.compare_dyn:
        subfolders = [f'trasf-{args.dataset}-st-new', f'trasf-{args.dataset}-rob-new']
        subfolders = [os.path.join(args.results, subfolder) for subfolder in subfolders]
        logs_st = load(subfolders[0])
        logs_st = logs_st.groupby('exp_id').apply(filter).reset_index(drop=True)
        logs_rob = load(subfolders[1])
        logs = concat((logs_st, logs_rob), ['st', 'rob'])

        plt.figure()
        sns.set_style('darkgrid')
        grid = sns.FacetGrid(col='mode', data=logs)
        grid.map_dataframe(sns.lineplot, "epoch", 'nat_prec1', hue='dataset',
                           style='type', style_order=['rob', 'st'])
        lgd = grid.axes[0, 2].legend(handles=grid._legend_data.values(), labels=grid._legend_data.keys(),
                               loc='center', bbox_to_anchor=(1.25, 0.5), facecolor='white',
                               edgecolor='white')
        plt.show()
        grid.savefig(f'results/dynamic_comparison_{args.dataset}.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.figure()
        tmp = logs.groupby(['exp_id', 'type']).max().reset_index()
        sns.lineplot(data=tmp, y='nat_prec1', x='mode', hue='dataset', style='type', markers=True)
        lgd = plt.legend(loc='center', bbox_to_anchor=(1.15, 0.5), facecolor='white', edgecolor='white')
        plt.savefig(f"results/dynamic_comparison_mode_{args.dataset}.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        tmp = logs.groupby(['dataset', 'mode', 'type']).max().reset_index()
        pivot = pd.pivot_table(tmp, index=['mode', 'type'], columns=['dataset'], values='nat_prec1')
        pivot.to_latex(f"results/dynamic_comparison_mode_{args.dataset}.tex", float_format="%.2f")

    # if args.few_shot:
    #     #dfs = readers.CollectionReader(args.results)
    #     #subfolders = [f'trasf-{args.dataset}-st-shot', f'trasf-{args.dataset}-rob-shot']
    #     subfolders = [f'trasf-{args.dataset}-st-new', f'trasf-{args.dataset}-rob-new']
    #     fol = os.path.join(args.results, subfolders[0])
    #     dfs = readers.CollectionReader(fol)
    #     logs = dfs.df('logs')
    #     #meta = dfs.df('metadata')

    #     nat_prec1 = logs.groupby('exp_id').apply(lambda x: x['nat_prec1'].max())
    #     tmp = pd.DataFrame(nat_prec1.index)['exp_id'].apply(lambda x: x.split('_'))
    #     tr_dt, mode, shot = tmp.apply(lambda x: x[0]), tmp.apply(lambda x: x[1]), tmp.apply(lambda x: x[2])
    #     df = pd.DataFrame()
    #     df['nat_prec1'] = nat_prec1.values
    #     df['dataset'] = tr_dt
    #     df['mode'] = mode
    #     #df['shot'] = shot
    #     #df['shot'] = df['shot'].apply(lambda x: int(x))
    #     #df = df.sort_values('shot')

    #     from IPython import embed
    #     embed()

    #     fig, ax = plt.subplots()
    #     for dataset_name, group in df.groupby('dataset'):
    #         for mode_name in group.groupby('mode'):
    #             group.plot(x='epoch', y='nat_prec1', label=name, ax=ax, legend=False)
    #             plt.savefig(os.path.join(args.results, f'trasf_accuracy_{args.dataset}_{dataset_name}.pdf'))
                #plt.clf()

        # for name, group in logs.groupby('exp_id'):
        #     group.plot(x='epoch', y='nat_prec1', label=name, ax=ax, legend=False)
        #     handles, labels = ax.get_legend_handles_labels()
        #     labels = list(map(lambda x: x.split('_')[1], labels))
        #     labels = list(map(lambda x: r"$\varepsilon =" + x + "$", labels))

    
    # bashCommand = "rsync -av -R results/cifar/**/*.pdf ~/Dropbox/tnse --include 'tsne'"
    # import subprocess
    # process = subprocess.run(bashCommand, shell=True)
    # output, error = process.communicate()

if __name__ == "__main__":
    main()


