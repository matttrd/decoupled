from itertools import product
import os, sys, subprocess, json, argparse, time
from itertools import product
import torch as th
from multiprocessing.pool import ThreadPool

parser = argparse.ArgumentParser(description='Quick dirty hyperoptim',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--command',   help='Main command', type=str, required=True)
parser.add_argument('-p','--params',    help='JSON dict of the hyper-parameters', type=str)
parser.add_argument('--gpus',           help='array of gpus to use', type=str, default='')
parser.add_argument('-r', '--run',      help='run',  action='store_true')
parser.add_argument('-j', '--max_jobs',     help='max jobs',    type=int, default = 1)
parser.add_argument('--dist',           help='using dist sgd',    action='store_true')
parser.add_argument('--exp_keys', help='keys to form the exp name', type=str)
opt = vars(parser.parse_args())


if opt['gpus'] == '':
    gs = range(th.cuda.device_count())
else:
    gs = json.loads(opt['gpus'])

running = [None for g in gs]

def runner():
    def helper(idx):
        if not len(cmds):
            return
        c = cmds.pop()
        p = subprocess.Popen(c, shell=True)
        running[idx] = p
        print('[Launch] ', p, c)

    while True:
        try:
            for idx, p in enumerate(running):
                if p is not None:
                    if p.poll() is None:
                        continue
                    else:
                        print('[Cleanup] ', p, p.returncode)
                        running[idx] = None
                        helper(idx)
                else:
                    helper(idx)

            num_running = sum([1 if r is not None else 0 for r in running])
            if len(cmds) == 0 and  num_running == 0:
                print('Finished')
                sys.exit()

            time.sleep(0.5)

        except KeyboardInterrupt:
            print('[Killing everything]')
            for p in running:
                p.kill()
            sys.exit()

exp_keys = eval(opt['exp_keys'])

cmd = opt['command']
params = json.loads(opt['params'])

cmds = []
if opt['gpus'] == '':
    gs = range(th.cuda.device_count())
else:
    gs = json.loads(opt['gpus'])
keys,values = zip(*params.items())
count = 0
for v in product(*values):
    p = dict(zip(keys,v))
    s = ''
    for k in p:
        if len(k) > 1:
            s += ' --'+k+' '+str(p[k])
        else:
            s += ' -'+k+' '+str(p[k])

    c = cmd+s#+' -l'
    if not opt['dist']:
        # c = c + (' -g %d')%(gs[len(cmds)%len(gs)])
        c = 'CUDA_VISIBLE_DEVICES=%d '%(gs[len(cmds)%len(gs)]) + c
    exp_name = []
    for n in exp_keys:
        exp_name.append(str(p[n]))
    exp_name = '_'.join(exp_name)
    c += f' --exp {exp_name}'
    cmds.append(c)
    count += 1

if not opt['run']:
    for c in cmds:
        print(c)
else:
    runner()