import os, sys, subprocess, json, argparse, time
from itertools import product
import torch as th
from multiprocessing.pool import ThreadPool

parser = argparse.ArgumentParser(description='Jobber',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i','--input',   help='processes to run', type=str, required=True)
parser.add_argument('--gpus',         help='array of gpus to use', type=str, default='')
parser.add_argument('-r', '--run',    help='run',  action='store_true')
opt = vars(parser.parse_args())

if opt['gpus'] == '':
    gs = range(th.cuda.device_count())
else:
    gs = json.loads(opt['gpus'])

with open(opt['input'], 'r') as f:
    cmds = f.readlines()
cmds = [x.strip() for x in cmds if x[0] != '#']

running = [None for g in gs]
current_g = [None for g in gs]

def runner():
    def helper(idx), g=None:
        if not len(cmds):
            return
        c = cmds.pop()
        if g is None:
            c = c + (' -g %d')%(gs[idx%len(gs)])
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
                        helper(idx, g=current_g[idx])
                else:
                    helper(idx)
                    current_g[idx] = gs[idx%len(gs)]

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

if not len(cmds):
    print('No commands found')
else:
    if not opt['run']:
        for c in cmds:
            print(c)
    else:
        runner()