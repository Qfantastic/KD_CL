# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path
import torch
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/cifar100.pt', help='input directory')
parser.add_argument('--o', default='cifar100.pt', help='output file')
parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))  # x_tr : train image 50000 * 3072(32*32*3)   x_te : test image 10000 * 3072
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0              # y_tr : train image label 50000   y_te : test image label 10000
x_te = x_te.float().view(x_te.size(0), -1) / 255.0

print("shape:",x_tr.size())
print("shape:",y_tr.size())
print("shape:",x_te.size())
print("shape:",y_te.size())
cpt = int(100 / args.n_tasks)     #capacity of every task

for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])     #tasks 1.(range of every task), 2.train data in this task, 3. train data label in this task
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])   # test data part

print("tasts_tr",np.shape(tasks_tr))
n_inputs = tasks_tr[0][1]
print(n_inputs)

n_tasks = len(tasks_tr)    # number of task
task_permutation = range(n_tasks)    
task_permutation = torch.randperm(n_tasks).tolist()   # random the tasks' order
print('task_permutation:',task_permutation)
p = torch.randperm(50)[0:10]
print(p)
N = tasks_tr[0][1].size(0)
print("N:",N)


for i, task in enumerate(tasks_tr):
    t = i
    x = task[1]
    y = task[2]
    eval_bs = x.size(0)
    print("task:",t)
    print("eval_bs:",eval_bs)


torch.save([tasks_tr, tasks_te], args.o)
