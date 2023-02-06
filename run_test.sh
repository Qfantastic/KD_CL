#!/bin/bash

MY_PYTHON="python"

CIFAR_100i="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results_2/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt           --cuda yes --seed 0"

# build datasets
cd data/
cd raw/

$MY_PYTHON raw.py

cd ..



$MY_PYTHON cifar100.py \
	--o cifar100.pt \
	--seed 0 \
	--n_tasks 20

cd ..

# # model "single"

# $MY_PYTHON main.py $CIFAR_100i --model single --lr 1.0
 
# # model "independent"

# $MY_PYTHON main.py $CIFAR_100i --model independent --lr 0.3  --finetune yes 



# # model "EWC"

# $MY_PYTHON main.py $CIFAR_100i --model ewc --lr 1.0  --n_memories 10   --memory_strength 1

# # model "iCARL"
# $MY_PYTHON main.py $CIFAR_100i --model icarl --lr 1.0 --n_memories 1280 --memory_strength 1

# model "GEM"

$MY_PYTHON main.py $CIFAR_100i --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5

# plot results
cd results_2/
$MY_PYTHON plot_results.py
cd ..
