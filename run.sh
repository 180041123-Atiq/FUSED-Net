#!/bin/bash

# dataset=$1


# python train_net.py --num-gpus 1 --mtsdd \
#         --config-file configs/BDTSD/mulAug/simpleIFN_5shot_3.yaml 2>&1 | tee logMulAug/MulAug_5shot_3repeat.txt
# python train_net.py --num-gpus 1 --mtsdd \
#         --config-file configs/BDTSD/mulAug/simpleIFN_10shot_0.yaml 2>&1 | tee logMulAug/MulAug_10shot_0repeat.txt

for repeat_id in 0 1 2 3 4 5 6
do
	for shot in  1 3 5 10
	do
		python train_net.py --num-gpus 1 --mtsdd \
        --config-file configs/BDTSD/simpleIFN_${shot}shot.yaml 2>&1 | tee log/${shot}shot_${repeat_id}repeat.txt
	done
done