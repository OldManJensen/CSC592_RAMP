#!/bin/bash
# pretrain on wideresnet
for seed in 0
do
    CUDA_VISIBLE_DEVICES='0' python RAMP_wide_resnet.py --lr-max 0.1  --lr-schedule=step --lr-milestone 50 70 --lr-gamma 0.1 --at_iter 10 --epochs 100 --save_freq 20 --eval_freq 20 --fname RAMP_beta_0.5_lbd_2_wide_trades_$seed --kl --max --final_eval --gp --lbd 2 --seed $seed --wide
done

for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES='0' python RAMP.py --lr-max 0.05  --lr-schedule=step --lr-milestone 100 150 180 --lr-gamma 0.1 --at_iter 10 --epochs 200 --save_freq 20 --eval_freq 20 --fname RAMP_beta_0.5_lbd_5_$seed --kl --max --final_eval --gp --lbd 5 --seed $seed
done
