gpu=0
n_gpus=8
for i in $(seq 0 9); do
    for minus_log_lr in $(seq 1 4); do
        for minus_log_wd in $(seq 1 8); do
            ipython3 train.py -- --bs-infer 100000 --bs-train 1000000 --ds ml-10m/partitions/partition-$i --gpu $gpu --model "BiasedMF(n_users, n_items, 512, r_mean)" --n-iters 1 --optim "Adam(model.parameters(), 1e-$minus_log_lr, weight_decay=1e-$minus_log_wd)" --p-train 0.8 --p-val 0.1 --logdir runs/partition-$i-$minus_log_lr-$minus_log_wd &
            gpu=$((($gpu + 1) % $n_gpus))
            if [ $gpu -eq 0 ]; then
                wait
            fi
        done
    done
done
