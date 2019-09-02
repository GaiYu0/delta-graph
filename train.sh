gpu=0
n_gpus=8

bs_train=1000000
d=512
n_iters=1000
p_train=0.8
p_val=0.1

# for i in $(seq 0 9); do
#     for minus_log_lr in $(seq 1 4); do
#         for minus_log_wd in $(seq 1 4); do
#             python3 train.py --bs-train $bs_train --ds ml-10m/partitions/partition-$i --gpu $gpu --model "BiasedMF(n_users, n_items, $d, r_mean)" --n-iters $n_iters --optim "Adam(model.parameters(), 1e-$minus_log_lr, weight_decay=1e-$minus_log_wd)" --p-train $p_train --p-val $p_val --logdir runs/partition-$i-$minus_log_lr-$minus_log_wd &
#             gpu=$((($gpu + 1) % $n_gpus))
#             if [ $gpu -eq 0 ]; then
#                 wait
#             fi
#         done
#     done
# done

for minus_log_lr in $(seq 1 4); do
#   for minus_log_wd in $(seq 1 4); do
    for minus_log_wd in $(seq 5 8); do
        python3 train_on_snapshot.py --bs-train $bs_train --ds ml-10m/snapshots/snapshot-1 --gpu $gpu --logdir runs/cbmf/$minus_log_lr-$minus_log_wd --model "CollapsedBiasedMF(n_users, n_items, $d, r_mean)" --n-iters $n_iters --optim "Adam(model.parameters(), 1e-$minus_log_lr, weight_decay=1e-$minus_log_wd)" --p-train $p_train --p-val $p_val --semi &
        gpu=$((($gpu + 1) % $n_gpus))
        if [ $gpu -eq 0 ]; then
            wait
        fi
    done
done

# for minus_log_lr in $(seq 1 4); do
#     for minus_log_wd in $(seq 1 4); do
#         python3 train_on_snapshot.py --bs-train $bs_train --ds ml-10m/snapshots/snapshot-9 --gpu $gpu --logdir runs/tbmf/$minus_log_lr-$minus_log_wd --model "TemporalBiasedMF(n_users, n_items, $d, list(map(th.mean, rs_train)), 1)" --n-iters $n_iters --optim "Adam(model.parameters(), 1e-$minus_log_lr, weight_decay=1e-$minus_log_wd)" --p-train $p_train --p-val $p_val --semi &
#         gpu=$((($gpu + 1) % $n_gpus))
#         if [ $gpu -eq 0 ]; then
#             wait
#         fi
#     done
# done
