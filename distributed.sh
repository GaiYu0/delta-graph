hosts=(3.94.29.70 34.236.254.237 3.222.200.79 34.205.87.121)
gpu_ptr=($(seq 8 8 $((8 * ${#hosts[*]}))))
gpus=($(for i in $(seq 1 ${#hosts[*]}); do echo $(seq 0 7); done))

bs_train=100000
d=256
n_iters=1000
p_train=0.8
p_val=0.1

rm -fr runs/* logs/*
for host in ${hosts[*]}; do
    ssh -n ubuntu@$host "rm -fr delta-graph/runs/*"
done

# host_idx=0
# gpu_idx=0
# for i in $(seq 0 9); do
#     for minus_log_lr in $(seq 1 4); do
#         for minus_log_wd in $(seq 1 8); do
#             echo ${hosts[$host_idx]} ${gpus[$gpu_idx]}
#             ssh -n ubuntu@${hosts[$host_idx]} "source activate pytorch_p36; cd delta-graph; python3 train.py --bs-train $bs_train --ds MovieLens/ml-10M100K/partitions/$i --gpu ${gpus[$gpu_idx]} --model \"BiasedMF(n_users, n_items, $d, r_mean)\" --n-iters $n_iters --optim \"Adam(model.parameters(), 1e-$minus_log_lr, weight_decay=1e-$minus_log_wd)\" --p-train $p_train --p-val $p_val --logdir runs/$i-$minus_log_lr-$minus_log_wd" | tee logs/$i-$minus_log_lr-$minus_log_wd &
#             gpu_idx=$(($gpu_idx + 1))
#             if [ $gpu_idx -eq ${gpu_ptr[$host_idx]} ]; then
#                 host_idx=$(($host_idx + 1))
#             fi
#             if [ $host_idx -eq ${#hosts[*]} ]; then
#                 wait
#                 host_idx=0
#                 gpu_idx=0
#             fi
#         done
#     done
# done

host_idx=0
gpu_idx=0
for minus_log_lr in $(seq 1 1); do
    for minus_log_wd in $(seq 1 1); do
        echo ${hosts[$host_idx]} ${gpus[$gpu_idx]}
        ssh -n ubuntu@${hosts[$host_idx]} "source activate pytorch_p36; cd delta-graph; python3 train_on_snapshot.py --bs-train $bs_train --ds MovieLens/ml-10M100K/snapshot --gpu ${gpus[$gpu_idx]} --model \"CollapsedBiasedMF(n_users, n_items, $d, r_mean)\" --n-iters $n_iters --optim \"Adam(model.parameters(), 1e-$minus_log_lr, weight_decay=1e-$minus_log_wd)\" --p-train $p_train --p-val $p_val --logdir runs/$i-$minus_log_lr-$minus_log_wd" | tee logs/$i-$minus_log_lr-$minus_log_wd &
        gpu_idx=$(($gpu_idx + 1))
        if [ $gpu_idx -eq ${gpu_ptr[$host_idx]} ]; then
            host_idx=$(($host_idx + 1))
        fi
        if [ $host_idx -eq ${#hosts[*]} ]; then
            wait
            host_idx=0
            gpu_idx=0
        fi
    done
done

for host in ${hosts[*]}; do
    scp -r ubuntu@$host:delta-graph/runs/* runs &
done
wait
rm -fr runs/null
