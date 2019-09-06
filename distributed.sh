hosts=(3.210.203.13 100.27.35.98)
gpu_ptr=($(seq 8 8 $((8 * ${#hosts[*]}))))
gpus=($(for i in $(seq 1 ${#hosts[*]}); do echo $(seq 0 7); done))

bs_train=100000
d=256
n_iters=1
p_train=0.8
p_val=0.1

host_idx=0
gpu_idx=0
for i in $(seq 0 0); do
    for minus_log_lr in $(seq 1 4); do
        for minus_log_wd in $(seq 1 4); do
            echo ${hosts[$host_idx]} ${gpus[$gpu_idx]}
            ssh -n ubuntu@host "source activate pytorch_p36; python3 train.py --bs-train $bs_train --ds ml-10m/partitions-100/partition-$i --gpu $gpu --model "BiasedMF(n_users, n_items, $d, r_mean)" --n-iters $n_iters --optim "Adam(model.parameters(), 1e-$minus_log_lr, weight_decay=1e-$minus_log_wd)" --p-train $p_train --p-val $p_val --logdir runs/bmf/$i-$minus_log_lr-$minus_log_wd" > logs/$i-$minus_log_lr-$minus_log_wd &
            gpu_idx=$(($gpu_idx + 1))
            if [ $gpu_idx -eq ${gpu_ptr[$host_idx]} ]; then
                host_idx=$(($host_idx + 1))
            fi
            if [ $host_idx -eq ${#hosts[*]} ]; then
                wait
                for host in ${hosts[*]}; do
                    scp -r ubuntu@$host:delta-graph/runs/* runs &
                done
                wait
                host_idx=0
                gpu_idx=0
            fi
        done
    done
done
