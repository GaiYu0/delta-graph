gpu=0
n_gpus=8
for lr in 1e-1 1e-2 1e-3 1e-4; do
    for wd in 1e-1 1e-2 1e-3 1e-4; do
        ipython3 train_auto_rec.py -- -d 1024 -f "lambda x: x" -g "th.sigmoid" --ds ml-1m --gpu $gpu --lr $lr --model auto_rec.IAutoRec --n-iters 250 --opt optim.Adam --p-train 0.8 --p-val 0.1 --path $lr-$wd --wd $wd &
        gpu=$(((gpu + 1) % $n_gpus))
        if [[ gpu -eq 0 ]]; then
            wait
        fi
    done
done
