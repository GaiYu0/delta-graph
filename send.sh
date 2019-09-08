hosts=(3.94.29.70 34.236.254.237 3.222.200.79 34.205.87.121)
for host in ${hosts[*]}; do
    scp -r $1 ubuntu@$host:$2 &
done
wait
