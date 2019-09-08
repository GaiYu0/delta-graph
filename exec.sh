hosts=(3.94.29.70 34.236.254.237 3.222.200.79 34.205.87.121)
for host in ${hosts[*]}; do
    ssh -n ubuntu@$host "$1";
done
wait
