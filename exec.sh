hosts=($2)
if [ -z $hosts ]; then
    hosts=($(cat hosts))
fi

for host in ${hosts[*]}; do
    ssh -n ubuntu@$host "$1";
done
wait
