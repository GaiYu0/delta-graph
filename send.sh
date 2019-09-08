hosts=($3)
if [ -z $hosts ]; then
    hosts=($(cat hosts))
fi

for host in ${hosts[*]}; do
    scp -r $1 ubuntu@$host:$2 &
done
wait
