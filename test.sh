test_list=("replica-kitchen" "replica-apartment0" "replica-apartment1" "replica-room0")

for i in "${test_list[@]}"  
do  
    # bash run/train.sh $i
    echo "start $i"
    # bash run/train.sh $i | tee -a train_$i.log
    bash run/train.sh $i
    echo "finish $i"
done