for batch_size in 1 2 4 8 16
do
    for factor in 50 100 200 400
    do
        bash kernel_sm_runtime_L2cache.sh resnet-18 $batch_size $factor
    done
done