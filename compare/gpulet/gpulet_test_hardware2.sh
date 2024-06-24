modelname=$1
for batchsize in 1 2 4 8 16
do
    for mps_set in 10 20 40 50 60 80 100
    do
    bash compare/gpulet/gpulet_test_hardware1.sh $modelname $batchsize $mps_set
    done
done
