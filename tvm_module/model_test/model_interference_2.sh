#测试resnet-18与其他模型共置的干扰,干扰模型的资源分配比例为10%-%90%
modelname=$1
batch_size=$2
factor=$3
interference_model=$4
interference_batch_size=$5
interference_factor=$6





for i in 10 20 30 40 50 60
do
    mps=$((100-$i))
    for j in 1 2 3 #干扰模型的数量
    do
    sleep 1
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$i python model_interference_test_multi.py --mps_set $mps \
    --model_name $modelname --batch_size $batch_size --factor $factor \
    --interference_model_name $interference_model --interference_batch_size $interference_batch_size --interference_factor $interference_factor \
    --interference_num $j
    
    done
    echo "-------------------"
done > profiles/exectime_txt/multi-interference_"$modelname"-"$batch_size"-"$factor"_"$interference_model"-"$interference_batch_size"-"$interference_factor".txt

