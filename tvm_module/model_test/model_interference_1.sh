#测试resnet-18与其他模型共置的干扰,干扰模型的资源分配比例为10%-%90%
modelname=$1
batch_size=$2
factor=$3
interference_model=$4
interference_batch_size=$5
interference_factor=$6


file_path="./profiles/exectime_txt/mps_"$modelname"-"$batch_size"-"$factor".txt"
if [ -f "$file_path" ]; then
    echo "$file_path exist"
else
    echo "$file_path not exist"
    for i in 10 20 30 40 50 60 70 80 90 100
    do
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$i python single_model_run.py \
    --model_name $modelname --batch_size $batch_size --factor $factor
    done > profiles/exectime_txt/mps_"$modelname"-"$batch_size"-"$factor".txt
fi

for i in 10 20 30 40 50 60 70 80 90
do
j=10
    while(($j<=$((100-$i))))
    do
    sleep 1
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$i python model_interference_test.py --mps_set $j \
    --model_name $modelname --batch_size $batch_size --factor $factor \
    --interference_model_name $interference_model --interference_batch_size $interference_batch_size --interference_factor $interference_factor
    j=$(($j+10))
    done
    echo "-------------------"
done > profiles/exectime_txt/interference_"$modelname"-"$batch_size"-"$factor"_"$interference_model"-"$interference_batch_size"-"$interference_factor".txt

