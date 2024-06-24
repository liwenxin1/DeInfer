modelname=$1
batch_size=$2
factor=$3

file_path="./profiles/exectime_txt/mps_"$modelname"-"$batch_size"-"$factor".txt"
if [ -f "$file_path" ]; then
    echo " "
else
    echo "$file_path not exist"
    for i in 10 20 30 40 50 60 70 80 90 100
    do
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$i python single_model_run.py \
    --model_name $modelname --batch_size $batch_size --factor $factor
    done > profiles/exectime_txt/mps_"$modelname"-"$batch_size"-"$factor".txt
fi