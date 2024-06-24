#bin/bash 测试每一个kernel的运行时间，sm利用率，l2cache利用率，l2cache命中率
modelname=$1
batchsize=$2
factor=$3
interference_model_name=$4
interference_batch_size=$5
interference_factor=$6
cuda_active_thread_percentage=$7
mps_set=$8

export MPSID=`echo get_server_list | nvidia-cuda-mps-control`

mpsid=$MPSID
sleep 1
echo set_active_thread_percentage "$mpsid" $cuda_active_thread_percentage | nvidia-cuda-mps-control
sleep 1

ncu --target-processes all \
--metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct \
--kernel-name regex:tvmgen  -o ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache_with_interference -f \
python3 kernel_sm_runtime_L2cache_with_interference.py --model_name "$modelname" --batch_size "$batchsize" --factor "$factor" \
--interference_model_name "$interference_model_name" --interference_batch_size "$interference_batch_size" --interference_factor "$interference_factor" --mps_set "$mps_set"

ncu --import ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache_with_interference.ncu-rep --csv > ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache_with_interference.csv
rm ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache_with_interference.ncu-rep