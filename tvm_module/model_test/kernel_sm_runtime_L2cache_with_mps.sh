#bin/bash 模型在不同的资源分配下的内核参数，注意打开mps
modelname=$1
batchsize=$2
factor=$3

export MPSID=`echo get_server_list | nvidia-cuda-mps-control`
mpsid=$MPSID

for mps in 10 20 30 40 50 60 70 80 90 100
do
sleep 1
echo set_active_thread_percentage "$mpsid" $mps | nvidia-cuda-mps-control
sleep 1

ncu --target-processes all \
--metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct \
--kernel-name regex:tvmgen  -o ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache_with_mps_"$mps" -f \
python3 kernel_sm_runtime_L2cache_with_interference.py --model_name "$modelname" --batch_size "$batchsize" --factor "$factor" \
 --mps_set 0

ncu --import ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache_with_mps_"$mps".ncu-rep --csv > ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache_with_mps_"$mps".csv
rm ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache_with_mps_"$mps".ncu-rep
done