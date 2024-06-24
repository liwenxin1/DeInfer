#bin/bash 测试每一个kernel的运行时间，sm利用率，l2cache利用率，l2cache命中率
#example: bash kernel_sm_runtime_L2cache.sh resnet-18 1 200 
modelname=$1
batchsize=$2
factor=$3
ncu --target-processes all \
--metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct \
--kernel-name regex:tvmgen  -o ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache -f \
python3 kernel_sm_runtime_L2cache.py --model_name "$modelname" --batch_size "$batchsize" --factor "$factor"

ncu --import ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache.ncu-rep --csv > ./profiles/model_feature/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache.csv
rm ./profiles/"$modelname"_"$batchsize"_"$factor"_kernel_sm_runtime_L2cache.ncu-rep