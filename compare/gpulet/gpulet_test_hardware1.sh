modelname=$1
batchsize=$2
mps_set=$3

ncu --target-processes all \
--metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed \
--kernel-name regex:tvmgen  -o compare/gpulet/data/"$modelname"_"$batchsize"_"$mps_set"_kernel_sm_runtime_L2cache -f \
python3 compare/gpulet/gpulet_test_hardware.py --model_name "$modelname" --batch_size "$batchsize" --mps_set "$mps_set"

ncu --import compare/gpulet/data/"$modelname"_"$batchsize"_"$mps_set"_kernel_sm_runtime_L2cache.ncu-rep --csv > compare/gpulet/data/"$modelname"_"$batchsize"_"$mps_set"_kernel_sm_runtime_L2cache.csv
rm compare/gpulet/data/"$modelname"_"$batchsize"_"$mps_set"_kernel_sm_runtime_L2cache.ncu-rep