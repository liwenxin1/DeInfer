#用于统计各种模型的kernel参数，包括kernel数目，内核执行时间，L2缓存命中率，L2缓存利用率，SM利用率
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
def kernel_sm_runtime_L2cache_analyse(file):
    pd_data=pd.read_csv(file)
    pd_data["Metric Value"] = pd_data["Metric Value"].astype(str)
    pd_data["Metric Value"] = pd_data["Metric Value"].str.replace(',', '')
    pd_data["Metric Value"] = pd_data["Metric Value"].astype(float)
    #这里的参数应该求基于运行时间的加权平均值
    pd_gpu_time_duration=pd_data[pd_data["Metric Name"]=="gpu__time_duration.sum"].reset_index(drop=True)
    pd_l2_hit_rate=pd_data[pd_data["Metric Name"]=="lts__t_sector_hit_rate.pct"].reset_index(drop=True)
    pd_l2_utilization=pd_data[pd_data["Metric Name"]=="lts__t_sectors.avg.pct_of_peak_sustained_elapsed"].reset_index(drop=True)
    pd_sm_efficiency=pd_data[pd_data["Metric Name"]=="sm__throughput.avg.pct_of_peak_sustained_elapsed"].reset_index(drop=True)
    
    time_duaration_sum=pd_gpu_time_duration["Metric Value"].sum()
    mean_gpu_time_duration=pd_gpu_time_duration["Metric Value"].mean()
    pd_gpu_time_duration["Metric Value"]=pd_gpu_time_duration["Metric Value"]/time_duaration_sum
    mean_l2_hit_rate=(pd_l2_hit_rate["Metric Value"]*pd_gpu_time_duration["Metric Value"]).sum()
    mean_l2_utilization=(pd_l2_utilization["Metric Value"]*pd_gpu_time_duration["Metric Value"]).sum()
    mean_sm_efficiency=(pd_sm_efficiency["Metric Value"]*pd_gpu_time_duration["Metric Value"]).sum()
    kernel_num=pd_gpu_time_duration.shape[0]
    return kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--file_name",type=str,default="")
    args=parser.parse_args()
    file_name=args.file_name
    
    file_name="./profiles/model_feature/vgg-13_2_untuned_kernel_sm_runtime_L2cache.csv"
    csv_name=file_name
    kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency=kernel_sm_runtime_L2cache_analyse(csv_name)
    print("kernel num              gpu time duration              l2 hit rate              l2 utilization              sm efficiency")
    # print("  {}                {}                {}          {}             {}".\
    #       format(kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency))
    print(kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency)