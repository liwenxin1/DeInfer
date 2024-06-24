#用于统计各种模型的kernel参数，包括kernel数目，内核执行时间，L2缓存命中率，L2缓存利用率，SM利用率
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json
#获取硬件参数
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
    pd_dram_utilization=pd_data[pd_data["Metric Name"]=="dram__throughput.avg.pct_of_peak_sustained_elapsed"].reset_index(drop=True)
    
    time_duaration_sum=pd_gpu_time_duration["Metric Value"].sum()
    mean_gpu_time_duration=pd_gpu_time_duration["Metric Value"].mean()
    pd_gpu_time_duration["Metric Value"]=pd_gpu_time_duration["Metric Value"]/time_duaration_sum
    mean_l2_hit_rate=(pd_l2_hit_rate["Metric Value"]*pd_gpu_time_duration["Metric Value"]).sum()
    mean_l2_utilization=(pd_l2_utilization["Metric Value"]*pd_gpu_time_duration["Metric Value"]).sum()
    mean_sm_efficiency=(pd_sm_efficiency["Metric Value"]*pd_gpu_time_duration["Metric Value"]).sum()
    mean_dram_utilization=(pd_dram_utilization["Metric Value"]*pd_gpu_time_duration["Metric Value"]).sum()
    kernel_num=pd_gpu_time_duration.shape[0]
    return kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency,mean_dram_utilization
#获取运行时参数
def get_model_runtime_info(model_name):
    dict1={}
    dict1[model_name]={}
    dict1[model_name]["latency"]={}
    dict2={}
    dict2[model_name]={}
    dict2[model_name]["throughput"]={}
    file_path="compare/gpulet/data/{}_mps_runtime.txt".format(model_name)
    with open(file_path,"r") as f:
        for line in f.readlines():
            modelname=line.split(" ")[0]
            batch_size=str(line.split(" ")[1].split(":")[1])
            mps_set=str(line.split(" ")[2].split(":")[1])
            latency=float(line.split(" ")[3].split(":")[1])
            if batch_size not in dict1[model_name]["latency"]:
                dict1[model_name]["latency"][batch_size]={}
                dict2[model_name]["throughput"][batch_size]={}
            
            dict1[model_name]["latency"][batch_size][mps_set]=latency*1000
            dict2[model_name]["throughput"][batch_size][mps_set]=int(batch_size)/latency
    return dict1,dict2

def get_model_hardware_info(model_name,batch_size_list,mps_set_list):
    dict1={}
    dict1[model_name]={}
    dict1[model_name]["l2cache"]={}
    dict2={}
    dict2[model_name]={}
    dict2[model_name]["dram"]={}
    
    for batch_size in batch_size_list:
        batch_size=str(batch_size)
        for mps_set in mps_set_list:
            mps_set=str(mps_set)
            csv_name="compare/gpulet/data/{}_{}_{}_kernel_sm_runtime_L2cache.csv".format(model_name,batch_size,mps_set)
            kernel_num,\
            mean_gpu_time_duration,\
            mean_l2_hit_rate,\
            mean_l2_utilization,\
            mean_sm_efficiency,\
            mean_dram_utilization=kernel_sm_runtime_L2cache_analyse(csv_name)
            if batch_size not in dict1[model_name]["l2cache"]:
                dict1[model_name]["l2cache"][batch_size]={}
                dict2[model_name]["dram"][batch_size]={}
            dict1[model_name]["l2cache"][batch_size][mps_set]=mean_l2_utilization
            dict2[model_name]["dram"][batch_size][mps_set]=mean_dram_utilization
            
    return dict1,dict2
        
            
    
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="")
    args=parser.parse_args()
    model_name=args.model_name
    
    
    mps_set_list=[10,20,40,50,60,80,100]
    batch_size_list=[1,2,4,8,16]
    latency_dict,throughput_dict=get_model_runtime_info(model_name)
    l2cache_dict,dram_dict=get_model_hardware_info(model_name,batch_size_list,mps_set_list)
    file_name="compare/gpulet/gpulet.config"
    with open(file_name,"a") as f:
        f.write(json.dumps(latency_dict)+'\n')
        f.write(json.dumps(throughput_dict)+'\n')
        f.write(json.dumps(l2cache_dict)+'\n')
        f.write(json.dumps(dram_dict)+'\n')
        
        
    