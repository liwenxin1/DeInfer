#获取所有模型的L2 cache 以及GFLOPS
from modelDeploy.dispatch.model_info import Model_Info,Model_Runtime_Info
from tvm_module.utils.utils import model_para_get
import os
import pandas as pd
model_info=Model_Info("modelDeploy/modelRepository")
model_runtime_info=Model_Runtime_Info("modelDeploy/modelRepository")

modelname_list=["shufflenet","resnet-18","resnet-34","resnet-50","resnet-101","resnet-152","vgg-11","vgg-13",\
        "vgg-16","vgg-19","squeezenet","mobilenet","transformer"]

def kernel_sm_runtime_L2cache_analyse(file):
    pd_data=pd.read_csv(file)
    pd_data["Metric Value"] = pd_data["Metric Value"].astype(str)
    pd_data["Metric Value"] = pd_data["Metric Value"].str.replace(',', '')
    pd_data["Metric Value"] = pd_data["Metric Value"].astype(float)
    
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
    return mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency

write_file="modelDeploy/modelProfile/profiles/model_L2_gflops.csv"
col_names=["modelname","batch","L2cache","GFLOPS"]
df=pd.DataFrame(columns=col_names)

for modelname in modelname_list:
    
    batch_support_list=model_info.get_model_batchsize_support(modelname)
    for batch in batch_support_list:
        if "resnet" in modelname:
            filename="tvm_module/model_test/profiles/model_feature/{}_{}_200_kernel_sm_runtime_L2cache.csv".format(modelname,batch)
        elif modelname=="squeezenet":
            filename="tvm_module/model_test/profiles/model_feature/{}_{}_untuned_kernel_sm_runtime_L2cache.csv".format("squeezenet-1.0",batch)
        elif modelname=="LSTM":
            filename="tvm_module/model_test/profiles/model_feature/{}_{}_untuned_kernel_sm_runtime_L2cache.csv".format("LSTM-2",batch)
        else:
            filename="tvm_module/model_test/profiles/model_feature/{}_{}_untuned_kernel_sm_runtime_L2cache.csv".format(modelname,batch)
            
        
        L2cache=kernel_sm_runtime_L2cache_analyse(filename)[2]
        gflops=model_para_get(modelname,batch)[0]/model_runtime_info.get_model_mps_runtime(modelname,batch,100)
        
       
        df=df._append(pd.Series([modelname,batch,L2cache,gflops],index=df.columns),ignore_index=True)
        
df.to_csv(write_file,index=False)




